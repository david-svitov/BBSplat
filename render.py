#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import math
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
import torch.nn.functional as F
import xatlas
from pytorch3d.io import save_obj
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from utils.general_utils import build_scaling_rotation
from utils.loss_utils import l1_loss
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.render_utils import generate_path, create_videos, save_img_u8
from utils.sh_utils import SH2RGB


def unwrap_uvmap(mesh, device="cuda"):
    v_np = np.asarray(mesh.vertices)  # [N, 3]
    f_np = np.asarray(mesh.triangles)  # [M, 3]

    print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

    # unwrap uv in contracted space
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
    pack_options = xatlas.PackOptions()
    # pack_options.blockAlign = True
    # pack_options.bruteForce = False
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

    print("UV shape:", vt.shape)

    v_torch = torch.from_numpy(v_np.astype(np.float32)).to(device)
    f_torch = torch.from_numpy(f_np).to(device)

    return v_torch, f_torch, vt, ft

def render_mesh(v_torch, f_torch, uv, uv_idx, cudactx, texture, cam):
    mvp = cam.full_proj_transform
    vertices_clip = torch.matmul(F.pad(v_torch, pad=(0, 1), mode='constant', value=1.0), mvp).float().unsqueeze(0)
    rast, _ = dr.rasterize(cudactx, vertices_clip, f_torch, resolution=[cam.image_height, cam.image_width])
    texc, _ = dr.interpolate(uv[None, ...], rast, uv_idx)
    color = dr.texture(texture[None, ...], texc, filter_mode='linear')[0]
    return color

def train_texture(v_torch, f_torch, uv, uv_idx, cudactx, texture, scene):
    optimizer = torch.optim.Adam([texture], lr=0.01)
    for epoch in tqdm(range(300)):
        for cam in scene.getTrainCameras():
            optimizer.zero_grad()
            color = render_mesh(v_torch, f_torch, uv, uv_idx, cudactx, F.sigmoid(texture), cam)

            gt = torch.permute(cam.original_image.cuda(), (1, 2, 0))
            Ll1 = l1_loss(color, gt)
            #ssim_map = ssim(color, gt, size_average=False).mean()
            loss = Ll1  # * 0.8 + ssim_map * 0.2
            loss.backward()
            optimizer.step()

def billboard_to_plane(xyz, transform, rgb, alpha, texture_size, num_textures_x, vertices, faces, stitched_texture, uv, uv_idx):
    vertices_local = torch.tensor([[-1, -1, 0], [1, 1, 0], [1, -1, 0], [-1, 1, 0]], dtype=torch.float32).cuda()
    faces_local = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int32).cuda()
    
    # Scaling + Rotation
    vertices_local = vertices_local @ transform.T
    # Offset 
    vertices_local += xyz
    
    # Add to the "mesh"
    faces_local += 4 * len(faces)
    faces.append(faces_local)
    vertices.append(vertices_local) 
    
    # Add tile to the texture
    num = len(vertices) - 1
    y = num // num_textures_x
    x = num % num_textures_x
    h, w = alpha.shape
    stitched_texture[:3, y*texture_size: y*texture_size + h, x*texture_size: x*texture_size + w] = rgb
    stitched_texture[3:, y*texture_size: y*texture_size + h, x*texture_size: x*texture_size + w] = alpha[None]
    
    u = x*texture_size / stitched_texture.shape[2]
    v = y*texture_size / stitched_texture.shape[1]
    offset_u = h / stitched_texture.shape[2]
    offset_v = w / stitched_texture.shape[1]
    uv_local = torch.tensor([[u, v], [u + offset_u, v + offset_v], [u + offset_u, v], [u, v + offset_v]], dtype=torch.float32).cuda()
    uv.append(uv_local)
    uv_idx.append(faces_local)
    
def billboards_to_mesh(gaussians, save_folder):
    num_points = len(gaussians.get_xyz)
    gaps = 2
    texture_size = gaussians.get_texture_alpha.shape[-1] + gaps
    num_textures_x = int(math.sqrt(num_points))
    globa_texture_size = num_textures_x * texture_size
    global_rgba = torch.zeros([4, globa_texture_size + texture_size*2, globa_texture_size]).cuda()
    
    transform = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
    
    vertices = [] 
    faces = []
    uv = [] 
    uv_idx = []
    for i in tqdm(range(num_points)):
        #if gaussians.get_scaling[i].min() > 1:
        #    continue 
        billboard_to_plane(
            gaussians.get_xyz[i], transform[i], gaussians.get_texture_color[i] + SH2RGB(gaussians.get_features_first[i])[0, :, None, None], 
            gaussians.get_texture_alpha[i], texture_size, num_textures_x,
            vertices, faces, global_rgba, uv, uv_idx,
        )
    vertices = torch.concat(vertices)
    faces = torch.concat(faces)
    uv = torch.concat(uv)
    uv_idx = torch.concat(uv_idx)
    
    print(vertices.shape, faces.shape)
    
    global_rgba = torch.permute(global_rgba, (1, 2, 0))
    global_rgba = torch.flip(global_rgba, [0])
    save_obj(
        os.path.join(save_folder, "planes_mesh.obj"),
        verts=vertices,
        faces=faces,
        verts_uvs=uv,
        faces_uvs=uv_idx,
        texture_map=global_rgba[..., :3],
    )
    print(global_rgba.shape)
    global_rgba = global_rgba.detach().cpu().numpy()
    global_rgba[..., :3] = cv2.cvtColor(global_rgba[..., :3], cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_folder, "planes_mesh.png"), global_rgba * 255)

def prune_based_on_visibility(scene, gaussians, pipe, background):
    with torch.no_grad():
        # Calculate impact
        acc_impact = None
        for camera in scene.getTrainCameras():
            render_pkg = render(camera, gaussians, pipe, background)
            impact = render_pkg["impact"]
            if acc_impact is None:
                acc_impact = impact
            else:
                acc_impact += impact

        prob = acc_impact / acc_impact.sum()
        mask = prob > 1e-6

        mask = mask & (torch.amax(gaussians.get_texture_alpha, dim=(1, 2)) > 0.2)
        gaussians.prune_postproc(mask)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--save_planes", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=0.004, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=3.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=1000, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, texture_preproc=True)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, additional_return=True)

    speed_data = {"points": len(gaussians.get_xyz)}
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        mean_time, std_time = gaussExtractor.reconstruction(scene.getTrainCameras())
        speed_data["train_time"] = mean_time
        speed_data["train_time_std"] = std_time
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        mean_time, std_time = gaussExtractor.reconstruction(scene.getTestCameras())
        speed_data["test_time"] = mean_time
        speed_data["test_time_std"] = std_time
        gaussExtractor.export_image(test_dir)

    with open(os.path.join(args.model_path, "speed.json"), "w") as f:
        json.dump(speed_data, f)

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 480
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir, export_gt=False) #, print_fps=True
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if args.save_planes:
        # CONVERT TO SET OF PLANES
        prune_based_on_visibility(scene, gaussians, pipe, background)
        billboards_to_mesh(gaussians, args.model_path)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        print("ckpt 1 ...")
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            #mesh = gaussExtractor.extract_mesh_bounded(voxel_size=args.voxel_size, sdf_trunc=5*args.voxel_size, depth_trunc=args.depth_trunc)
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        print("ckpt 2 ...")
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

        # TEXTURE EXTRACTION
        device = "cuda"
        # Unwrap the uv-map for the mesh
        v_cuda, f_cuda, uv, uv_idx = unwrap_uvmap(mesh, device)

        texture = 0.5 + torch.randn((1024, 1024, 3), dtype=torch.float32, device=device) * 0.001
        texture = torch.nn.Parameter(texture, requires_grad=True)

        cudactx = dr.RasterizeCudaContext()

        # Train texture from input images
        train_texture(v_cuda, f_cuda, uv, uv_idx, cudactx, texture, scene)
        texture = F.sigmoid(texture)

        # Render textured mesh to the folder
        mesh_path = os.path.join(train_dir, "mesh")
        os.makedirs(mesh_path, exist_ok=True)
        for idx, cam in enumerate(scene.getTrainCameras()):
            mvp = cam.full_proj_transform
            color = render_mesh(v_cuda, f_cuda, uv, uv_idx, cudactx, texture, cam)
            color = torch.permute(color, (2, 0, 1))
            save_img_u8(color, os.path.join(mesh_path, '{0:05d}'.format(idx) + ".png"))

        save_obj(
            os.path.join(args.model_path, "textured_mesh.obj"),
            verts=v_cuda,
            faces=f_cuda,
            verts_uvs=uv,
            faces_uvs=uv_idx,
            texture_map=torch.flip(texture, [0]),
        )
