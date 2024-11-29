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
import math
import os

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import torchvision

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import mkdir_p


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.color_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_color_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, texture_preproc: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._texture_alpha = torch.empty(0)
        self._texture_color = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.split_criterion_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._texture_color_lr = 0
        self._texture_is_active = False
        self._gaussians_is_active = True
        self._texture_preproc = texture_preproc
        self.setup_functions()

        image = cv2.imread("assets/alpha_init_gaussian_small.png")[..., 0] / 255.0
        self._texture_alpha_init = torch.tensor([image], dtype=torch.float, device="cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._texture_alpha,
            self._texture_color,
            self.max_radii2D,
            self.split_criterion_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation,
        self._texture_alpha,
        self._texture_color,
        self.max_radii2D,
        split_criterion_gradient_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.split_criterion_gradient_accum = split_criterion_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_texture_status(self):
        return self._texture_is_active

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_texture_alpha(self):
        if self._texture_preproc:
            return self._texture_alpha
        else:
            return self.opacity_activation(self._texture_alpha)

    @property
    def get_texture_color(self):
        if self._texture_preproc:
            return self._texture_color
        else:
            return self.color_activation(self._texture_color)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def fibonacci_sphere(self, samples=1000):
        # https://arxiv.org/pdf/0912.4540

        points = []
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append((x, y, z))

        return np.asarray(points)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, add_sky_box: bool):
        self.spatial_lr_scale = spatial_lr_scale
        pcd_points = np.asarray(pcd.points)
        radius = np.max(np.abs(pcd_points))
        sky_box_points = self.fibonacci_sphere(10000) * radius
        pcd_colors = np.asarray(pcd.colors)

        if add_sky_box:
            total_points = np.concatenate([sky_box_points, pcd_points], axis=0)
            total_colors = np.concatenate([np.ones_like(sky_box_points), pcd_colors], axis=0)
            print("Add sky box")
        else:
            total_points = pcd_points
            total_colors = pcd_colors

        fused_point_cloud = torch.tensor(total_points).float().cuda()
        RGB_points = torch.tensor(total_colors)
        fused_color = RGB2SH(RGB_points.float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color - 0.1
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(total_points).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Setup trainable alpha textures
        num_points = fused_point_cloud.shape[0]
        texture_alpha = self._texture_alpha_init.repeat(num_points, 1, 1)
        texture_alpha = self.inverse_opacity_activation(0.5 * texture_alpha)
        self._texture_alpha = nn.Parameter(texture_alpha.requires_grad_(True))

        # Setup trainable RGB textures
        texture_size = texture_alpha.shape[-1]
        texture_color = torch.ones([num_points, 3, texture_size, texture_size]) * 0.1
        texture_color = texture_color.type(torch.float)
        texture_color = self.inverse_color_activation(texture_color.to("cuda"))
        self._texture_color = nn.Parameter(texture_color.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.split_criterion_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._texture_alpha], 'lr': 0, "name": "texture_alpha"},
            {'params': [self._texture_color], 'lr': 0, "name": "texture_color"},
        ]
        self._texture_opacity_lr = training_args.texture_opacity_lr
        self._texture_color_lr = training_args.texture_color_lr

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def compress_texture(self, texture_alpha, texture_color):
        texture_color = self.color_activation(texture_color)
        texture_alpha = self.opacity_activation(texture_alpha)

        texture_alpha = texture_alpha - self._texture_alpha_init
        texture_alpha = (texture_alpha + 1) * 0.5

        texture_color = texture_color.detach().cpu().numpy()
        texture_alpha = texture_alpha.detach().cpu().numpy()

        texture_alpha = np.round(texture_alpha * 255).astype(np.uint8)
        texture_color = np.round(texture_color * 255).astype(np.uint8)

        return texture_alpha, texture_color

    def decompress_texture(self, texture_alpha, texture_color):
        texture_alpha = torch.tensor(texture_alpha, dtype=torch.float, device="cuda") / 255

        texture_alpha = (texture_alpha * 2 - 1) + self._texture_alpha_init
        texture_alpha = torch.clip(texture_alpha, 0, 1)

        # Visualization of bbsplats
        #texture_alpha[texture_alpha < 0.5] = 0
        #texture_alpha[texture_alpha >= 0.5] = 1

        texture_alpha = self.inverse_opacity_activation(texture_alpha)
        texture_alpha = nn.Parameter(texture_alpha.requires_grad_(True))

        texture_color = torch.tensor(texture_color, dtype=torch.float, device="cuda") / 255
        texture_color = self.inverse_color_activation(texture_color)
        texture_color = nn.Parameter(texture_color.requires_grad_(True))

        return texture_alpha, texture_color

    def save_texture(self, folder_path):
        texture_alpha, texture_color = self.compress_texture(self._texture_alpha, self._texture_color)
        np.savez_compressed(os.path.join(folder_path, "texture_alpha.npz"), texture_alpha=texture_alpha)
        np.savez_compressed(os.path.join(folder_path, "texture_color.npz"), texture_color=texture_color)

        indeces = np.random.randint(0, len(self.get_texture_color), 16*16)

        texture_grid = torchvision.utils.make_grid(self.get_texture_alpha[indeces, None, ...], 16)
        texture_grid = texture_grid.detach().cpu().numpy()[0]
        cv2.imwrite(os.path.join(folder_path, "texture_alpha.png"), texture_grid * 255)

        rgb = SH2RGB(self.get_features[indeces, 0]) - 0.5
        rgb = rgb[..., None, None]
        texutes = self.get_texture_color[indeces, ...] + rgb
        texture_grid = torchvision.utils.make_grid(texutes, 16)
        texture_grid = texture_grid.detach().cpu().numpy()
        texture_grid = np.transpose(texture_grid, (1, 2, 0))
        cv2.imwrite(os.path.join(folder_path, "texture_color.png"), texture_grid[..., ::-1] * 255)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_texture(self, folder_path):
        data = np.load(os.path.join(folder_path, "texture_alpha.npz"))
        texture_alpha = data["texture_alpha"]
        data = np.load(os.path.join(folder_path, "texture_color.npz"))
        texture_color = data["texture_color"]

        self._texture_alpha, self._texture_color = self.decompress_texture(texture_alpha, texture_color)

        if self._texture_preproc:
            self._texture_alpha = self.opacity_activation(self._texture_alpha)
            self._texture_color = self.color_activation(self._texture_color)

        # Uncomment for visualization of Billboards shapes
        # self._texture_color = torch.zeros_like(self._texture_color)
        # self._texture_alpha[torch.amax(self._texture_alpha, dim=(1, 2)) < 0.1] = 0
        # self._texture_alpha *= 2

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print(xyz.shape)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # Random color initialization
        # ========================================
        # features = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2))
        # np.random.seed(123)
        # features[:, :3, 0] = np.random.uniform(-1, 1, (xyz.shape[0], 3))
        # features[:, 3:, 1:] = 0.0
        # features_dc = features[:, :, 0:1]
        # features_extra = features[:, :, 1:]
        # ========================================

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def activate_texture_training(self):
        if self._texture_is_active:
            return
        for group in self.optimizer.param_groups:
            if group["name"] == "texture_alpha":
                group["lr"] = self._texture_opacity_lr
            if group["name"] == "texture_color":
                group["lr"] = self._texture_color_lr
        self._texture_is_active = True
        print("Texture is activated")

    def deactivate_texture_training(self):
        if not self._texture_is_active:
            return
        for group in self.optimizer.param_groups:
            if group["name"] == "texture_alpha":
                group["lr"] = 0
            if group["name"] == "texture_color":
                group["lr"] = 0
        self._texture_is_active = False
        print("Texture is frozen")

    def deactivate_gaussians_training(self):
        if not self._gaussians_is_active:
            return
        for group in self.optimizer.param_groups:
            if group["name"] not in ["f_rest"]:
                group["lr"] = 0
        self._gaussians_is_active = False
        print("Gaussians are frozen")


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0).contiguous()
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0).contiguous()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).contiguous().requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).contiguous().requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_color_tex, new_alpha_tex):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "texture_color": new_color_tex,
        "texture_alpha": new_alpha_tex}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._texture_alpha = optimizable_tensors["texture_alpha"]
        self._texture_color = optimizable_tensors["texture_color"]

        self.split_criterion_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
                        "f_dc": self._features_dc,
                        "f_rest": self._features_rest,
                        "scaling": self._scaling,
                        "rotation": self._rotation,
                        "texture_color": self._texture_color,
                        "texture_alpha": self._texture_alpha,
                        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._texture_color = optimizable_tensors["texture_color"]
        self._texture_alpha = optimizable_tensors["texture_alpha"]

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_scaling = self._scaling[idxs].clone()

        N_idx = ratio[idxs, :, None] + 1
        opacity_old = self.get_texture_alpha[idxs]
        new_alpha_textures = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / N_idx)

        new_alpha_textures = self.inverse_opacity_activation(new_alpha_textures)

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_scaling, \
        self._rotation[idxs], self._texture_color[idxs], new_alpha_textures

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        size = len(alive_indices)
        probs = self.get_texture_alpha[alive_indices].view(size, -1).mean(1)
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices],
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
            self._texture_color[dead_indices],
            self._texture_alpha[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._texture_alpha[reinit_idx] = self._texture_alpha[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._scaling.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        size = len(self.get_texture_alpha)
        probs = self.get_texture_alpha.view(size, -1).mean(1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_scaling,
            new_rotation,
            new_texture_color,
            new_texture_alpha,
        ) = self._update_params(add_idx, ratio=ratio)

        self._texture_alpha[add_idx] = new_texture_alpha
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation,
                                   new_texture_color, new_texture_alpha)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs
