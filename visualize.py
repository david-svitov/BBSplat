import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from utils.general_utils import build_rotation
from utils.reconstruction_utils import GaussianExtractor

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    control_panel = cv2.imread("assets/control_panel.png")[..., ::-1].astype(np.float32) / 255.

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, texture_preproc=True)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, additional_return=False)

    speed_data = {"points": len(gaussians.get_xyz)}

    idx = 0
    cameras = scene.getTestCameras()[idx: idx+1].copy()
    frame_num = 0
    while True:
        mean_time, std_time = gaussExtractor.reconstruction(cameras)
        render = gaussExtractor.rgbmaps[0].detach().cpu().numpy()
        render = np.transpose(render, (1, 2, 0)).copy()
        if frame_num == 0:
            scale = render.shape[1] / control_panel.shape[1]
            control_panel = cv2.resize(control_panel, None, fx=scale, fy=scale)

        if frame_num > 5:
            mean_time = int(mean_time)
            cv2.putText(render, 'FPS: ' + str(mean_time), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 3, 2)
            cv2.putText(render, 'FPS: ' + str(mean_time),(10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1,(255, 255, 255),1,2)

        render = cv2.vconcat([render,  control_panel])
        cv2.imshow("Render", render[..., ::-1])
        key = cv2.waitKey(-1) & 0b11111111

        speed_t = 0.03
        speed_r = speed_t / 2.0
        if key == ord("q"):
            break
        if key == ord("a"):
            cameras[0].world_view_transform[3, 0] += speed_t
        if key == ord("d"):
            cameras[0].world_view_transform[3, 0] -= speed_t
        if key == ord("w"):
            cameras[0].world_view_transform[3, 2] -= speed_t
        if key == ord("s"):
            cameras[0].world_view_transform[3, 2] += speed_t
        if key == ord("e"):
            cameras[0].world_view_transform[3, 1] += speed_t
        if key == ord("f"):
            cameras[0].world_view_transform[3, 1] -= speed_t

        if key == ord("j"):
            R = build_rotation(torch.tensor([[1-speed_r, -speed_r, 0, 0]]).cuda())[0]
            cameras[0].world_view_transform[:3, :3] = torch.mm(cameras[0].world_view_transform[:3, :3], R)
            cameras[0].world_view_transform[3:, :3] = torch.matmul(cameras[0].world_view_transform[3:, :3], R)
        if key == ord("u"):
            R = build_rotation(torch.tensor([[1-speed_r, speed_r, 0, 0]]).cuda())[0]
            cameras[0].world_view_transform[:3, :3] = torch.mm(cameras[0].world_view_transform[:3, :3], R)
            cameras[0].world_view_transform[3:, :3] = torch.matmul(cameras[0].world_view_transform[3:, :3], R)
        if key == ord("k"):
            R = build_rotation(torch.tensor([[1-speed_r, 0, speed_r, 0]]).cuda())[0]
            cameras[0].world_view_transform[:3, :3] = torch.mm(cameras[0].world_view_transform[:3, :3], R)
            cameras[0].world_view_transform[3:, :3] = torch.matmul(cameras[0].world_view_transform[3:, :3], R)
        if key == ord("h"):
            R = build_rotation(torch.tensor([[1-speed_r, 0, -speed_r, 0]]).cuda())[0]
            cameras[0].world_view_transform[:3, :3] = torch.mm(cameras[0].world_view_transform[:3, :3], R)
            cameras[0].world_view_transform[3:, :3] = torch.matmul(cameras[0].world_view_transform[3:, :3], R)
        if key == ord("l"):
            R = build_rotation(torch.tensor([[1-speed_r, 0, 0, speed_r]]).cuda())[0]
            cameras[0].world_view_transform[:3, :3] = torch.mm(cameras[0].world_view_transform[:3, :3], R)
            cameras[0].world_view_transform[3:, :3] = torch.matmul(cameras[0].world_view_transform[3:, :3], R)
        if key == ord("i"):
            R = build_rotation(torch.tensor([[1-speed_r, 0, 0, -speed_r]]).cuda())[0]
            cameras[0].world_view_transform[:3, :3] = torch.mm(cameras[0].world_view_transform[:3, :3], R)
            cameras[0].world_view_transform[3:, :3] = torch.matmul(cameras[0].world_view_transform[3:, :3], R)

        if key == 32:
            idx += 1
            if idx >= len(scene.getTestCameras()):
                idx = 0
            cameras = scene.getTestCameras()[idx: idx+1].copy()

        cameras[0].update_proj_matrix()
        frame_num += 1

