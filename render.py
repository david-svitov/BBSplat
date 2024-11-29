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
import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene
from utils.reconstruction_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, texture_preproc=True)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color, additional_return=False)

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
