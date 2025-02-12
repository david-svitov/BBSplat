#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#
import os
from functools import partial
from statistics import mean, stdev

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.render_utils import save_img_u8


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None, additional_return=True):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background, additional_return=additional_return)
        self._additional_return = additional_return
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []
        self.times = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        times = []
        if len(self.viewpoint_stack) > 1:
            iterator = tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields")
        else:
            iterator = enumerate(self.viewpoint_stack)

        for i, viewpoint_cam in iterator:
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            times.append(render_pkg['fps'])
            rgb = render_pkg['render']
            self.rgbmaps.append(rgb.cpu())
            if self._additional_return:
                alpha = render_pkg['rend_alpha']
                normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
                depth = render_pkg['surf_depth']
                depth_normal = render_pkg['surf_normal']
                self.depthmaps.append(depth.cpu())
                self.alphamaps.append(alpha.cpu())
                self.normals.append(normal.cpu())
                self.depth_normals.append(depth_normal.cpu())

        self.times = times
        mean_time = mean(times)
        std_time = 0
        if len(times) > 1:
            std_time = stdev(times)
            print("FPS:", mean_time, " std:", std_time)
        #self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        if self._additional_return:
            self.depthmaps = torch.stack(self.depthmaps, dim=0)
            self.alphamaps = torch.stack(self.alphamaps, dim=0)
            self.depth_normals = torch.stack(self.depth_normals, dim=0)

        return mean_time, std_time

    @torch.no_grad()
    def export_image(self, path, export_gt=True, print_fps=False):
        render_path = os.path.join(path, "renders")
        os.makedirs(render_path, exist_ok=True)
        if export_gt:
            gts_path = os.path.join(path, "gt")
            os.makedirs(gts_path, exist_ok=True)

        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            if export_gt:
                gt = viewpoint_cam.original_image[0:3, :, :]
                save_img_u8(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

            image = self.rgbmaps[idx]
            if print_fps:
                fps = '{:4d}'.format(int(self.times[idx]))
                image = image.numpy()
                image = np.transpose(image, (1, 2, 0)).copy()
                cv2.putText(image, 'FPS: ' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 3, 2)
                cv2.putText(image, 'FPS: ' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (1, 1, 1), 1, 2)
                image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image)
            save_img_u8(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
