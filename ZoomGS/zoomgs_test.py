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
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene#, GaussianModel
from scene.gaussian_model import GaussianModel
import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth

from utils.image_utils import psnr
from utils.loss_utils import ssim
import numpy as np
from lpipsPyTorch import lpips

import utils.Spline as Spline
import numpy as np
import torch
from scene.cameras import Camera


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import torch
import math
import torch.nn as nn

import utils.Spline as Spline
import numpy as np
import torch
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args, refine=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    log_dir = os.path.join(model_path, name, "ours_{}".format(iteration), "metrics.txt")
    f = open(log_dir, 'a')

    uw_psnr_list = []
    uw_ssim_list = []
    uw_lpips_list = []

    wide_psnr_list = []
    wide_ssim_list = []
    wide_lpips_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if view.image_name.split('_')[0] == "uw":
            rendering = render(view, gaussians, pipeline, background, info={"c":0.0, "target":args.target})
            gt = view.original_image[0:3, :, :]

            render_image = rendering["render"].clamp(0., 1.).unsqueeze(0)
            gt = gt.clamp(0., 1.).unsqueeze(0)
            psnr_metric = psnr(render_image, gt).mean().double().item()
            ssim_metric = ssim(render_image, gt).mean().double().item()
            lpips_metric = lpips(render_image, gt, net_type='vgg').mean().double().item()

            uw_psnr_list.append(psnr_metric)
            uw_ssim_list.append(ssim_metric)
            uw_lpips_list.append(lpips_metric)

        if view.image_name.split('_')[0] == "w":
            rendering = render(view, gaussians, pipeline, background, info={"c":1.0, "target":args.target})
            gt = view.original_image[0:3, :, :]

            render_image = rendering["render"].clamp(0., 1.).unsqueeze(0)
            gt = gt.clamp(0., 1.).unsqueeze(0)
            psnr_metric = psnr(render_image, gt).mean().double().item()
            ssim_metric = ssim(render_image, gt).mean().double().item()
            lpips_metric = lpips(render_image, gt, net_type='vgg').mean().double().item()

            wide_psnr_list.append(psnr_metric)
            wide_ssim_list.append(ssim_metric)
            wide_lpips_list.append(lpips_metric)

        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

    print('UW: PSNR:', np.mean(np.array(uw_psnr_list)), 'SSIM:', np.mean(np.array(uw_ssim_list)), 'LPIPS:', np.mean(np.array(uw_lpips_list)))
    print('Wide: PSNR:', np.mean(np.array(wide_psnr_list)), 'SSIM:', np.mean(np.array(wide_ssim_list)), 'LPIPS:', np.mean(np.array(wide_lpips_list)))
    f.write('UW : PSNR:\t' + str(np.mean(np.array(uw_psnr_list))) + '\t'+ str(np.mean(np.array(uw_ssim_list))) + '\t'+str(np.mean(np.array(uw_lpips_list))) +'\n')
    f.write('Wide : PSNR:\t' + str(np.mean(np.array(wide_psnr_list))) + '\t'+str(np.mean(np.array(wide_ssim_list))) + '\t'+str(np.mean(np.array(wide_lpips_list))) +'\n')
    f.flush()
    f.close()

def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False, get_all_cam="all")
        refine = None

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args, refine)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_depth", action="store_true")

    parser.add_argument("--target", default="", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)