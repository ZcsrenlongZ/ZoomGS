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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys

from scene import Scene
from scene.gaussian_model import GaussianModel

from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

import torch.nn as nn

import random
import numpy as np

def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from

    warmup_iter = 4999
    if args.stage == "uw_pretrain":
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, shuffle=False, get_all_cam="all")
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

        viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_stack_uw = viewpoint_stack[0 : len(viewpoint_stack)//2]
        viewpoint_stack_wide = viewpoint_stack[len(viewpoint_stack)//2 : ]

        ema_loss_for_log = 0.0
        first_iter += 1
        for iteration in range(first_iter, warmup_iter + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cam = viewpoint_stack_uw[(randint(0, len(viewpoint_stack_uw)-1))]
            
            if (iteration - 1) == debug_from:
                pipe.debug = True

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, info=None)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 =  l1_loss(image, gt_image)
            loss1 = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

            loss = loss1

            loss.backward()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                if (iteration == warmup_iter):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                if iteration < opt.densify_until_iter:  
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True) 

    elif args.stage == "uw2wide":
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(args, gaussians, load_iteration=warmup_iter, shuffle=False, get_all_cam="all")
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

        viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_stack_uw = viewpoint_stack[0 : 10]
        viewpoint_stack_wide = viewpoint_stack[10 : ]

        ema_loss_for_log = 0.0
        first_iter += 1

        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cam = viewpoint_stack_uw[(randint(0, len(viewpoint_stack_uw)-1))]
            viewpoint_cam_wide = viewpoint_stack_wide[(randint(0, len(viewpoint_stack_wide)-1))]
            
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # uw reconstruction loss
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, info=None)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 =  l1_loss(image, gt_image)
            loss1 = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

            # w reconstruction loss
            render_pkg_wide = render(viewpoint_cam_wide, gaussians, pipe, background, info={"c":1., "target":"cx"})
            image_wide = render_pkg_wide["render"] 
            viewspace_point_tensor, visibility_filter, radii = render_pkg_wide["viewspace_points"], render_pkg_wide["visibility_filter"], render_pkg_wide["radii"]
            gt_image_wide = viewpoint_cam_wide.original_image.cuda()
            loss2 = ((1.0 - opt.lambda_dssim) * l1_loss(image_wide, gt_image_wide) + opt.lambda_dssim * (1.0 - ssim(image_wide, gt_image_wide)))
            
            # identity regulation
            gt_image = viewpoint_cam.original_image.cuda()
            render_pkg_uwc0 = render(viewpoint_cam, gaussians, pipe, background, info={"c":0., "target":"cx"})
            deta_x = render_pkg_uwc0["deta_x"]
            deta_c = render_pkg_uwc0["deta_c"]
            loss3 = torch.abs(deta_x).mean() + torch.abs(deta_c).mean()
            
            # lip regulation
            loss4 = gaussians._mlp.get_lipschitz_loss() * 1e-7

            loss = 0.5*loss1 + 0.5*loss2 + 1.*loss3 + loss4

            loss.backward()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                                testing_iterations, scene, render, (pipe, background))

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                if iteration < opt.densify_until_iter:  
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                if iteration < opt.iterations:
                    # optimize base gs
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                   # optimize camTrans module
                    gaussians.mlp_optimizer.step()
                    gaussians.mlp_optimizer.zero_grad(set_to_none = True)
                    gaussians.mlp_scheduler.step()               

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--stage", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)



    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")