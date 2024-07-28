import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import shutil
import math
import os
import cv2
import torch
import lpips
import os.path as osp
import numpy as np
from model.FI_models.EDSCVgg import Model as EDSC
from model.FI_models.IFRNetVgg import Model as IFRNet
from model.FI_models.RIFEVgg import Model as RIFE
from model.FI_models.AMTVgg import Model as AMT
from model.FI_models.UPRNetVgg import Model as UPRNet
from model.FI_models.EMAVFIVgg import Model as EMAVFI
from argparse import ArgumentParser
import glob
import lpips
import nriqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate(I0, I1, timestep):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    mid = model.inference(I0, I1, timestep=timestep)[0]
    mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    mid = (mid * 255.).astype(np.uint8)
    imgs.append(mid)
    return imgs

def crop_center(hr, ph, pw):
    ih, iw = hr.shape[0:2]
    lr_patch_h, lr_patch_w = ph, pw
    ph = ih // 2 - lr_patch_h // 2
    pw = iw // 2 - lr_patch_w // 2

    return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

def lpips_norm(img, range):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (range / 2.) - 1
	return torch.Tensor(img).to(device)

def calc_lpips(out, target, loss_fn_alex, range):
	lpips_out = lpips_norm(out, range)
	lpips_target = lpips_norm(target, range)
	LPIPS = loss_fn_alex(lpips_out, lpips_target)
	return LPIPS.detach().cpu().item()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="RIFE", help="FI model (EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI)")
    parser.add_argument("--log_dir", type=str, default="./ckpt/RIFE_finetuned", help="log path")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/DCSZ_dataset/DCSZ_real", help="train data path")
    parser.add_argument('--save_dir', type=str, default='./real_results/', help='where to save image results')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.log_dir, args.save_dir)
    

    if args.model == "EDSC":
        model = EDSC()
        model.load_model(args.log_dir, suffix='99', rank=-1)
    elif args.model == "IFRNet":
        model = IFRNet()
        model.load_model(args.log_dir, suffix='99', rank=-1)
    elif args.model == "RIFE":
        model = RIFE()
        model.load_model(args.log_dir, suffix='99', rank=-1)
    elif args.model == "AMT":
        model = AMT()
        model.load_model(args.log_dir, suffix='99', rank=-1)
    elif args.model == "UPRNet":
        model = UPRNet()
        model.load_model(args.log_dir, suffix='99', rank=-1)
    elif args.model == "EMAVFI":
        model = EMAVFI()
        model.load_model(args.log_dir, suffix='99', rank=-1)
    else:
        print("Warning: unsupported FI model.")
        exit(2)

    model.eval()
    model.device()

    data_dir = os.path.join(args.dataset_dir, "test")
    ids = os.listdir(data_dir)
    for id in ids:
        files = sorted(glob.glob(os.path.join(os.path.join(data_dir, id), '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0]))
        print(os.path.join(data_dir, id), len(files))
        uw_image = cv2.imread(files[0])
        wide_image = cv2.imread(files[-1])
        os.makedirs(args.save_dir, exist_ok=True)
        
        I1 = wide_image

        shape = uw_image.shape

        start = 1
        end = 32

        scale = 0.85  / 0.6  
        I0 = cv2.resize(uw_image, (int(uw_image.shape[1]*scale), int(uw_image.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        I0 = crop_center(I0, shape[0], shape[1])


        save_img = I0
        os.makedirs(os.path.join(args.save_dir, id), exist_ok=True)
        save_path = osp.join(args.save_dir, id, '{:03d}.png'.format(start - 1))
        cv2.imwrite(save_path, save_img)

        save_img = I1
        os.makedirs(os.path.join(args.save_dir, id), exist_ok=True)
        save_path = osp.join(args.save_dir, id, '{:03d}.png'.format(end))
        cv2.imwrite(save_path, save_img)


        for ii in range(start, end, 1):
            timestep = ii / 32
            
            gif_imgs = [I0, I1]

            gif_imgs_temp = [gif_imgs[0], ]
            for i, (img_start, img_end) in enumerate(zip(gif_imgs[:-1], gif_imgs[1:])):
                interp_imgs = interpolate(img_start, img_end, timestep=timestep)
                gif_imgs_temp += interp_imgs
                gif_imgs_temp += [img_end, ]
            gif_imgs = gif_imgs_temp

            save_img = gif_imgs[1]

            save_path = osp.join(args.save_dir, id, '{:03d}.png'.format(ii))
            cv2.imwrite(save_path, save_img)
    nriqa.main(args.save_dir, 0)





    

