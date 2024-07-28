# -*- coding: utf-8 -*-

import glob
import os
from PIL import Image
from tqdm import tqdm
import torch
import sys
import cv2
import numpy as np
from collections import OrderedDict
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY


class InferenceModel(torch.nn.Module):
    """Common interface for quality inference of images with default setting of each metric."""

    def __init__(
            self,
            metric_name,
            as_loss=False,
            loss_weight=None,
            loss_reduction='mean',
            device=None,
            **kwargs  # Other metric options
    ):
        super(InferenceModel, self).__init__()

        self.metric_name = metric_name

        # ============ set metric properties ===========
        self.lower_better = DEFAULT_CONFIGS[metric_name].get('lower_better', False)
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', None)
        if self.metric_mode is None:
            self.metric_mode = kwargs.pop('metric_mode')
        elif 'metric_mode' in kwargs:
            kwargs.pop('metric_mode')

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.as_loss = as_loss
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction

        # =========== define metric model ===============
        net_opts = OrderedDict()
        # load default setting first
        if metric_name in DEFAULT_CONFIGS.keys():
            default_opt = DEFAULT_CONFIGS[metric_name]['metric_opts']
            net_opts.update(default_opt)
        # then update with custom setting
        net_opts.update(kwargs)
        network_type = net_opts.pop('type')
        self.net = ARCH_REGISTRY.get(network_type)(**net_opts)
        self.net = self.net.to(self.device)
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        self.device = torch.device(device)
        return self

    def forward(self, target, ref=None, **kwargs):
        with torch.set_grad_enabled(self.as_loss):
            if 'fid' in self.metric_name:
                output = self.net(target, ref, device=self.device, **kwargs)
            else:
                if not torch.is_tensor(target):
                    target = imread2tensor(target)
                    target = target.unsqueeze(0)
                    if self.metric_mode == 'FR':
                        assert ref is not None, 'Please specify reference image for Full Reference metric'
                        ref = imread2tensor(ref)
                        ref = ref.unsqueeze(0)
                if self.metric_mode == 'FR':
                    output = self.net(target.to(self.device), ref.to(self.device), **kwargs)
                elif self.metric_mode == 'NR':
                    output = self.net(target.to(self.device), **kwargs)
        return output


def imread2tensor(img):
    img_tensor = torch.from_numpy(np.float32(img).transpose(2, 0, 1) / 255.)
    return img_tensor


def main(input_dir, device):
	device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

	# set up IQA model
	iqa_model_niqe = InferenceModel(metric_name='niqe', metric_mode='NR', device=device)
	iqa_model_pi = InferenceModel(metric_name='pi', metric_mode='NR', device=device)
	iqa_model_clip = InferenceModel(metric_name='clipiqa', metric_mode='NR', device=device)
	iqa_model_mus = InferenceModel(metric_name='musiq', metric_mode='NR', device=device)

	print('niqe', iqa_model_niqe.lower_better)
	print('pi', iqa_model_pi.lower_better)
	print('clip', iqa_model_clip.lower_better)
	print('mus', iqa_model_mus.lower_better)

	input_file = input_dir
	save_file = os.path.join(input_dir, 'real_metrics.txt')

	if os.path.isfile(input_file):
		input_paths = [input_file]
	else:
		input_dir = os.path.join(input_file, '*', '*.png')
		input_paths = sorted(glob.glob(input_dir, recursive = True))

	sf = open(save_file, 'a')
	sf.write(f'input address:\t{input_file}\n')
	p = sf.tell()

	avg_score_niqe = 0
	avg_score_pi = 0
	avg_score_clip = 0
	avg_score_mus = 0
	test_img_num = len(input_paths)
	tqdm_input_paths = tqdm(input_paths)

	for idx, img_path in enumerate(tqdm_input_paths):
		img_name = os.path.basename(img_path)
		tar_img = cv2.imread(img_path, -1)[..., ::-1]

		H, W, C = tar_img.shape

		ref_img = None

		pre_img_niqe = 0
		pre_img_pi = 0
		pre_img_clip = 0
		pre_img_mus = 0

		img = tar_img

		try:
			score_niqe = iqa_model_niqe(img, ref_img)
			pre_img_niqe += score_niqe
			torch.cuda.empty_cache()
		except Exception:
			pass

		score_pi = iqa_model_pi(img, ref_img)
		pre_img_pi += score_pi
		torch.cuda.empty_cache()
		
		score_clip = iqa_model_clip(img, ref_img)
		pre_img_clip += score_clip
		torch.cuda.empty_cache()

		score_mus = iqa_model_mus(img, ref_img)
		pre_img_mus += score_mus
		torch.cuda.empty_cache()

		avg_score_niqe += pre_img_niqe 
		avg_score_pi += pre_img_pi 
		avg_score_clip += pre_img_clip 
		avg_score_mus += pre_img_mus 

		# print(f'{metric_name} score of {img_name} is: {score}')
		sf.write('%s  \t niqe: %.3f, \t pi: %.3f, \t clipiqa: %.4f, \t musiq: %.3f\n' % 
		         (img_name, pre_img_niqe, pre_img_pi, pre_img_clip , pre_img_mus))


	avg_score_niqe /= test_img_num
	avg_score_pi /= test_img_num
	avg_score_clip /= test_img_num
	avg_score_mus /= test_img_num

	print('Average niqe score with %s images is: %.3f \n' % (test_img_num, avg_score_niqe))
	print('Average pi score with %s images is: %.3f \n' % (test_img_num, avg_score_pi))
	print('Average clipiqa score with %s images is: %.4f \n' % (test_img_num, avg_score_clip))
	print('Average musiq score with %s images is: %.3f \n' % (test_img_num, avg_score_mus))

	sf.seek(p)
	sf.write('Average niqe score with %s images is: %.3f \n' % (test_img_num, avg_score_niqe))
	sf.write('Average pi score with %s images is: %.3f \n' % (test_img_num, avg_score_pi))
	sf.write('Average clipiqa score with %s images is: %.4f \n' % (test_img_num, avg_score_clip))
	sf.write('Average musiq score with %s images is: %.3f \n' % (test_img_num, avg_score_mus))
	sf.close()

if __name__ == '__main__':
	with torch.no_grad():
		main()

# pip install git+https://github.com/chaofengc/IQA-PyTorch.git
# Ubuntu >= 18.04
# Python >= 3.8
# Pytorch >= 1.8.1
# CUDA >= 10.1 (if use GPU)
# 缺少 version python set_up.py develop