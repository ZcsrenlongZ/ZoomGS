# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp, sqrt
from torch.nn import L1Loss, MSELoss
from torchvision import models


def normalize_batch(batch):
	mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
	std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
	return (batch - mean) / std
	
class VGG19(torch.nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		features = models.vgg19(pretrained=True).features
		self.relu1_1 = torch.nn.Sequential()
		self.relu1_2 = torch.nn.Sequential()

		self.relu2_1 = torch.nn.Sequential()
		self.relu2_2 = torch.nn.Sequential()

		self.relu3_1 = torch.nn.Sequential()
		self.relu3_2 = torch.nn.Sequential()
		self.relu3_3 = torch.nn.Sequential()
		self.relu3_4 = torch.nn.Sequential()

		self.relu4_1 = torch.nn.Sequential()
		self.relu4_2 = torch.nn.Sequential()
		self.relu4_3 = torch.nn.Sequential()
		self.relu4_4 = torch.nn.Sequential()

		self.relu5_1 = torch.nn.Sequential()
		self.relu5_2 = torch.nn.Sequential()
		self.relu5_3 = torch.nn.Sequential()
		self.relu5_4 = torch.nn.Sequential()

		for x in range(2):
			self.relu1_1.add_module(str(x), features[x])

		for x in range(2, 4):
			self.relu1_2.add_module(str(x), features[x])

		for x in range(4, 7):
			self.relu2_1.add_module(str(x), features[x])

		for x in range(7, 9):
			self.relu2_2.add_module(str(x), features[x])

		for x in range(9, 12):
			self.relu3_1.add_module(str(x), features[x])

		for x in range(12, 14):
			self.relu3_2.add_module(str(x), features[x])

		for x in range(14, 16):
			self.relu3_3.add_module(str(x), features[x])

		for x in range(16, 18):
			self.relu3_4.add_module(str(x), features[x])

		for x in range(18, 21):
			self.relu4_1.add_module(str(x), features[x])

		for x in range(21, 23):
			self.relu4_2.add_module(str(x), features[x])

		for x in range(23, 25):
			self.relu4_3.add_module(str(x), features[x])

		for x in range(25, 27):
			self.relu4_4.add_module(str(x), features[x])

		for x in range(27, 30):
			self.relu5_1.add_module(str(x), features[x])

		for x in range(30, 32):
			self.relu5_2.add_module(str(x), features[x])

		for x in range(32, 34):
			self.relu5_3.add_module(str(x), features[x])

		for x in range(34, 36):
			self.relu5_4.add_module(str(x), features[x])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		relu1_1 = self.relu1_1(x)
		relu1_2 = self.relu1_2(relu1_1)

		relu2_1 = self.relu2_1(relu1_2)
		relu2_2 = self.relu2_2(relu2_1)

		relu3_1 = self.relu3_1(relu2_2)
		relu3_2 = self.relu3_2(relu3_1)
		relu3_3 = self.relu3_3(relu3_2)
		relu3_4 = self.relu3_4(relu3_3)

		relu4_1 = self.relu4_1(relu3_4)
		relu4_2 = self.relu4_2(relu4_1)
		relu4_3 = self.relu4_3(relu4_2)
		relu4_4 = self.relu4_4(relu4_3)

		relu5_1 = self.relu5_1(relu4_4)
		relu5_2 = self.relu5_2(relu5_1)
		relu5_3 = self.relu5_3(relu5_2)
		relu5_4 = self.relu5_4(relu5_3)

		out = {
			'relu1_1': relu1_1,
			'relu1_2': relu1_2,

			'relu2_1': relu2_1,
			'relu2_2': relu2_2,

			'relu3_1': relu3_1,
			'relu3_2': relu3_2,
			'relu3_3': relu3_3,
			'relu3_4': relu3_4,

			'relu4_1': relu4_1,
			'relu4_2': relu4_2,
			'relu4_3': relu4_3,
			'relu4_4': relu4_4,

			'relu5_1': relu5_1,
			'relu5_2': relu5_2,
			'relu5_3': relu5_3,
			'relu5_4': relu5_4,
		}
		return out


class SWDLoss(nn.Module):
	def __init__(self):
		super(SWDLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = SWD()
		# self.SWD = SWDLocal()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		N, C, H, W = x.shape  # 192*192
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		swd_loss = 0.0
		swd_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'], k=H//4//p) * 1  # H//4=48
		swd_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'], k=H//8//p) * 1  # H//4=24
		swd_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'], k=H//16//p) * 2  # H//4=12

		return swd_loss #* 8 / 100.0

class SWD(nn.Module):
	def __init__(self):
		super(SWD, self).__init__()
		self.l1loss = torch.nn.L1Loss() 

	def forward(self, fake_samples, true_samples, k=0):
		N, C, H, W = true_samples.shape

		num_projections = C//2

		true_samples = true_samples.view(N, C, -1)
		fake_samples = fake_samples.view(N, C, -1)

		projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
		projections = torch.FloatTensor(projections).to(true_samples.device)
		projections = F.normalize(projections, p=2, dim=1)

		projected_true = projections @ true_samples
		projected_fake = projections @ fake_samples

		sorted_true, true_index = torch.sort(projected_true, dim=2)
		sorted_fake, fake_index = torch.sort(projected_fake, dim=2)
		return self.l1loss(sorted_true, sorted_fake).mean() 
	

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = nn.L1Loss()

    def forward(self, img1, img2):
        x = normalize_batch(img1)
        y = normalize_batch(img2)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0

        content_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2']) * 1
        content_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2']) * 1
        content_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2']) * 2

        return content_loss / 4.