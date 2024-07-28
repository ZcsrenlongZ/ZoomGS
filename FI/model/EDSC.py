import torch
import torchvision
import cv2
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torchvision.models as models
import math
import sys
import torch.nn.functional as F
from model import dsepconv



class Efficient_HetConv2d(nn.Module):
    def __init__(self, in_feats, out_feats, p=4, ks=3, pad=1):
        super(Efficient_HetConv2d, self).__init__()
        if in_feats % p != 0:
            raise ValueError('in_channels must be divisible by p')
        if out_feats % p != 0:
            raise ValueError('out_channels must be divisible by p')
        self.conv3x3 = nn.Conv2d(in_feats, out_feats, kernel_size=ks, padding=pad, groups=p)
        self.conv1x1_ = nn.Conv2d(in_feats, out_feats, kernel_size=1, groups=p)
        self.conv1x1 = nn.Conv2d(in_feats, out_feats, kernel_size=1)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x) - self.conv1x1_(x)


class Network(nn.Module):
    '''
    Network of EDSC
    '''
    def __init__(self, Generated_ks=5, Het_p=4, useBias=True, isMultiple=False):
        super(Network, self).__init__()
        self.het_p = Het_p
        self.generated_ks = Generated_ks
        self.useBias = useBias
        self.isMultiple = isMultiple
        if self.isMultiple:
            self.estimator_in = 65
        else:
            self.estimator_in = 64

        def Basic(intInput, intOutput, ks, pad):
            return torch.nn.Sequential(
                Efficient_HetConv2d(in_feats=intInput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=ks, pad=pad),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Efficient_HetConv2d(in_feats=intOutput, out_feats=intOutput, p=self.het_p, ks=3, pad=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def KernelNet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1)
                )
        # end

        def Offsetnet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1)
                )


        def Masknet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=self.estimator_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3,
                             stride=1, padding=1),
                    torch.nn.Sigmoid()
                )
                

        def Biasnet():
            return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
                )
                

        self.moduleConv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=32, out_feats=32, p=self.het_p, ks=3, pad=1),
                torch.nn.ReLU(inplace=False),
                Efficient_HetConv2d(in_feats=32, out_feats=32, p=self.het_p, ks=3, pad=1),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv2 = Basic(32, 64, 3, 1)
        self.moduleConv3 = Basic(64, 128, 3, 1)
        self.moduleConv4 = Basic(128, 256, 3, 1)
        self.moduleConv5 = Basic(256, 512, 3, 1)

        self.moduleDeconv5 = Basic(512, 512, 3, 1)
        self.moduleDeconv4 = Basic(512, 256, 3, 1)
        self.moduleDeconv3 = Basic(256, 128, 3, 1)
        self.moduleDeconv2 = Basic(128, 64, 3, 1)

        self.moduleUpsample5 = Upsample(512, 512)
        self.moduleUpsample4 = Upsample(256, 256)
        self.moduleUpsample3 = Upsample(128, 128)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleVertical1 = KernelNet()
        self.moduleVertical2 = KernelNet()
        self.moduleHorizontal1 = KernelNet()
        self.moduleHorizontal2 = KernelNet()

        self.moduleOffset1x = Offsetnet()
        self.moduleOffset1y = Offsetnet()
        self.moduleOffset2x = Offsetnet()
        self.moduleOffset2y = Offsetnet()

        self.moduleMask1 = Masknet()
        self.moduleMask2 = Masknet()

        self.moduleBias = Biasnet()
        # self.moduleOffset1x.register_backward_hook(self._set_lr)

    # end
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, tensors):
        tensorFirst = tensors[0]
        tensorSecond = tensors[1]
        if self.isMultiple:
            tensorTime = tensors[2]

        tensorConv1 = self.moduleConv1(torch.cat([tensorFirst, tensorSecond], 1))
        tensorConv2 = self.moduleConv2(torch.nn.functional.avg_pool2d(input=tensorConv1, kernel_size=2, stride=2))
        tensorConv3 = self.moduleConv3(torch.nn.functional.avg_pool2d(input=tensorConv2, kernel_size=2, stride=2))
        tensorConv4 = self.moduleConv4(torch.nn.functional.avg_pool2d(input=tensorConv3, kernel_size=2, stride=2))
        tensorConv5 = self.moduleConv5(torch.nn.functional.avg_pool2d(input=tensorConv4, kernel_size=2, stride=2))

        tensorDeconv5 = self.moduleUpsample5(
            self.moduleDeconv5(torch.nn.functional.avg_pool2d(input=tensorConv5, kernel_size=2, stride=2)))
        tensorDeconv4 = self.moduleUpsample4(self.moduleDeconv4(tensorDeconv5 + tensorConv5))
        tensorDeconv3 = self.moduleUpsample3(self.moduleDeconv3(tensorDeconv4 + tensorConv4))
        tensorDeconv2 = self.moduleUpsample2(self.moduleDeconv2(tensorDeconv3 + tensorConv3))

        tensorCombine = tensorDeconv2 + tensorConv2

        tensorFirst = torch.nn.functional.pad(input=tensorFirst,
                                              pad=[int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0)),
                                                   int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0))],
                                              mode='replicate')
        tensorSecond = torch.nn.functional.pad(input=tensorSecond,
                                               pad=[int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0)),
                                                    int(math.floor(5 / 2.0)), int(math.floor(5 / 2.0))],
                                               mode='replicate')


        if self.isMultiple:
            v1 = self.moduleVertical1(torch.cat([tensorCombine, tensorTime], 1))
            v2 = self.moduleVertical2(torch.cat([tensorCombine, 1. - tensorTime], 1))
            h1 = self.moduleHorizontal1(torch.cat([tensorCombine, tensorTime], 1))
            h2 = self.moduleHorizontal2(torch.cat([tensorCombine, 1. - tensorTime], 1))

            tensorDot1 = dsepconv.FunctionDSepconv(tensorFirst, v1, h1,
                                                   self.moduleOffset1x(torch.cat([tensorCombine, tensorTime], 1)),
                                                   self.moduleOffset1y(torch.cat([tensorCombine, tensorTime], 1)),
                                                   self.moduleMask1(torch.cat([tensorCombine, tensorTime], 1)))
            tensorDot2 = dsepconv.FunctionDSepconv(tensorSecond, v2, h2,
                                                   self.moduleOffset2x(torch.cat([tensorCombine, 1. - tensorTime], 1)),
                                        self.moduleOffset2y(torch.cat([tensorCombine, 1. - tensorTime], 1)),
                                        self.moduleMask2(torch.cat([tensorCombine, 1. - tensorTime], 1)))
        else:
            v1 = self.moduleVertical1(tensorCombine)
            v2 = self.moduleVertical2(tensorCombine)
            h1 = self.moduleHorizontal1(tensorCombine)
            h2 = self.moduleHorizontal2(tensorCombine)
            offset1x = self.moduleOffset1x(tensorCombine)
            offset1y = self.moduleOffset1y(tensorCombine)
            offset2x = self.moduleOffset2x(tensorCombine)
            offset2y = self.moduleOffset2y(tensorCombine)
            mask1 = self.moduleMask1(tensorCombine)
            mask2 = self.moduleMask2(tensorCombine)

            tensorDot1 = dsepconv.FunctionDSepconv(tensorFirst, v1, h1, offset1x, offset1y, mask1)
            tensorDot2 = dsepconv.FunctionDSepconv(tensorSecond, v2, h2, offset2x, offset2y, mask2)

        if self.useBias:
                return tensorDot1 + tensorDot2 + self.moduleBias(tensorCombine)
        else:
                return tensorDot1 + tensorDot2
            

    # end forward
# end class
