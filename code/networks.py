#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/17 13:40
# @Author  : Jianming Ip
# @Site    :
# @File    : get_homepath.py
# @Company : VMC Lab in Peking University

import torch
from torch import nn
from torch.nn import functional as F
from Utils import L2Norm
from torch.nn import init
# add by yjm 2018.04.17   :for using channel-wise pooling
# add by yjm 2018.05.20   :for all kinds of networks
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

def input_norm(x):
    flat = x.view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + 1e-7
    return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(
        -1).unsqueeze(-1).unsqueeze(1).expand_as(x)

def eye_init(m):
    if isinstance(m, nn.Linear):
        for i in range(m.weight.data.size(0)):
            m.weight.data[i][i]=1
        try:
            nn.init.constant(m.bias.data, 0.0)
        except:
            pass
    return


class LayerNorm(nn.Module):
    def __init__(self, features_len, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features_len))
        self.beta = nn.Parameter(torch.zeros(features_len))
        self.eps = eps

    def forward(self, x):
        # print("x",x.size())
        mean = x.mean(-1, keepdim=True)
        # print("mean",mean)
        std = torch.sqrt((x*x).sum(-1, keepdim=True))
        return self.gamma * (x - mean.expand(std.size(0),x.size(1))) / (std.expand(std.size(0),x.size(1)) + self.eps) + self.beta


class HardNet(nn.Module):
    """HardNet model definition
    """

    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)
        return

    def forward(self, input):
        x_features = self.features(input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)

class CDbin_NET(nn.Module):
    """CDbin_NET model definition
    """

    def __init__(self):
        super(CDbin_NET, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
        )
        self.features1 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256, bias=True),
            LayerNorm(256),
            nn.Dropout(p=0.5),
        )
        self.features2 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        self.features2.apply(eye_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features.view(x_features.size(0), -1))
        y = self.features2(y)
        y = y*20
        x = y.view(y.size(0), -1)
        return L2Norm()(x)
