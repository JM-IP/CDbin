#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/5 16:02
# @Author  : Jianming Ip
# @Site    : 
# @File    : networks_CDbin_structure_v2.py
# @Company : VMC Lab in Peking University


import torch
from torch.autograd import Variable
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

class KSP(nn.Module):
    """KSP model definition
    """
    def __init__(self):
        super(KSP, self).__init__()
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
            # nn.Dropout(0.3),
            # nn.Conv2d(128, 128, kernel_size=8, bias=False),
            # nn.BatchNorm2d(128, affine=False),
        )
        self.k=16
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        y = self.features(input_norm(input))
        x = y.view(y.size(0), y.size(1), -1)
        torch.zeros([y.size(0), y.size(1), self.k])
        for i in range(x.size(0)):
            tmpx=x[i,:,:].view(y.size(1),-1)
            U, S, VT = tmpx.svd()
            # print(tmpx.size(),U.size(),S.size(),VT.size())
            new_U = U[:tmpx.size(0), :self.k]
            new_VT = VT[:self.k, :(tmpx.size(-1))]
            SD = Variable(torch.eye(self.k)).cuda() * S[:self.k]
            # print("SD.size is",SD.size())
            # print("new_VT.size is", new_VT.size())
            b = torch.mm(SD, new_VT)
            newData = torch.mm(new_U, b)
            # print("newData size is:", newData.size())
        return L2Norm()(newData)

class CDbin_NET_deep2(nn.Module):
    """CDbin_NET_deep2 model definition
    """
    def __init__(self,fcnum):
        super(CDbin_NET_deep2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, fcnum, kernel_size=8, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)


class CDbin_NET_deep3(nn.Module):
    """CDbin_NET_deep3 model definition
    """
    def __init__(self,fcnum):
        super(CDbin_NET_deep3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, fcnum, kernel_size=8, bias=True),
            # nn.Linear(256 * 8 * 8, fcnum, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)

class CDbin_NET_deep3_2(nn.Module):
    """CDbin_NET_deep3_2 model definition
    """
    def __init__(self,fcnum):
        super(CDbin_NET_deep3_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, fcnum, kernel_size=8, bias=True),
            # nn.Linear(256 * 8 * 8, fcnum, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)

class CDbin_NET_deep4_1(nn.Module):
    """CDbin_NET_deep4_1 model definition
    """
    def __init__(self,fcnum):
        super(CDbin_NET_deep4_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, fcnum, kernel_size=8, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return


    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)


class CDbin_NET_deep4_2(nn.Module):
    """CDbin_NET_deep4_2 model definition
    """
    def __init__(self,fcnum):
        super(CDbin_NET_deep4_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, fcnum, kernel_size=4, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)


class CDbin_NET_deep5_1(nn.Module):
    """CDbin_NET_deep5_1 model definition
    """

    def __init__(self, fcnum):
        super(CDbin_NET_deep5_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2
                      , bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, fcnum, kernel_size=8, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)

class CDbin_NET_deep5_2(nn.Module):
    """CDbin_NET_deep5_2 model definition
    """

    def __init__(self, fcnum):
        super(CDbin_NET_deep5_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, fcnum, kernel_size=4, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return

    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        return L2Norm()(x)
