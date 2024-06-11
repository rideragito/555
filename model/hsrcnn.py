import functools
import torch.nn.functional as F
import torch
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
from model import common


def make_model(args, parent=False):
    return HSRCNN(args)


class res_block1(nn.Module):
    '''定义了带实线部分的残差块'''

    def __init__(self, in_channels, out_channels):
        super(res_block1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        y = F.prelu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class res_block2(nn.Module):
    '''定义了带虚线部分的残差块'''

    def __init__(self, in_channels, out_channels):
        super(res_block2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = F.prelu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)


class HSRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HSRCNN, self).__init__()
        scale = args.scale[0]
        n_feats = args.n_feats
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, 3)]

        m_body = [res_block1(n_feats, n_feats), res_block2(n_feats, n_feats)]

        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
