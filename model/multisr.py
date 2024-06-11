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
from option import args


def make_model(args, parent=False):
    return MULTISR(args)


class HighwayEncoder(nn.Module):
    """
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for transform, gate in zip(self.transforms, self.gates):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            h = torch.sigmoid(transform(x))
            z = torch.sigmoid(gate(x))
            x = z * h + (1 - z) * x

        return x


# SK卷积自动选择卷积核
class SKConv(nn.Module):
    def __init__(self, features, WH=1, M=2, G=1, r=2, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class multi_block(nn.Module):
    def __init__(self, n_feats):
        super(multi_block, self).__init__()
        # self.inchanels = in_channel
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=1, padding=2)
        #self.highencoder = HighwayEncoder(3, 64)
        self.skconv = SKConv(features=n_feats)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv3(y)
        y = self.conv2(y)
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.conv3(z)
        w = self.conv1(x)
        w = self.conv2(w)
        w = self.conv2(w)
        #e = self.highencoder(x)
        output = F.relu(y + w + z + x)
        output = self.skconv(output)
        return F.relu(output + x)


class MULTISR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MULTISR, self).__init__()

        n_multi_blocks = args.n_multi_blocks
        n_feats = args.n_feats
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = nn.Sequential(
            conv(args.n_colors, 3, 7),
            nn.ReLU(),
            conv(3, n_feats, 3),
            nn.ReLU()
        )

        # define body module
        m_body = [
            multi_block(n_feats) for _ in range(n_multi_blocks)
        ]
        m_body.append(conv(n_feats, n_feats, 3))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, 3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x



