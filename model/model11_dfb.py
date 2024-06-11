import torch.nn as nn
import torch
import os
from torch.nn import functional as F
from model import common
from model import esa

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d,self).__init__()
    def forward(self, x):
        return nn.AvgPool1d(x,kernel_size=x.shape[2])

class block1(nn.Module):
    def __init__(self,inchan=64,outchan=64,conv=common.default_conv):
        super(block1, self).__init__()
        self.conv0 = conv(in_channels=inchan,out_channels=outchan,kernel_size=1)
        self.conv1 = conv(in_channels=outchan,out_channels=outchan,kernel_size=1)
        self.dwconv0 = common.dwconv(inchan=outchan,outchan=outchan,ker2=7)
        self.conv2 = conv(in_channels=outchan,out_channels=outchan,kernel_size=1)
        self.sigmod = nn.Sigmoid()
        self.conv3 = conv(in_channels=outchan,out_channels=outchan,kernel_size=1)
        self.dwconv1 = common.dwconv(inchan=outchan,outchan=outchan,ker2=3)
        self.dwconv2 = common.dwconv(inchan=outchan,outchan=outchan,ker2=3)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.conv4 = conv(in_channels=outchan*2,out_channels=outchan,kernel_size=1)
    def forward(self,x):
        x = self.conv0(x)
        x0 = self.sigmod(self.conv3(self.dwconv0(self.conv1(x)))) * x
        x0 = self.dwconv1(x0)
        x1 = self.dwconv2(self.conv2(x))
        x1 = self.gelu(x1)

        return x1 + x0

class block0(nn.Module):
    def __init__(self,n_feats,conv=default_conv,dwconv=common.DepthWiseConv):
        super(block0, self).__init__()
        self.pool1 = GlobalAvgPool1d()
        self.pool2 = GlobalAvgPool1d()


        #self.stride_conv0 = conv(n_feats, n_feats, kernel_size=3, stride=2, padding=0)  # strided conv
        #self.stride_conv1 = conv(n_feats, n_feats, kernel_size=3, stride=2, padding=0)

        self.conv_begin = conv(n_feats,n_feats,kernel_size=1)
        # self.conv0 = dwconv(n_feats, n_feats)
        # self.conv1 = dwconv(n_feats, n_feats)
        # self.conv2 = dwconv(n_feats, n_feats)

        self.conv0 = block2(n_feats, n_feats)
        self.conv1 = block2(n_feats, n_feats)
        self.conv2 = block2(n_feats, n_feats)

        self.conv_end = conv(n_feats*3,n_feats,1)


        self.gelu = nn.GELU()
    def forward(self,x):
        x = self.conv_begin(x)
        b1 = self.gelu(self.conv0(x))

        b2 = F.adaptive_max_pool2d(b1, (x.size(2)//2,x.size(3)//2))  # pooling
        b2 = self.gelu(self.conv1(b2))

        b3 = F.adaptive_max_pool2d(b2, (x.size(2) // 4, x.size(3) // 4))  # pooling
        b3 = self.gelu(self.conv2(b3))

        b2 = F.interpolate(b2, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)  # upsample

        # b3 = F.interpolate(b2, (x.size(2)*4, x.size(3)*4), mode='bilinear', align_corners=True)  # upsample
        # b3 = self.gelu(self.conv2(b3))
        # p_size = (x.size(2),x.size(3))
        # b3 = F.adaptive_max_pool2d(b3, p_size)  # pooling


        b3 = F.interpolate(b3, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)  # upsample

        m = self.conv_end(torch.cat([b1,b2,b3],dim=1))
        return m

class block2(nn.Module):
    def __init__(self,inchan,outchan,dwconv=common.DepthWiseConv):
        super(block2, self).__init__()
        self.conv0 = dwconv(inchan,outchan)
        self.conv1 = dwconv(outchan,outchan)
        self.conv2 = dwconv(outchan,outchan)
        self.gelu = nn.GELU()
    def forward(self,x):
        x = self.gelu(self.conv0(x))
        x0 = self.conv1(x)
        x1 = self.gelu(self.conv2(x0))
        return x1


# class ChannelMLP(nn.Module):
#     def __init__(self, dim, growth_rate=2.0):
#         super().__init__()
#         hidden_dim = int(dim * growth_rate)
#
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 1, 1, 0),
#             nn.GELU(),
#             nn.Conv2d(hidden_dim, dim, 1, 1, 0)
#         )
#
#     def forward(self, x):
#         return self.mlp(x)
class base_block0(nn.Module):
    def __init__(self,dim=64,conv=common.default_conv):
        super(base_block0, self).__init__()
        self.norm0 = LayerNorm(dim)
        self.norm1 = LayerNorm(dim)
        self.block0 = block0(n_feats=dim)
        self.block1 = block1(dim)
        self.conv = conv(dim,dim,1)
        #self.MLP = ChannelMLP(dim=dim)
    def forward(self,x):
        #c1 = self.norm0(x)
        x0 = self.block0(x) + x
        x1 = self.block1(x0) + x0
        return self.conv(x1)
class base_block1(nn.Module):
    def __init__(self,nfeats=64,dwconv=common.DepthWiseConv,conv=default_conv):
        super(base_block1, self).__init__()
        # self.base_block0 = nn.Sequential(
        #     base_block0(nfeats,nfeats),
        #     Myattn(nfeats),
        #     basic_block(nfeats,nfeats)
        #
        # )
        self.base0 = base_block0(nfeats)
        self.base1 = base_block0(nfeats)
        self.base2 = base_block0(nfeats)
        self.base3 = base_block0(nfeats)


        self.attn0 = esa.ESA(nfeats)

        self.attn2 = esa.ESA(nfeats)
        self.conv_begin = dwconv(in_channel=nfeats,out_channel=nfeats)
        #self.conv_begin = conv(in_channels=inchan,out_channels=outchan,kernel_size=3)
        self.conv0 = conv(nfeats, nfeats,1)
        self.conv1 = conv(nfeats, nfeats, 1)
        self.conv2 = conv(nfeats, nfeats, 1)
        self.conv3 = conv(nfeats*4, nfeats, 1)
        self.conv_end = dwconv(nfeats,nfeats)

    def forward(self,x):
        x0 = self.base0(x)
        x0 = self.base1(x0)
        x0 = self.base2(x0)
        x0 = self.base3(x0)
        out = self.attn0(x0,x0)

        #x = self.conv_begin(x)

        # c = self.conv0(x)
        # x0 = self.base_block0(x)
        # c0 = self.conv1(x0)
        # x1 = self.base_block1(x0)
        # c1 = self.conv2(x1)
        # x2 = self.base_block2(x1)
        # x2 = self.conv_end(x2)
        # out = self.conv3(torch.cat([x2,c,c0,c1],dim=1))
        # out = self.attn1(out,out)


        return out+x

class NET(nn.Module):
    def __init__(self, scale=4,nfeats=64, conv=common.default_conv):
        super(NET, self).__init__()
        self.conv_begin = conv(in_channels=3,out_channels=64,kernel_size=3)
        self.conv_end = conv(in_channels=64,out_channels=3,kernel_size=3)
        self.block0 = base_block1(nfeats)
        self.block1 = base_block1(nfeats)
        self.block2 = base_block1(nfeats)
        self.block3 = base_block0(dim=64)
        self.block4 = base_block0(dim=64)
        self.block5 = base_block0(dim=64)
        self.block6 = base_block0(dim=64)
        self.block7 = base_block0(dim=64)
        # self.block1 = base_block1(inchan=64, outchan=64)
        # self.block2 = base_block1(inchan=64, outchan=64)
        # self.block3 = base_block1(inchan=64, outchan=64)
        # self.block4 = base_block1(inchan=64, outchan=64)
        # self.block5 = base_block1(inchan=64, outchan=64)
        # self.block6 = base_block1(inchan=64, outchan=64)
        # self.block7 = base_block1(inchan=64, outchan=64)
        self.conv0 = conv(nfeats, nfeats, 1)
        self.conv1 = conv(nfeats, nfeats, 1)
        self.conv2 = conv(nfeats, nfeats, 1)
        self.conv3 = conv(nfeats * 4, nfeats, 1)
        self.up = common.Upsampler(conv=conv, scale=scale, n_feats=64)

        # self.up = nn.Sequential(
        #     conv(64*4,64,3),
        #     nn.PixelShuffle(2)
        # )

        self.sub_mean = common.MeanShift(255)
        self.add_mean = common.MeanShift(255, sign=1)
    def forward(self,x):
        x = self.sub_mean(x)
        x = self.conv_begin(x)

        # x1 = self.block0(x)
        # x2 = self.block1(x1)
        # x3 = self.block2(x2)
        # x4 = self.block3(x3)
        # x5 = self.block4(x4) + x4
        # x6 = self.block5(x5) + x3
        # x7 = self.block6(x6) + x2
        # x8 = self.block7(x7) + x1
        c = self.conv0(x)
        x0 = self.block0(x)
        c0 = self.conv1(x0)
        x1 = self.block1(x0)
        c1 = self.conv2(x1)
        x2 = self.block2(x1)


        out = self.conv3(torch.cat([x2,c,c0,c1],dim=1))

        out = self.up(out+x)

        out = self.conv_end(out)
        out = self.add_mean(out)
        return out
def make_model(args, parent=False):
    return NET()





