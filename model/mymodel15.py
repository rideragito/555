import torch
import torch.nn as nn
import os
from esa import ESA
from common import dwconv
import common
class res(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(res, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dwconv = dwconv(inchan=self.in_ch,outchan=self.out_ch)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=self.out_ch,out_channels=self.out_ch,kernel_size=3,padding=1)
    def forward(self, x):
        x1 = self.dwconv(x)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        return x+x1

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        self.res1 = res(in_ch=64, out_ch=64)
        self.res2 = res(in_ch=64, out_ch=64)
        self.res3 = res(in_ch=64, out_ch=64)
        self.res4 = res(in_ch=64, out_ch=64)
        self.res5 = res(in_ch=64, out_ch=64)
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)


        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.up = nn.PixelShuffle(2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        self.conv_end = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)
        self.conv_d1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_d2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv_d3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.esa1 = ESA(n_feats=128)
        self.esa2 = ESA(n_feats=128)
        self.conv_d4 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
    def forward(self, x):
        # print(x.shape)
        x = self.sub_mean(x)
        x = self.conv(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x2_1 = self.conv_d1(x2)
        x3 = self.res3(x2)
        x3_1 = self.conv_d2(x3)
        y1 = torch.cat([x1, x2_1], dim=1)

        y1 = self.esa1(y1,y1)
        y1 = self.conv_2(y1)
        y2 = torch.cat([x2, x3_1], dim=1)
        y2 = self.esa2(y2,y2)
        y2 = self.conv_3(y2)
        # print('y1.shape:{},y2.shape:{}'.format(y1.shape,y2.shape))
        output = y1 + self.conv_d3(y2)
        #print(output.shape)
        output = self.conv_d4(output)
        output = self.res4(output)
        output = self.res5(output)
        out = output + x
        out = self.up(out)
        # print('up.shape: ',out.shape)
        out = self.conv_end(out)

        out = self.add_mean(out)
        return out


def make_model():
    return NET()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
    model = NET()

    x = torch.randn(4, 3, 96, 96)

    out = model(x)
    print('out.shape: ', out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

