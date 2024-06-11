import torch
import torch.nn as nn
from model.common import RES
from model import common

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.res1 = RES(inchan=64,outchan=64)
        self.res2 = RES(inchan=64, outchan=64)
        self.res3 = RES(inchan=64, outchan=64)
        self.res4 = RES(inchan=64, outchan=64)
        self.res5 = RES(inchan=64, outchan=64)
        self.res6 = RES(inchan=64, outchan=64)
        self.res7 = RES(inchan=64, outchan=64)
        self.conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1)
        self.convd1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        self.convd2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.convd3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.convd4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.convd5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64*4,kernel_size=3,padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)
        )
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)
    def forward(self,x):
        x = self.sub_mean(x)
        x = self.conv(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        y1 = self.res5(self.convd1(x1+x2) + x)
        y2 = self.res6(self.convd2(x2+x3) + x)
        y3 = self.res7(self.convd3(x3+x4) + x)
        z1 = self.convd4(y1+y2) + x
        z2 = self.convd5(y2+y3) + x
       
        output = z1 + z2
        output = self.up(output)
        output = self.add_mean(output)
        return output
def make_model(args):
    return NET()
if __name__ == '__main__':
    model = NET()
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    print('out.shape: ', out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))