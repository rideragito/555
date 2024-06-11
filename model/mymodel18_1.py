import torch
import torch.nn as nn
from model import common
class body(nn.Module):
    def __init__(self,conv=common.default_conv):
        super(body, self).__init__()
        self.res1 = common.ResBlock(conv=conv, n_feats=64, kernel_size=3)
        self.res2 = common.ResBlock(conv=conv, n_feats=64, kernel_size=3)
        self.res3 = common.ResBlock(conv=conv, n_feats=64, kernel_size=3)
        self.res4 = common.ResBlock(conv=conv, n_feats=64, kernel_size=3)
    def forward(self,x):
        x1 = self.res1(x) + x
        x2 = self.res2(x1) + x
        x3 = self.res3(x2) + x2 + x1
        x4 = self.res4(x3)
        return x4

class NET(nn.Module):
    def __init__(self,args,conv=common.default_conv):
        super(NET, self).__init__()
        self.conv = common.default_conv(in_channels=3,out_channels=64,kernel_size=3)
        self.body1 = body()
        self.body2 = body()
        self.body3 = body()
        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            common.default_conv(64,4*64,3),
            nn.PixelShuffle(2)
        )
        self.up1 = nn.Sequential(
            common.default_conv(64,4*64,3),
            nn.PixelShuffle(2),
            common.default_conv(64,3,3)
        )
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
    def forward(self,x):
        x = self.sub_mean(x)
        x = self.conv(x)
        x = self.body1(x) + x
        x4 = self.body3(x)
        x1 = self.up(x)
        x2 = self.body2(x1)
        x3 = self.down(x2) + x + x4
        out = self.up1(x3)
        out = self.add_mean(out)
        return out

def make_model(args):
    return NET(args)
if __name__ == '__main__':
    model = NET()
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    print('out.shape: ', out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
