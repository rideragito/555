import torch
import torch.nn as nn
import os
from model.common import RES

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.res1 = RES(inchan=64,outchan=64)
        self.conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,padding=2)
        self.res2 = RES(inchan=64,outchan=64)
        self.res3 = RES(inchan=64, outchan=64)
        self.res4 = RES(inchan=64, outchan=64)
        self.res5 = RES(inchan=64, outchan=64)
        self.res6 = RES(inchan=64, outchan=64)
        self.res7 = RES(inchan=64, outchan=64)
        self.res8 = RES(inchan=64, outchan=64)
        self.res9 = RES(inchan=64, outchan=64)
        self.res10 = RES(inchan=64, outchan=64)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.up = nn.PixelShuffle(2)
        self.conv_end = nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,padding=1)
    def forward(self, x):
        x = self.conv(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        y1 = torch.cat([x1,x2],dim=1)
        y1 = self.conv1(y1)
        y2 = torch.cat([x2,x3],dim=1)
        y2 = self.conv2(y2)
        y3 = torch.cat([x3,x4],dim=1)
        y3 = self.res7(self.conv3(y3))
        y2 = self.res6(y3+y2)
        y1 = self.res5(y2+y1)
        z4 = y1 + x
        z1 = self.conv4(torch.add(y3,y2))
        z2 = self.conv5(torch.add(y2,y1))
        z3 = self.conv6(torch.add(y1,z4))
        z1 = self.res10(z1)
        z2 = self.res9(z1+z2)
        z3 = self.res8(z2+z3)
        output = z3 + z4
        output = self.up(output)
        output = self.conv_end(output)
        return output

def make_model(args):
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


