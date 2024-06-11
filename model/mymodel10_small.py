import torch
import torch.nn as nn
import os
class res(nn.Module):
    def __init__(self,inchan,outchan):
        super(res, self).__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.conv = nn.Conv2d(in_channels=self.inchan, out_channels=self.outchan, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=self.outchan,out_channels=self.outchan,kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.outchan,out_channels=self.outchan,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.outchan,out_channels=self.outchan,kernel_size=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.conv3(x1)
        return self.relu(x+x1)

# class RES(nn.Module):
#     def __init__(self,inchan,outchan):
#         super(RES, self).__init__()
#         self.inchan = inchan
#         self.outchan = outchan
#         self.conv = nn.Conv2d(in_channels=self.inchan,out_channels=self.outchan,kernel_size=3,padding=1)
#         self.res = res(inchan=self.outchan,outchan=self.outchan)
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         x = self.conv(x)
#         x1 = self.res(x) + x
#         x2 = self.res(x1) + self.relu(x1)
#         x3 = self.res(x2) + self.relu(x2)
#         x4 = self.res(x3)
#         out = self.relu(x4+x)
#         return out
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.PReLU()
        )
        self.up = nn.Upsample(scale_factor=2)
        self.res1 = res(inchan=64,outchan=128)
        self.res2 = res(inchan=128,outchan=256)
        self.res3 = res(inchan=256,outchan=512)
        self.down1 = nn.Conv2d(512,256, kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(256,128,kernel_size=2,stride=2)
        self.down3 = nn.Conv2d(128,64,kernel_size=2,stride=2)
        self.res4 = res(inchan=1024,outchan=512)
        self.res5 = res(inchan=512,outchan=256)
        self.res6 = res(inchan=256,outchan=128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.conv_end = nn.Conv2d(16,3,kernel_size=3,padding=1)
    def forward(self, x):
        x = self.conv(x)
        x1 = self.up(self.res1(x))
        x2 = self.up(self.res2(x1))
        x3 = self.up(self.res3(x2))
        x4 = self.conv1(x3)
        y1 = torch.cat([x3,x4],dim=1)
        y1 = self.down1(self.res4(y1))
        y2 = torch.cat([x2,y1],dim=1)
        y2 = self.down2(self.res5(y2))
        y3 = torch.cat([x1,y2],dim=1)
        y3 = self.down3(self.res6(y3))
        output = y3 + x
        output = self.upsample(output)
        output = self.conv_end(output)
        return output

def make_model(args):
    return NET()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
    model = NET().to(device)
    x = torch.randn(4, 3, 96, 96).to(device)
    out = model(x)
    print('out.shape: ', out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))





