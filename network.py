import torch
from torchsummary import summary
import torch.nn as nn
import numpy as np
import os
from torch.nn import init
import torch.nn.functional as F

class UnetBlock(nn.Module):
    def __init__(self, input_chan, output_chan, flag="down"):
        super(UnetBlock, self).__init__()
        self.type = flag
        if self.type == "down":
            self.conv1 = nn.Conv2d(input_chan, output_chan, kernel_size=3)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(output_chan)
            self.conv2 = nn.Conv2d(output_chan, output_chan, kernel_size=3)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.Conv2d(output_chan, output_chan, kernel_size=4, padding=1, stride=2)
            self.relu3 = nn.ReLU(True)
            self.bn3 = nn.BatchNorm2d(output_chan)
        elif self.type =="up":
            self.conv1 = nn.Conv2d(input_chan, int(input_chan/2), kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(int(input_chan/2))
            self.conv2 = nn.Conv2d(int(input_chan/2), output_chan, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.ConvTranspose2d(output_chan, output_chan, kernel_size=4, stride=2, padding=1)
            self.relu3 = nn.ReLU(True)
            self.bn3 = nn.BatchNorm2d(output_chan)
        elif self.type =="last":
            self.conv1 = nn.Conv2d(input_chan, output_chan, kernel_size=3)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(output_chan)
            self.conv2 = nn.Conv2d(output_chan, output_chan, kernel_size=3)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.Conv2d(output_chan, 2, kernel_size=3, padding=1)
            self.relu3 = nn.Softmax()
        elif self.type == "bottom":
            self.conv1 = nn.Conv2d(input_chan, int(input_chan*2), kernel_size=3, padding=1)
            self.relu1 = nn.ReLU(True)
            self.bn1 = nn.BatchNorm2d(int(input_chan*2))
            self.conv2 = nn.Conv2d(int(input_chan*2), output_chan, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU(True)
            self.bn2 = nn.BatchNorm2d(output_chan)
            self.conv3 = nn.ConvTranspose2d(output_chan, output_chan, kernel_size=4, stride=2, padding=1)
            self.relu3 = nn.ReLU(True)
            self.bn3 = nn.BatchNorm2d(output_chan)

    def forward(self, x):
        if self.type == "down":
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.bn2(out)
            skip = out
            out = self.conv3(out)
            out = self.relu3(out)
            out = self.bn3(out)
            return out, skip

        elif self.type == "up" or self.type == "bottom":
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.relu3(out)
            out = self.bn3(out)
            return out
        elif self.type == "last":
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.bn1(out)
            out = self.conv2(out)
            out = self.relu2(out)
            out = self.bn2(out)
            out = self.conv3(out)
            out = self.relu3(out)
            return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # Down path
        self.l1st_down = UnetBlock(input_chan=1, output_chan=64, flag="down")
        self.l2nd_down = UnetBlock(input_chan=64, output_chan=128, flag="down")
        self.l3rd_down = UnetBlock(input_chan=128, output_chan=256, flag="down")
        self.l4th_down = UnetBlock(input_chan=256, output_chan=512, flag="down")
        # up path
        self.l1st_up = UnetBlock(input_chan=512, output_chan=512, flag="bottom")
        self.l2nd_up = UnetBlock(input_chan=1024, output_chan=256, flag="up")
        self.l3rd_up = UnetBlock(input_chan=512, output_chan=128, flag="up")
        self.l4th_up = UnetBlock(input_chan=256, output_chan=64, flag="up")
        # last layer
        self.last = UnetBlock(input_chan=128, output_chan=64, flag="last")

    def forward(self, x):
        out, skip1 = self.l1st_down(x)
        skip1 = F.interpolate(skip1, size=512)
        out, skip2 = self.l2nd_down(out)
        skip2 = F.interpolate(skip2, size=256)
        out, skip3 = self.l3rd_down(out)
        skip3 = F.interpolate(skip3, size=128)
        out, skip4 = self.l4th_down(out)
        out = self.l1st_up(out)
        out = torch.cat((out, skip4),1)
        out = self.l2nd_up(out)
        out = torch.cat((out, skip3), 1)
        out = self.l3rd_up(out)
        out = torch.cat((out, skip2), 1)
        out = self.l4th_up(out)
        out = torch.cat((out, skip1), 1)

        out = self.last(out)
        return out