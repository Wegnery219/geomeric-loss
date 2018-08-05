import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def conv_relu(in_channel, out_channel, kernel_size, stride, padding):
    layer=nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel, eps=0.01),
        nn.ReLU(True)
    )
    return layer


class BasicCov(nn.Module):
    def __init__(self, in_channels,out_channels, **kwargs):
        super(BasicCov, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.01)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, inchannels, out1_1, out5_1, out5_2, out3_1, out3_2, out4_1):
        super(Inception, self).__init__()
        self.branch1x1 = BasicCov(inchannels, out1_1, kernel_size=1)

        self.branch5x5_1 = BasicCov(inchannels, out5_1, kernel_size=1)
        self.branch5x5 = BasicCov(out5_1, out5_2, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicCov(inchannels, out3_1, kernel_size=1)
        self.branch3x3 = BasicCov(out3_1, out3_2, kernel_size=3, padding=1)

        self.branch_pool = BasicCov(inchannels, out4_1, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1x1(x)

        branch5 = self.branch5x5_1(x)
        branch5 = self.branch5x5(branch5)

        branch3 = self.branch3x3_1(x)
        branch3 = self.branch3x3(branch3)

        branchpool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branchpool = self.branch_pool(branchpool)

        outputs = [branch1, branch3, branch5, branchpool]

        return torch.cat(outputs, 1)


class posenet(nn.Module):
    def __init__(self, inchannels, sx, sq):
        super(posenet, self).__init__()
        self.sx = nn.Parameter(torch.from_numpy(np.array([sx])).float())
        self.sq = nn.Parameter(torch.from_numpy(np.array([sq])).float())
        self.block1 = nn.Sequential(
            conv_relu(inchannels, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2),
            conv_relu(64, 64, 3, 1, 1),
            conv_relu(64, 192, 3, 1, 1)
        )

        self.block2 = nn.Sequential(
            Inception(192, 64, 16, 32, 96, 128, 32),
            Inception(256, 128, 32, 96, 128, 192, 64),
            nn.MaxPool2d(3, 2)
        )

        self.block3 = nn.Sequential(
            Inception(480, 192, 16, 48, 96, 208, 64),
            Inception(512, 160, 24, 64, 112, 224, 64),
            Inception(512, 128, 24, 64, 128, 256, 64),
            Inception(512, 112, 32, 64, 144, 288, 64),
            Inception(528, 256, 32, 128, 160, 320, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block4 = nn.Sequential(
            Inception(832, 256, 32, 128, 160, 320, 128),
            Inception(832, 384, 48, 128, 192, 384, 128),
            nn.AvgPool2d(7, 1)
        )

        self.dropout = nn.Dropout(p=0.4)
        self.classifier0 = nn.Linear(50176, 1024)
        self.classifier1 = nn.Linear(1024, 3)
        self.classifier2 = nn.Linear(1024, 4)

    def forward(self, x, posex, poseq):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), 50176)
        x = self.dropout(x)
        x = self.classifier0(x)
        pose_xyz = self.classifier1(x)
        pose_wpqr = self.classifier2(x)

        outputs = [pose_xyz, pose_wpqr]

        lx = torch.norm(posex - pose_xyz, 1)
        unitlength = torch.norm(pose_wpqr)
        newq = poseq / unitlength
        lq = torch.norm(poseq - newq, 1)
        #print(self.sx, self.sq)
        loss = lx * torch.exp(-self.sx) + self.sx + lq * torch.exp(-self.sq) + self.sq

        return torch.cat(outputs, 1), loss,lx,lq




