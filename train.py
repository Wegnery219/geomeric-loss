import math
import os
from newposenet import posenet as posenet
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data import get_data


# class lossfunction1(nn.Module):
#     def __init__(self):
#         super(lossfunction1, self).__init__()
#
#
#     def forward(self, posex, predictx, poseq, predictq, nsx, nsq):
#         lx = torch.norm(posex-predictx, 1)
#         unitlength = torch.norm(predictq)
#         newq = poseq/unitlength
#         lq = torch.norm(poseq-newq, 1)
#         print(lx, lq, nsx, nsq)
#         return lx*torch.exp(-nsx)+nsx+lq*torch.exp(-nsq)+nsq


def train(model, img_data, pose_data, optimizer, epoch):
    model.train()
    for i in range(len(img_data)):
        img = Variable(img_data[i]).float()
        img.unsqueeze_(0)
        optimizer.zero_grad()
        posex = torch.from_numpy(np.array(pose_data[i][4:7])).float()
        poseq = torch.from_numpy(np.array(pose_data[i][0:4])).float()
        posex = posex.to(device)
        poseq = poseq.to(device)
        output, loss,lx,lq = model(img, posex, poseq)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {}\tlx:{}\tlq:{}\tloss: {:.6f}'.format(
                epoch,lx,lq, loss.data.cpu().numpy()[0]))

def main():
    # load_data
    img_data, pose_data = get_data()
    #train
    model = posenet(3,0.0,-3.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    for epoch in range(10):
        train(model, img_data, pose_data, optimizer, epoch)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    main()
