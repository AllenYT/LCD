import os
import sys
import json
from turtle import forward
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import Tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# from model.mymodelv2 import LCDNet


class Loc_Loss(nn.Module):
    def __init__(self):
        super(Loc_Loss, self).__init__()
    
    def forward(self, pred : Tensor, gt: Tensor)-> Tensor:
        # （4，480，480）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        th = nn.Sigmoid()
        tot = transforms.ToTensor()
        pred = pred.transpose(1,2,0)
        pred = tot(pred)
        pred = th(pred).to(device)
        loss = 0.0
        for i in range(pred.shape[0]):
            temp1 = pred[i,:,:].reshape(-1)
            temp2 = gt[i,:,:].reshape(-1)
            one_num = (temp2 == 1).sum(dim=0)
            zero_num = (temp2 == 0).sum(dim=0)
            zero = (-((temp2-1) * temp1)).sum() /zero_num
            one = (temp1*temp2).sum() / one_num
            loss = loss + (zero - one)
        loss = loss / pred.shape[0] 
        return loss
    

# if __name__  == "__main__":
#     # input
#     a = torch.randn(2,2,2)
#     b = torch.randn(2,2,2)
#     b = (b>0.5).float()
#     print(a)
#     print(b)
#     loss = 0.0
#     for i in range(a.shape[0]):
#             temp1 = a[i].reshape(-1)
#             temp2 = b[i].reshape(-1)
#             one_num = (temp2 == 1).sum(dim=0)
#             zero_num = (temp2 == 0).sum(dim=0)
#             zero = (-((temp2-1) * temp1)).sum() / zero_num
#             print(zero)
#             one = (temp1*temp2).sum() / one_num
#             print(one)
#             loss = loss + (zero - one)
#     loss = loss / b.shape[0] 


    
