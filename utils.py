# -*- coding: utf-8 -*-
# @ProjectName: pytorch-classifaction-CAM
# @File : utils.py
# @Author : zyQin
# @Time : 2020/7/13 17:13
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt

def gray2rgb(img):
    a = copy.deepcopy(img)
    b = copy.deepcopy(img)
    c = copy.deepcopy(img)
    com = np.array([a, b, c])
    com = com.swapaxes(0, 1).swapaxes(1, 2)
    return com

def gray2rgbTorch(img):
    import torch
    a = copy.deepcopy(img)
    b = copy.deepcopy(img)
    c = copy.deepcopy(img)
    com = torch.tensor([a, b, c])
    com = com.transpose(0, 1).transpose(1, 2).contiguous()
    return com




if __name__ == '__main__':
    img = cv2.imread('E:/Data/CNV-cla-seg/Images/00001.png',cv2.IMREAD_GRAYSCALE)
    im = gray2rgb(img)
    # im = im.transpose(0, 1).transpose(1, 2).contiguous()

    plt.imshow(im)
    plt.show()