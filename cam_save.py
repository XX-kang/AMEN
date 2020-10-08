# -*- coding: utf-8 -*-
# @Time : 2020/6/30 16:25
# @Author : zyQin
# @File : cam_save.py
# @Software: PyCharm

# coding:utf-8
import os
import torch
import random
import numpy as np
from ruamel import yaml
import utils
import cv2
from os.path import join as pjoin

from torch import optim
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils import data

from ptclassifaction.loader import get_loader
from ptclassifaction.models import get_model

os.environ["CUDA_VISIBLE_DEVICES"] = '5'
devices_ids = 0

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import Augmentor
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from utils import gray2rgb, gray2rgbTorch

def test(cfg, pkls_dir, output_dir,split,alpha):
    # Setup seeds
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True

    is_densecrf = False

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    testloader = get_loader(cfg["dataset"])(
                             cfg["data_path"],
                             is_transform=True,
                             split=split,
                             test_mode=True,
                             img_size=(cfg["img_rows"], cfg["img_cols"]),
                            )
    test_loader = data.DataLoader(
        testloader,
        batch_size=1,
        num_workers=0
    )
    model = get_model(cfg["model_arch"],pretrained=False, num_classes=cfg["n_classes"], input_channels=cfg["input_channels"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0.08)

    #reload from checkpoint
    resume = "/{}_{}_best_model.pkl".format(
         cfg["model_arch"], cfg["data_name"])
    pkls = pkls_dir + resume
    print(pkls)
    if os.path.isfile(pkls):
        checkpoint = torch.load(pkls)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    model.eval()
    with torch.no_grad():
        for (test_images, test_labels, test_images_name) in test_loader:
            # val_images, val_seg_labels, val_labels, val_img_name
            import matplotlib.pyplot as plt
            #img = test_images.squeeze()
            #img = gray2rgbTorch(img)
            #plt.imshow(img)
            #plt.show()
            test_images = test_images.to(device)
            ################################################################################
            features_blo = []

            def hook_feature(module, input, outputs):
                features_blo.append(outputs)

            if cfg["model_arch"] == 'densenet121' or cfg["model_arch"] == 'vgg16_cam' or  cfg["model_arch"] == 'vgg19_cam':
                model._modules.get('features').register_forward_hook(hook_feature)
            if cfg["model_arch"] == 'googlenet':
                model._modules.get('inception5b').register_forward_hook(hook_feature)
            if cfg["model_arch"] == 'resnet50' or cfg["model_arch"] == 'resnet101':
                model._modules.get('layer4').register_forward_hook(hook_feature)

            logit = model(test_images)

            # get the softmax weight
            params = list(model.parameters())
            #weight_softmax = params[24].data.squeeze()

            weight_fc1 = params[-4].data.squeeze()

            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, class_idx = h_x.sort(0, True)
            probs = probs.cuda().data.cpu().numpy()
            class_idx = class_idx.cuda().data.cpu().numpy()

            feature_conv = features_blo[0]

            bz, nc, h, w = feature_conv.shape

            output_cam = []
            for idx in class_idx:
                # change class_num, so need dot weight_fc1
                #cam = torch.matmul(weight_softmax[idx], feature_conv.reshape((nc, h * w)))
                cam = torch.matmul(weight_fc1[idx], feature_conv.reshape((nc, h * w)))
                cam = cam.reshape(1, 1, h, w)
                cam = cam - torch.min(cam)  # Normalization
                cam = cam / torch.max(cam) * 255

                cam = np.squeeze(np.uint8(cam.cuda().data.cpu().numpy()))
                output_cam.append(cam)

                # output_cam = torch.cat((output_cam, cam), dim=0)
            ##################################################################################################
            heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (cfg["img_rows"], cfg["img_cols"])), cv2.COLORMAP_JET)
            heatmap_01 = heatmap/255
            img = np.squeeze(np.uint8(test_images.cuda().data.cpu().numpy()*255))
            if cfg["input_channels"] == 1:
                img = utils.gray2rgb(img)
                result1 = heatmap * alpha + img
                #result2 = heatmap
                #result2 = img * heatmap_01
                #result1 = heatmap[:, :, 0] * 0.2 + img * 0.7
                #result2 = img * heatmap_01[:, :, 0]
            if cfg["input_channels"] == 3:
                img = img.swapaxes(0, 1).swapaxes(1, 2)
                result1 = heatmap * alpha + img
                #result2 = img * heatmap_01
                #result2 = heatmap
            #result1 = heatmap
            #result2 = img
            img_result1_name = pjoin(output_dir, "result1", ''.join(test_images_name)+'.png')
            #img_result2_name = pjoin(output_dir, "result2", ''.join(test_images_name)+'.png')
            cv2.imwrite(img_result1_name, result1)
            #cv2.imwrite(img_result2_name, result2)

            # handle.remove()
def main(run_id,classes,split,alpha):
    #run_id = 'BCU_result2_googlenet_part_1006'
    logdir = os.path.join('./runs', run_id)
    pkls_dir = os.path.join('./pkls', run_id)
    output_dir1 = os.path.join('./outputs', run_id, 'result1/train/')
    output_dir2 = os.path.join('./outputs', run_id, 'result1/test/')
    #output_dir3 = os.path.join('./outputs', run_id, 'result2/train/')
    #output_dir4 = os.path.join('./outputs', run_id, 'result2/test/')
    for i in range(classes):
        output_dir11 = os.path.join(output_dir1, str(i))
        if not os.path.exists(output_dir11):
            os.makedirs(output_dir11)
        output_dir22 = os.path.join(output_dir2, str(i))
        if not os.path.exists(output_dir22):
            os.makedirs(output_dir22)
        # output_dir33 = os.path.join(output_dir3, str(i))
        # if not os.path.exists(output_dir33):
        #     os.makedirs(output_dir33)
        # output_dir44 = os.path.join(output_dir4, str(i))
        # if not os.path.exists(output_dir44):
        #     os.makedirs(output_dir44)
    output_dir = os.path.join('./outputs', run_id)

    with open(logdir + '/config.yaml') as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    test(cfg, pkls_dir, output_dir, split, alpha)

if __name__ == "__main__":
    run_id = 'BCU_result2_googlenet_part_1006'
    classes = 2
    split = 'test'
    alpha=0.001
    main(run_id,classes,split,alpha)
    # logdir = os.path.join('./runs', run_id)
    # pkls_dir = os.path.join('./pkls', run_id)
    # output_dir1 = os.path.join('./outputs', run_id, 'result1/train/')
    # output_dir2 = os.path.join('./outputs', run_id, 'result1/test/')
    # output_dir3 = os.path.join('./outputs', run_id, 'result2/train/')
    # output_dir4 = os.path.join('./outputs', run_id, 'result2/test/')
    # for i in range(2):
    #     output_dir11 = os.path.join(output_dir1, str(i))
    #     if not os.path.exists(output_dir11):
    #         os.makedirs(output_dir11)
    #     output_dir22 = os.path.join(output_dir2, str(i))
    #     if not os.path.exists(output_dir22):
    #         os.makedirs(output_dir22)
    #     output_dir33 = os.path.join(output_dir3, str(i))
    #     if not os.path.exists(output_dir33):
    #         os.makedirs(output_dir33)
    #     output_dir44 = os.path.join(output_dir4, str(i))
    #     if not os.path.exists(output_dir44):
    #         os.makedirs(output_dir44)
    # output_dir = os.path.join('./outputs', run_id)
    #
    # with open(logdir+'/config.yaml') as fp:
    #     cfg = yaml.load(fp, Loader=yaml.Loader)
    #
    # test(cfg, pkls_dir, output_dir)





