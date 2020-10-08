import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CAMLoss(nn.Module):
    def __init__(self, margin=70.0, thr=125):
        super(CAMLoss, self).__init__()
        self.margin = margin
        self.thr = thr
        self.size = 8
        self.batchsize = 8

    # @torchsnooper.snoop()
    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        bz, nc, h, w = feature_conv.shape

        size_upsample = (1, nc, h, w)

        output_cam = torch.empty((1, 1, h, w)).to(device)
        for idx in range(bz):
            cam = torch.matmul(weight_softmax[class_idx[idx]], feature_conv[idx].reshape((nc, h * w)))

            cam = cam.reshape(1, 1, h, w)
            cam = cam - torch.min(cam)  # Normalization
            cam = cam / torch.max(cam) * 255

            output_cam = torch.cat((output_cam, cam), dim=0)

        return output_cam[1:, :, :, :]

    def forward(self, pred, cla_truth, seg_truth, features_blobs, weight_softmax, idx):
        loss1 = F.cross_entropy(pred, cla_truth)

        cam_img0 = self.returnCAM(features_blobs, weight_softmax, idx[:, 0])
        cam_img1 = self.returnCAM(features_blobs, weight_softmax, idx[:, 1])
        cam_img2 = self.returnCAM(features_blobs, weight_softmax, idx[:, 2])

        bz, nc, h, w = features_blobs.shape

        dst = cam_img0.clone()
        dst[dst <= self.thr] = 0.0
        dst[dst > 0] = 1.0

        euclidean_distance1 = torch.mean(F.pairwise_distance(dst.reshape(bz,h,w), seg_truth.resize_((bz,h,w))), 1)

        cam_fea0 = torch.matmul(cam_img0, features_blobs)
        cam_fea1 = torch.matmul(cam_img1, features_blobs)
        cam_fea2 = torch.matmul(cam_img2, features_blobs)

        bz, nc, h, w = cam_fea0.shape
        euclidean_distance_0_1 = F.pairwise_distance(cam_fea0.reshape(bz, nc * h * w),
                                                     cam_fea1.reshape(bz, nc * h * w))/nc
        euclidean_distance_0_2 = F.pairwise_distance(cam_fea0.reshape(bz, nc * h * w),
                                                     cam_fea2.reshape(bz, nc * h * w)) / nc

        loss_dis = torch.mean(euclidean_distance1 + torch.clamp(self.margin - euclidean_distance_0_1 -
                                                                     euclidean_distance_0_2, min=0.0)) + loss1
        #
        # loss_dis = torch.mean(torch.clamp(self.margin - euclidean_distance_0_1 -
        #                                                         euclidean_distance_0_2, min=0.0)) + loss1
        # loss_dis = loss1
        #
        return loss_dis
