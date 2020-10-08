import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CAMLoss(nn.Module):
    def __init__(self, margin=2.0, thr=125):
        super(CAMLoss, self).__init__()
        self.margin = margin
        self.thr = thr
        self.size = 256
        self.batchsize = 8

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        bz, nc, h, w = feature_conv.shape

        size_upsample = (bz, self.size, self.size)

        output_cam = []
        for idx in class_idx:
            cam = torch.matmul(weight_softmax[idx], feature_conv.reshape((bz, nc, h * w)))

            # cam = weight_softmax[idx].dot(feature_conv.reshape((bz, nc, h * w)))
            cam = cam.reshape(bz, h, w)
            cam = cam - torch.min(cam)  # Normalization
            cam_img = cam / torch.max(cam) * 255
            output_cam = cam_img.resize_(size_upsample)
            # cam_img = np.uint8(255 * cam_img)
            # output_cam = cv2.resize(cam_img, size_upsample)    # upsample
            output_cam = np.resize(cam_img, size_upsample)
            feature_cam = np.resize(feature_conv, size_upsample)
        return output_cam, feature_cam

    def forward(self, pred, cla_truth, seg_truth, features_blobs, weight_softmax, idx):
        # loss1 = F.cross_entropy(pred, cla_truth)

        cam_img0, feature_cam = self.returnCAM(features_blobs, weight_softmax, idx[:, 0])
        cam_img1, _ = self.returnCAM(features_blobs, weight_softmax, idx[:, 1])
        ## cam_img2, _ = self.returnCAM(features_blobs, weight_softmax, idx[:, 2])

        cam_img0[cam_img0 <= self.thr] = 0.0
        cam_img0[cam_img0 > 0] = 1.0
        dst = torch.from_numpy(np.array(cam_img0)).float().to("cuda:0")
        euclidean_distance1 = torch.mean(F.pairwise_distance(dst, seg_truth), 1)


        cam_fea0 = torch.from_numpy(np.multiply(cam_img0, feature_cam)).float().to("cuda:0")
        cam_fea1 = torch.from_numpy(np.multiply(cam_img1, feature_cam)).float().to("cuda:0")
        euclidean_distance2 = torch.mean(F.pairwise_distance(cam_fea0, cam_fea1), 1)

        loss_dis = torch.mean(euclidean_distance1 + torch.clamp(self.margin - euclidean_distance2, min=0.0))
        # euclidean_distance2 = torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        return loss_dis