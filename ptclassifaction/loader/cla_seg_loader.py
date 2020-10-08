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
#import cv2

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms

class ClaSegLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=True,
        img_size=512,
        split="train",
        test_mode=False,
        img_norm=True,
        n_classes=2,
        fold_series='1',
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 2
        self.mean = np.array([125.08347])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.n_classes = n_classes
        self.fold_series = fold_series

        if not self.test_mode:
            for split in ["train", "test"]:
                path = pjoin(self.root, "ImageSets", self.fold_series, split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list
        else:
            path = pjoin(self.root, "ImageSets", self.fold_series, "test.txt")
            file_list = tuple(open(path, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files["test"] = file_list

        normMean = [0.498]
        normStd = [0.206]


        self.tf = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # transforms.Normalize(normMean, normStd),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_cla_name = self.files[self.split][index].split()
        im_name = im_cla_name[0]
        im_path = pjoin(self.root, "Images", im_name)
        lbl_path = pjoin(self.root, "SegmentationClass", im_name)
        im = Image.open(im_path)
        seg_lbl = Image.open(lbl_path)
        if self.is_transform:
            im, seg_lbl = self.transform(im, seg_lbl)
        cla_lbl = int(im_cla_name[1])

        return im, seg_lbl, cla_lbl, im_name[:5]

    def transform(self, img, lbl):
        if img.size == self.img_size:
            pass
        else:
            img = img.resize(self.img_size, Image.ANTIALIAS)  # uint8 with RGB mode
            lbl = lbl.resize(self.img_size, Image.ANTIALIAS)
        img = self.tf(img)
        # img = img.half()
        lbl = torch.from_numpy(np.array(lbl)).float()
        lbl[lbl > 0] = 1
        # lbl = lbl.half()
        return img, lbl

