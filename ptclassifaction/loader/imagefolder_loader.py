import os
from os.path import join as pjoin
import collections
import torch
import numpy as np
from glob import glob
from random import shuffle
import re

from PIL import Image
from torch.utils import data
from torchvision import transforms
from augmentor.Pipeline import Pipeline
from utils import gray2rgb, gray2rgbTorch

class ImageFolderLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=True,
        img_size=256,
        split="train",
        test_mode=False,
        img_norm=True,
        n_classes=8,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.is_augmentations = True
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([125.08347, 124.99436, 124.99769])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.files = collections.defaultdict(list)
        self.n_classes = n_classes

        # if not self.test_mode:
        # for split in ["train", "test"]:
        #     path = pjoin(self.root, split + ".txt")
        #     file_list = tuple(open(path, "r"))
        #     file_list = [id_.rstrip() for id_ in file_list]
        #     self.files[split] = file_list
        for split in ["train", "test"]:
            file_list = []
            for sub_classes in os.listdir(self.root + split):
                path = pjoin(root, split, sub_classes + "/*png")
                file_list += glob(path)
            self.files[split] = file_list
            # self.setup_annotations()

        # normMean = [0.498, 0.497, 0.497]
        # normStd = [0.206, 0.206, 0.206]
        normMean = [0.498]
        normStd = [0.206]

        self.tf = transforms.Compose(
            [
                #transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                #transforms.Normalize(normMean, normStd),
            ]
        )

    def __len__(self):
        k = len(self.files[self.split])
        return k
    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_name_split = re.split(r'[/\\]', im_name[:-4])
        #im = Image.open(im_name).convert('L')
        im = Image.open(im_name)
        lbl = int(re.split(r'[/\\]', im_name)[-2])
        img_name = im_name_split[-3] + '/' + im_name_split[-2] + '/' + im_name_split[-1]
        #print(im_name_split)
        if self.is_transform:
            im = self.transform(im)
        return im, lbl ,img_name

    # def __getitem__(self, index):
    #     im_name = self.files[self.split][index]
    #     #print('im_name', im_name)
    #     lbl = int(im_name[-1])
    #     im_name_split = re.split(r'[/\\]', im_name[:-2])
    #     img_name = im_name_split[-3]+'/'+ im_name_split[-2] + '/' + im_name_split[-1]
    #     #print("img_name",img_name)
    #     #print("path", pjoin(self.root, img_name))
    #     im = Image.open(pjoin(self.root, img_name))
    #     im = im.convert('RGB')
    #
    #     im_n,im_m = im.size
    #
    #     if self.is_augmentations:
    #         im = self.augmentations(im)
    #     if self.is_transform:
    #         im = self.transform(im)
    #     return im, lbl, img_name


    def transform(self, img):
        if img.size == self.img_size:
            pass
        else:
            img = img.resize(self.img_size, Image.ANTIALIAS)  # uint8 with RGB mode
        img_rgb = self.tf(img)
        return img_rgb


    def augmentations(self, img, lbl = None):
        if lbl is None:
            lbl = img
        p = Pipeline(img, lbl)
        # Add operations to the pipeline as normal:
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.skew_left_right(probability=0.5)
        p.flip_left_right(probability=0.5)
        img2, lbl2 = p.sample()
        return img2
