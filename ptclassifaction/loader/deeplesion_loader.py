# coding:utf-8
from os.path import join as pjoin
import collections
import torch
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms

from ptclassifaction.utils import minmaxscaler

class DeepLesionLoader(data.Dataset):

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
        self.n_classes = 8
        self.mean = np.array([125.08347])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.n_classes = n_classes
        self.fold_series = fold_series

        if not self.test_mode:
            for split in ["train", "val"]:
                path = pjoin(self.root, "ImageSets", self.fold_series, split + ".txt")
                file_list = tuple(open(path, "r"))
                file_list = [id_.rstrip() for id_ in file_list]
                self.files[split] = file_list

        normMean = [0.498]
        normStd = [0.206]

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normMean, normStd),
            ]
        )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        seg_lbl = []
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "Images", im_name[0:12], im_name[13:20])
        cla_lbl = int(im_name[21])
        im = Image.open(im_path)
        if self.is_transform:
            im, seg_lbl = self.transform(im, seg_lbl)
        return im, seg_lbl, cla_lbl

    def transform(self, img, lbl):
        if img.size == self.img_size:
            pass
        else:
            img = img.resize(self.img_size, Image.ANTIALIAS)  # uint8 with RGB mode
        img = np.array(img) - 32768
        img[img < 0] = 0
        img = minmaxscaler(img)*255
        img = self.tf(img).float()
        lbl = torch.Tensor(lbl)
        return img, lbl

