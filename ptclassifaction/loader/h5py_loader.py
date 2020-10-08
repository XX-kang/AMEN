import os
from os.path import join as pjoin
import collections
import torch
import numpy as np
from glob import glob
import h5py
import re

from PIL import Image
from torch.utils import data
from torchvision import transforms

class H5pyLoader(data.Dataset):

    def __init__(
        self,
        root,
        is_transform=True,
        img_size=79,
        split="train",
        test_mode=False,
        img_norm=True,
        n_classes=2,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([125.08347, 124.99436, 124.99769])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.files = collections.defaultdict(list)
        self.n_classes = n_classes

        path = pjoin(root, split, split + ".h5")
        with h5py.File(path) as f:
            if split=="train":
                self.data = f['data'][:]
                self.label = f["label"][:]
                f.close()
            else:
                self.data = f['data'][:]
                f.close()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = np.squeeze(self.data[index])
        lbl = self.label[index][1]

        return im, lbl



