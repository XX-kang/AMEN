import os
from os.path import join as pjoin
import collections
from glob import glob
import re
import h5py
import pandas as pd
import numpy as np
import torch.nn.functional as F

# testa_data_path = 'E:\Data\Classifacation_h5py_AD/test/testa.h5'
# testb_data_path = 'E:\Data\Classifacation_h5py_AD/test/testb.h5'
# with h5py.File(testa_data_path) as f:
#     testa_data = f['data'][:]
#     f.close()
#
# test_data = np.concatenate((testa_data, testb_data),axis=0)
# with h5py.File('E:/Data/Classifacation_h5py_AD/test/test.h5', 'w') as f:
#     f['data'] = test_data
#     f.close()

from ptclassifaction.loss.cam_loss import CAMLoss

import torch


e = np.ones([20, 768])
dt = h5py.special_dtype(vlen=np.dtype('float64'))

with h5py.File('test.h5', 'w') as f:
    g = f.create_group(name='g_group')
    s_group = g.create_group(name='sentence')
    s_group.create_dataset(name='each_s',shape=(e.shape))
    s_group['each_s'][:] = e
    # f['data'] = lits # 报错
    f.close()
