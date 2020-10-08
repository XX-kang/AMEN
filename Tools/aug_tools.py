# -*- coding: UTF-8 -*-
import numpy as np
import os
import shutil
import re
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

import augmentor as Augmentor


def get_image_list(dir):
    img_name_list = []
    for filename in os.listdir(dir):
        img_name_list.append(filename.rstrip('.jpg'))  # 只保存名字，去除后缀.jpg

    return img_name_list


def write_txt(dir,name_list):
    f1 = open(dir, 'a+')
    for i in range(len(name_list)):
        f1.write(name_list[i]+'\r')
    f1.close()


def load_image(img_dir,txt_dir, save_dir):
    label_dir = img_dir[:len(img_dir) - len(img_dir) - 11] + 'labels_aug/'

    with open(txt_dir, 'r') as f:
        train_list = f.readlines()
    f.close()

    img_list = get_image_list(img_dir)
    img_num = len(img_list)
    temp = 0
    flag = img_list[0][5:9]
    train_name = []
    for j in range(img_num):
        file_name = img_list[j]
        if flag != file_name[5:9]:
            temp = 0
            flag = file_name[5:9]

        new_name = str(int(file_name[0])-1) + '%02d' % temp + flag + '.png'


        for list_name in train_list:
            if new_name[3:-4] == list_name[3:-1]:
                train_name.append(new_name[:-4])
                break

        # if flag:
        #     shutil.copy(img_dir + file_name + '.jpg', os.path.join(save_dir, 'Images', new_name))
        #     shutil.copy(label_dir + file_name + '.jpg', os.path.join(save_dir, 'SegmentationClass', new_name))
        temp += 1

    return train_name


def got_aug(img_dir, label_dir, output_dir, aug_num):
    p = Augmentor.Pipeline(img_dir, output_dir)
    p.ground_truth(label_dir)
    # Add operations to the pipeline as normal:
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.skew_left_right(probability=0.5)
    p.flip_left_right(probability=0.5)

    p.sample(aug_num)


if __name__ == '__main__':
    '''
    root_dir is root dir for this part code
    save_path is save dir for result, this dir has two folder: Images and SegmentationClass
    img_px and Label_px are original images and labels based on classes
    txt_dir is path that original data set's train/test.txt for 10-fold cross-validation
    可能会有没有被增广的原图，可以手动复制过去
    '''

    # aug data
    root_dir = 'E:/MLearning/Data/CNV/纠正后数据/cnv-康侯处理-qin/'
    aug_save_path = root_dir + 'CNV-OR-GT-aug/'
    voc_save_path = root_dir + 'VOC_aug/voc_aug_cnv3/'

    # img_dir = 'E:/MLearning/Data/CNV/纠正后数据/cnv-康侯处理-qin/OR/'
    # label_dir = 'E:/MLearning/Data/CNV/纠正后数据/cnv-康侯处理-qin/GT/'
    # got_aug(img_dir, label_dir, output_dir=None, aug_num=1500)

    img_p1 = root_dir + 'CNV-OR-GT/OR/0/'
    label_p1 = root_dir + 'CNV-OR-GT/GT/0/'
    img_p2 = root_dir + 'CNV-OR-GT/OR/1/'
    label_p2 = root_dir + 'CNV-OR-GT/GT/1/'
    img_p3 = root_dir + 'CNV-OR-GT/OR/2/'
    label_p3 = root_dir + 'CNV-OR-GT/GT/2/'

    # got_aug(img_p1, label_p1, output_dir=aug_save_path, aug_num=500)
    # got_aug(img_p2, label_p2, output_dir=aug_save_path, aug_num=500)
    # got_aug(img_p3, label_p3, output_dir=aug_save_path, aug_num=500)

    for i in range(10):
        zhe_num = str(i+1)

        if not os.path.exists(voc_save_path + 'ImageSets/' + zhe_num):
            os.makedirs(voc_save_path + 'ImageSets/' + zhe_num)
        txt_dir = root_dir + 'VOC/voc_cnv/ImageSets/'+zhe_num+'/test.txt'

        train_name1 = load_image(aug_save_path + '0_images_aug/', txt_dir, voc_save_path)
        train_name2 = load_image(aug_save_path + '1_images_aug/', txt_dir, voc_save_path)
        train_name3 = load_image(aug_save_path + '2_images_aug/', txt_dir, voc_save_path)
        train_list = train_name1 + train_name2 + train_name3
        print(len(train_list))
        write_txt(voc_save_path + 'ImageSets/' + zhe_num + '/test.txt', train_list)

        # shutil.copy(root_dir + 'VOC/voc_cnv/ImageSets/'+ zhe_num + '/test.txt',
        #             voc_save_path + 'ImageSets/' + zhe_num + '/test.txt')

    # # ori data
    # test_list = get_image_list('E:/MLearning/Data/超声库/3、手动划分训练集和测试集/test/Images')
    # write_txt('E:/MLearning/Data/超声库/voc_set/ImageSets/test.txt', test_list)
    #
    # img_p1 = 'E:/MLearning/Data/超声库/3、手动划分训练集和测试集/train/良性/'
    # label_p1 = 'E:/MLearning/Data/超声库/3、手动划分训练集和测试集/train/良性label/'
    # # got_aug(img_p1, label_p1, 840)
    # img_p2 = 'E:/MLearning/Data/超声库/3、手动划分训练集和测试集/train/恶性/'
    # label_p2 = 'E:/MLearning/Data/超声库/3、手动划分训练集和测试集/train/恶性label/'
    # # got_aug(img_p2, label_p2, 840)
    # train_name1 = load_image(img_p1 + 'output/', save_path)
    # train_name2 = load_image(img_p2 + 'output/', save_path)
    # train_list = train_name1 + train_name2
    # write_txt('E:/MLearning/Data/超声库/voc_set/ImageSets/train.txt', train_list)
