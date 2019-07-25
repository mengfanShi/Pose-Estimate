# -*- coding:utf-8 -*-
# @TIME     :2019/2/18 9:45
# @Author   :Fan
# @File     :Preprocess.py
"""
Purpose: To provide different utility to process image which shape is (h, w, 3)
Result: An image which type is np.float32 and shape is (3, h, w)
"""

import numpy as np


def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image/256. - 0.5        # normalization
    image = image.transpose((2, 0, 1)).astype(np.float32)
    return image

def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.

    # use the mean and std provided by imageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # change the order of the channels
    image_pro = image.copy()[:, :, ::-1]

    for i in range(3):
        image_pro[:, :, i] -= mean[i]
        image_pro[:, :, i] /= std[i]
    image_pro = image_pro.transpose((2, 0, 1)).astype(np.float32)
    return image_pro

def inception_preprocess(image):
    image_pro = image.copy()[:, :, ::-1]
    image_pro = image_pro.astype(np.float32) / 128. - 1.
    image_pro = image_pro.transpose((2, 0, 1)).astype(np.float32)
    return image_pro

def ssd_preprocess(image):
    image_pro = image.copy()[:, :, ::-1]
    image_pro = image_pro.astype(np.float32) - (104., 117., 123.)
    image_pro = image_pro[:, :, ::-1]
    image_pro = image_pro.transpose((2, 0, 1)).astype(np.float32)
    return image_pro


