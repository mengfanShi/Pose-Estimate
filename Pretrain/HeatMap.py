# -*- coding:utf-8 -*-
# @TIME     :2019/2/18 10:47
# @Author   :Fan
# @File     :HeatMap.py
"""
Purpose: To generate the ground truth HeatMap of every channel
Params:
:Center: int with shape of (2,) represent the person's KeyPoint
:Accumulate_Map: one channel of HeatMap, which is accumulated
:Param_Trans: store the value of stride and crop_size
"""

import  numpy as np


def Get_HeatMap(Center, Accumulate_Map, Param_Trans):
    crop_x = Param_Trans['crop_size_x']
    crop_y = Param_Trans['crop_size_y']
    stride = Param_Trans['stride']
    sigma = Param_Trans['sigma']

    start = stride / 2.0 - 0.5
    grid_x = crop_x / stride
    grid_y = crop_y / stride
    range_x = [i for i in range(int(grid_x))]
    range_y = [i for i in range(int(grid_y))]
    xx, yy = np.meshgrid(range_x, range_y)
    xx = xx * stride + start
    yy = yy * stride + start

    # Gaussian distance
    L2 = (xx - Center[0]) ** 2 + (yy - Center[1]) ** 2
    exponent = L2 / (2. * sigma ** 2)
    max_value = np.log(100)
    mask = exponent <= max_value
    HeatMap = np.exp(-exponent)
    HeatMap = np.multiply(mask, HeatMap)

    Accumulate_Map += HeatMap
    Accumulate_Map[Accumulate_Map > 1.0] = 1.0
    return Accumulate_Map