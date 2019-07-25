# -*- coding:utf-8 -*-
# @TIME     :2019/2/18 12:18
# @Author   :Fan
# @File     :PAF.py
"""
Purpose: TO get the Part Affinity Fields
Params:
:CenterA: int with shape (2,) which point to CenterB
:CenterB: int with shape (2,) which is pointed by CenterA
:Accumulate_Map: one channel of PAF, which is accumulated
:Count: store how many PAFs overlapped in one coordinate of Accumulate_Map
:Param_Trans: store the value of stride and crop_size
"""

import numpy as np


def Get_PAF(CenterA, CenterB, Accumulate_Map, Count, Param_Trans):
    stride = Param_Trans['stride']
    crop_x = Param_Trans['crop_size_x']
    crop_y = Param_Trans['crop_size_y']
    limb_w = Param_Trans['limb_width']
    CenterA = CenterA.astype(float) / stride
    CenterB = CenterB.astype(float) / stride
    grid_x = crop_x / stride
    grid_y = crop_y / stride

    limb_vec = CenterB - CenterA
    norm = np.linalg.norm(limb_vec)

    # if the norm is too small, means that limb is too short
    if(norm == 0.0):
        return Accumulate_Map, Count

    limb_vec_unit = limb_vec / norm

    # To screen out those within the border of two points
    min_x = max(0, int(round(min(CenterA[0], CenterB[0]) - limb_w)))
    max_x = min(grid_x, int(round(max(CenterA[0], CenterB[0]) + limb_w)))
    min_y = max(0, int(round(min(CenterA[1], CenterB[1]) - limb_w)))
    max_y = min(grid_y, int(round(max(CenterA[1], CenterB[1]) + limb_w)))

    range_x = list(range(min_x, int(max_x), 1))
    range_y = list(range(min_y, int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    point_x = xx - CenterA[0]
    point_y = yy - CenterA[1]

    # Screen out the point within the limb width
    limb_width = np.abs(point_x * limb_vec_unit[1] - point_y * limb_vec_unit[0])
    mask = limb_width < limb_w

    map = np.copy(Accumulate_Map) * 0.0
    map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(map[:, :, 0]) > 0, np.abs(map[:, :, 1]) > 0))

    Accumulate_Map = np.multiply(Accumulate_Map, Count[:, :, np.newaxis])

    # Update the Accumulate_Map
    Accumulate_Map += map
    Count[mask == True] += 1

    # divide the number of non_zero vectors
    mask = (Count == 0)
    Count[mask == True] = 1
    Accumulate_Map = np.divide(Accumulate_Map, Count[:, :, np.newaxis])
    Count[mask == True] = 0

    return Accumulate_Map, Count