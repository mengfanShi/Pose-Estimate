# -*- coding:utf-8 -*-
# @TIME     :2019/1/19 10:57
# @Author   :Fan
# @File     :ImageAugment.py
"""
Purpose: To expand the dataset for our project
Contain: Aug_scale, Aug_crop, Aug_flip, Aug_rotate
Param_Trans: store the value of parameters
"""

"""
The order in this work:
0: nose 	        1: neck             2: right shoulder   3: right elbow      
4: right wrist      5: left shoulder    6: left elbow	     7: left wrist  
8: right hip        9: right knee	    10: right ankle	    11: left hip   
12: left knee       13: left ankle	    14: right eye	    15: left eye   
16: right ear       17: left ear 
"""

import cv2
import numpy as np



def Aug_scale(meta, img, mask_miss, Param_Trans):
    param = np.random.rand()    # range in (0, 1)
    scale_max = Param_Trans['scale_max']
    scale_min = Param_Trans['scale_min']

    # multiplier linear shear into [scale_min, scale_max]
    multiplier = scale_min + (scale_max - scale_min) * param
    scale = multiplier * (Param_Trans['target_dist'] / meta['scale_provided'])
    img = cv2.resize(img, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)
    mask_miss = cv2.resize(mask_miss, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC)

    # modify meta data
    meta['objpos'] *= scale
    meta['joint_self'][:, :2] *= scale
    if(meta['numOtherPeople'] != 0):
        meta['objpos_other'] *= scale
        meta['joint_others'][:, :, :2] *= scale
    return meta, img, mask_miss

def Aug_crop(meta, img, mask_miss, Param_Trans):
    offset_max = 2 * Param_Trans['crop_disturb']
    crop_x = int(Param_Trans['crop_size_x'])
    crop_y = int(Param_Trans['crop_size_y'])
    param_x = np.random.rand()
    param_y = np.random.rand()

    # offset has both positive and negative
    offset_x = int((param_x - 0.5) * offset_max)
    offset_y = int((param_y - 0.5) * offset_max)
    center = meta['objpos'] + np.array([offset_x, offset_y])
    center = center.astype(int)

    # padding
    pad_col = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
    pad_col_mask = np.ones((crop_y, mask_miss.shape[1]), dtype=np.uint8) * 255

    img = np.concatenate((pad_col, img, pad_col), axis=0)
    mask_miss = np.concatenate((pad_col_mask, mask_miss, pad_col_mask), axis=0)

    pad_row = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
    pad_row_mask = np.ones((mask_miss.shape[0], crop_x), dtype=np.uint8) * 255

    img = np.concatenate((pad_row, img, pad_row), axis=1)
    mask_miss = np.concatenate((pad_row_mask, mask_miss, pad_row_mask), axis=1)

    # crop
    img = img[int(center[1] + crop_y / 2):int(center[1] + crop_y * 3 / 2),
              int(center[0] + crop_x / 2):int(center[0] + crop_x * 3 / 2), :]
    mask_miss = mask_miss[int(center[1] + crop_y / 2):int(center[1] + crop_y * 3 / 2 + 1),
                          int(center[0] + crop_x / 2):int(center[0] + crop_x * 3 / 2 + 1)]

    offset_left = crop_x - (center[0] + crop_x / 2)
    offset_up = crop_y - (center[1] + crop_y / 2)
    offset = np.array([offset_left, offset_up])

    # modify meta data
    meta['objpos'] += offset
    meta['joint_self'][:, :2] += offset
    mask = np.logical_or.reduce((
        meta['joint_self'][:, 0] >= crop_x,
        meta['joint_self'][:, 0] < 0,
        meta['joint_self'][:, 1] >= crop_y,
        meta['joint_self'][:, 1] < 0
    ))

    # screen out the joint out of the cropped image
    meta['joint_self'][mask == True, 2] = 2
        # 0:visible + labeled   1:labeled   2:unlabeled
    if(meta['numOtherPeople'] != 0):
        meta['objpos_other'] += offset
        meta['joint_others'][:, :, :2] += offset
        mask_other = np.logical_or.reduce((
            meta['joint_others'][:, :, 0] >= crop_x,
            meta['joint_others'][:, :, 0] < 0,
            meta['joint_others'][:, :, 1] >= crop_y,
            meta['joint_others'][:, :, 1] < 0
        ))
        meta['joint_others'][mask_other == True, 2] = 2
    return meta, img, mask_miss

def Aug_flip(meta, img, mask_miss):
    param = np.random.rand()
    if (param <= 0.5):      # perform the flip (around y-axis)
        img_ = img.copy()
        mask_miss_ = mask_miss.copy()
        width = img.shape[1]

        # flipCode: 0:x-axis, positive value:y-axis, negative value:both
        cv2.flip(src=img_, flipCode=1, dst=img)
        cv2.flip(src=mask_miss_, flipCode=1, dst=mask_miss)

        # modify meta data
        meta['objpos'][0] = width - 1 - meta['objpos'][0]
        meta['joint_self'][:, 0] = width - 1 - meta['joint_self'][:, 0]
        meta['joint_self'] =  meta['joint_self'][[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16]]
        if (meta['numOtherPeople'] != 0):
            meta['objpos_other'][:, 0] = width - 1 - meta['objpos_other'][:, 0]
            meta['joint_others'][:, :, 0] = width - 1 - meta['joint_others'][:, :, 0]
            for i in range(meta['numOtherPeople']):
                meta['joint_others'][i] = meta['joint_others'][i][
                    [0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16]
                ]
    return meta, img, mask_miss

def rotate_point(point, mat):
    p = np.ones((3, 1))
    p[0] = point[0]
    p[1] = point[1]
    new_point = mat.dot(p)
    point[0] = new_point[0]
    point[1] = new_point[1]
    return point

def Aug_rotate(meta, img, mask_miss, Param_Trans, num_joint=18):
    rotate_max = 2 * Param_Trans['rotate_degree']
    degree = (np.random.rand() - 0.5) * rotate_max
    scale = 1.0
    num_joint = num_joint

    # perform rotation without being cut off
    (img_h, img_w) = img.shape[:2]
    (mask_h, mask_w) = mask_miss.shape[:2]

    # apply the negative of the angle to rotate clockwise
    img_mat = cv2.getRotationMatrix2D(center=(img_w // 2, img_h // 2),
                                      angle=-degree, scale=scale)
    mask_mat = cv2.getRotationMatrix2D(center=(mask_w // 2, mask_h // 2),
                                       angle=-degree, scale=scale)

    img_cos = np.abs(img_mat[0, 0] / scale)
    img_sin = np.abs(img_mat[0, 1] / scale)
    mask_cos = np.abs(mask_mat[0, 0] / scale)
    mask_sin = np.abs(mask_mat[0, 1] / scale)

    # compute the new size of the image
    img_nw = int((img_h * img_sin) + (img_w * img_cos))
    img_nh = int((img_h * img_cos) + (img_w * img_sin))
    mask_nw = int((mask_h * mask_sin) + (mask_w * mask_cos))
    mask_nh = int((mask_h * mask_cos) + (mask_w * mask_sin))

    # adjust the rotation mat
    img_mat[0, 2] += (img_nw / 2) - img_w // 2
    img_mat[1, 2] += (img_nh / 2) - img_h // 2
    mask_mat[0, 2] += (mask_nw / 2) - mask_w // 2
    mask_mat[1, 2] += (mask_nh / 2) - mask_h // 2

    # do the rotate
    img_rot = cv2.warpAffine(src=img, M=img_mat, dsize=(img_nw, img_nh), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    mask_rot = cv2.warpAffine(src=mask_miss, M=mask_mat, dsize=(mask_nw, mask_nh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255))

    # modify meta data
    meta['objpos'] = rotate_point(meta['objpos'], img_mat)
    for i in range(num_joint):
        meta['joint_self'][i, :] = rotate_point(meta['joint_self'][i, :], img_mat)

    if (meta['numOtherPeople'] != 0):
        for j in range(meta['numOtherPeople']):
            meta['objpos_other'][j, :] = rotate_point(meta['objpos_other'][j, :], img_mat)

            for i in range(num_joint):
                meta['joint_others'][j, i, :] = rotate_point(
                    meta['joint_others'][j, i, :], img_mat)
    return meta, img_rot, mask_rot





