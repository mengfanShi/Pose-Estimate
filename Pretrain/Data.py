# -*- coding:utf-8 -*-
# @TIME     :2019/2/19 10:50
# @Author   :Fan
# @File     :Data.py
"""
DataSet:
:train: 82783 images
:val:   40504 images

First 2644 images of val are marked by "isValidation = 1" as our val dataSet
So training dataSet have 82783 + 40504 - 2644 = 120643 images
"""

"""
MS COCO annotation order:
0: nose	            1: left eye	        2: right eye	    3: left ear	    
4: right ear        5: left shoulder	6: right shoulder	7: left elbow	
8: right elbow      9: left wrist		10: right wrist		11: left hip	
12: right hip	    13: left knee       14: right knee		15: left ankle		
16: right ankle

The order in this work:
0: nose 	        1: neck             2: right shoulder   3: right elbow      
4: right wrist      5: left shoulder    6: left elbow	     7: left wrist  
8: right hip        9: right knee	    10: right ankle	    11: left hip   
12: left knee       13: left ankle	    14: right eye	    15: left eye   
16: right ear       17: left ear 
"""

import numpy as np
import cv2
import os
import torch
import logging
import sys
from torch.utils.data import Dataset

from Pretrain.HeatMap import Get_HeatMap
from Pretrain.PAF import Get_PAF
from Pretrain.ImageAugment import Aug_crop, Aug_flip, Aug_rotate, Aug_scale
from Pretrain.Preprocess import rtpose_preprocess, ssd_preprocess, \
    vgg_preprocess, inception_preprocess

# print the special warning information
logger = logging.getLogger(__name__)


class COCOkeypoints(Dataset):
    def __init__(self, root, mask_dir, index_list, data, img_size,
                 stride, Preprocess='rtpose', transform=None,
                 target_transform=None, Param_Trans=None):

        self.Param_Trans = Param_Trans
        self.Param_Trans['crop_size_x'] = img_size
        self.Param_Trans['crop_size_y'] = img_size
        self.Param_Trans['stride'] = stride

        self.root = root
        self.mask_dir = mask_dir
        self.index_list = index_list
        self.number = len(index_list)
        self.data = data
        self.Preprocess = Preprocess
        self.transform = transform
        self.target_transform = target_transform

    def get_annotation(self, meta_data):
        """
        get meta data information
        """
        anno = dict()
        anno['dataset'] = meta_data['dataset']
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['people_index'] = int(meta_data['people_index'])
        anno['annolist_index'] = int(meta_data['annolist_index'])

        # 'objpos' means the position of object
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']
        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])
        anno['num_keypoints_other'] = np.array(
            meta_data['num_keypoints_other'])
        anno['joint_others'] = np.array(meta_data['joint_others'])
        anno['objpos_other'] = np.array(meta_data['objpos_other'])
        anno['scale_provided_other'] = meta_data['scale_provided_other']

        anno['bbox_other'] = meta_data['bbox_other']
        anno['segment_area_other'] = meta_data['segment_area_other']

        # if there exists one other people, list should be expand to keep the same
        if anno['numOtherPeople'] == 1:
            anno['joint_others'] = np.expand_dims(anno['joint_others'], 0)
            anno['objpos_other'] = np.expand_dims(anno['objpos_other'], 0)

        return anno

    def add_neck(self, meta):
        """
        To improve the accuracy of the project, we add one more joint -- neck
        It is provided by the middle of the right and left shoulder
        """
        Order_our = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

        right_shoulder = meta['joint_self'][6, :]
        left_shoulder = meta['joint_self'][5, :]
        neck = (right_shoulder + left_shoulder) / 2

        # Label the neck
        # 0:visible + labeled   1:labeled   2:unlabeled
        if right_shoulder[2] == 2 or left_shoulder[2] == 2:
            neck[2] = 2
        elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
            neck[2] = 1
        else:
            neck[2] = 0

        # Add the new joint neck to the joint list
        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        meta['joint_self'] = np.vstack((meta['joint_self'], neck))

        # change the order of data to adjust our order
        meta['joint_self'] = meta['joint_self'][Order_our, :]

        temp = []
        for i in range(meta['numOtherPeople']):
            right_shoulder = meta['joint_others'][i, 6, :]
            left_shoulder = meta['joint_others'][i, 5, :]
            neck = (right_shoulder + left_shoulder) / 2

            if right_shoulder[2] == 2 or left_shoulder[2] == 2:
                neck[2] = 2
            elif right_shoulder[2] == 1 and left_shoulder[2] == 1:
                neck[2] = 1
            else:
                neck[2] = 0

            neck = neck.reshape(1, len(neck))
            neck = np.round(neck)

            data = np.vstack((meta['joint_others'][i], neck))
            data = data[Order_our, :]
            temp.append(data)

        meta['joint_others'] = np.array(temp)

        return meta

    def rm_illegal_joint(self, meta):
        crop_x = int(self.Param_Trans['crop_size_x'])
        crop_y = int(self.Param_Trans['crop_size_y'])
        mask = np.logical_or.reduce((meta['joint_self'][:, 0] >= crop_x,
                                     meta['joint_self'][:, 0] < 0,
                                     meta['joint_self'][:, 1] >= crop_y,
                                     meta['joint_self'][:, 1] <0))
        meta['joint_self'][mask == True, :] = (1, 1, 2)

        if (meta['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((meta['joint_others'][:, :, 0] >= crop_x,
                                         meta['joint_others'][:, :, 0] < 0,
                                         meta['joint_others'][:, :, 1] >= crop_y,
                                         meta['joint_others'][:, :, 1] < 0))
            meta['joint_others'][mask == True, :] = (1, 1, 2)

        return meta

    def get_ground_truth(self, meta, mask_miss):
        stride = self.Param_Trans['stride']
        crop_x = self.Param_Trans['crop_size_x']
        crop_y = self.Param_Trans['crop_size_y']
        grid_x = crop_x / stride
        grid_y = crop_y / stride

        # number of HeatMap are defined by 18 parts and 1 background
        num_part = 18
        HeatMap = np.zeros((int(grid_y), int(grid_x), num_part + 1), dtype=np.float32)      #
        PAF = np.zeros((int(grid_y), int(grid_x), (num_part + 1) * 2), dtype=np.float32)    #

        mask_miss = cv2.resize(mask_miss, None, fx=1./stride, fy=1./stride,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32)
        mask_miss = mask_miss / 255.

        heat_mask = np.repeat(mask_miss[:, :, np.newaxis], num_part + 1, axis=2)
        paf_mask = np.repeat(mask_miss[:, :, np.newaxis], (num_part + 1) * 2, axis=2)

        # calculate the confidence map for each part
        for i in range(num_part):
            if (meta['joint_self'][i, 2] <= 1):
                center = meta['joint_self'][i, :2]
                map = HeatMap[:, :, i]
                HeatMap[:, :, i] = Get_HeatMap(
                    Center=center, Accumulate_Map=map,
                    Param_Trans=self.Param_Trans)

            for j in range(meta['numOtherPeople']):
                if (meta['joint_others'][j, i, 2] <= 1):
                    center = meta['joint_others'][j, i, :2]
                    map = HeatMap[:, :, i]
                    HeatMap[:, :, i] = Get_HeatMap(
                        Center=center, Accumulate_Map=map,
                        Param_Trans=self.Param_Trans)

        # calculate PAFs
        limb_1 = [ 1,  8,  9,  1, 11, 12,  1,  2,  3,
                   2,  1,  5,  6,  5,  1,  0,  0, 14, 15]

        limb_2 = [ 8,  9, 10, 11, 12, 13,  2,  3,  4,
                  16,  5,  6,  7, 17,  0, 14, 15, 16, 17]

        for i in range(num_part + 1):
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            if (meta['joint_self'][limb_1[i], 2] <= 1
                and meta['joint_self'][limb_2[i], 2] <= 1):

                centerA = meta['joint_self'][limb_1[i], :2]
                centerB = meta['joint_self'][limb_2[i], :2]
                map = PAF[:, :, 2 * i: 2 * i + 2]
                PAF[:, :, 2 * i: 2 * i + 2], count = Get_PAF(
                    CenterA=centerA, CenterB=centerB, Count=count,
                    Accumulate_Map=map, Param_Trans=self.Param_Trans)

            for j in range(meta['numOtherPeople']):
                if (meta['joint_others'][j, limb_1[i], 2] <= 1
                    and meta['joint_others'][j, limb_2[i], 2] <= 1):

                    centerA = meta['joint_others'][j, limb_1[i], :2]
                    centerB = meta['joint_others'][j, limb_2[i], :2]
                    map = PAF[:, :, 2 * i: 2 * i + 2]
                    PAF[:, :, 2 * i: 2 * i + 2], count = Get_PAF(
                    CenterA=centerA, CenterB=centerB, Count=count,
                    Accumulate_Map=map, Param_Trans=self.Param_Trans)

        # calculate the background HeatMap
        HeatMap[:, :, -1] = np.maximum(1 - np.max(HeatMap[:, :, :num_part], axis=2), 0.)
        return heat_mask, HeatMap, paf_mask, PAF

    def __getitem__(self, index):
        idx = self.index_list[index]
        img = cv2.imread(sys.path[0] + self.root + self.data[idx]['img_paths'])
        img_idx = self.data[idx]['img_paths'][-16:-3]

        # Load mask_miss
        if "COCO_val" in self.data[idx]['dataset']:
            mask_miss = cv2.imread(sys.path[0] + self.mask_dir + 'val2014/mask_COCO_val2014_'
                                   + img_idx + 'jpg', 0)
        elif "COCO" in self.data[idx]['dataset']:
            mask_miss = cv2.imread(sys.path[0] + self.mask_dir + 'train2014/mask_COCO_train2014_'
                                   + img_idx + 'jpg', 0)
        else:
            mask_miss = np.ones(img.shape, dtype=np.float32)
            logger.warning('can not find mask_miss file, pay attention to LoadData 242 row')

        meta_data = self.get_annotation(self.data[idx])
        meta_data = self.add_neck(meta_data)

        # Conduct the image augmentation
        meta_data, img, mask_miss = Aug_scale(meta=meta_data, img=img,
                                             mask_miss=mask_miss, Param_Trans=self.Param_Trans)
        #
        #meta_data, img, mask_miss = Aug_rotate(meta=meta_data, img=img,
        #                                        mask_miss=mask_miss, Param_Trans=self.Param_Trans)
        #
        meta_data, img, mask_miss = Aug_crop(meta=meta_data, img=img,
                                             mask_miss=mask_miss, Param_Trans=self.Param_Trans)
        #
        meta_data, img, mask_miss = Aug_flip(meta=meta_data,
                                             img=img, mask_miss=mask_miss)

        meta_data = self.rm_illegal_joint(meta_data)

        # Get the ground truth information
        heat_mask, HeatMap, paf_mask, PAF = self.get_ground_truth(meta=meta_data,
                                                                  mask_miss=mask_miss)

        # Do the image preprocess
        if self.Preprocess == 'rtpose':
            img = rtpose_preprocess(img)

        elif self.Preprocess == 'vgg':
            img = vgg_preprocess(img)

        elif self.Preprocess == 'inception':
            img = inception_preprocess(img)

        elif self.Preprocess == 'ssd':
            img = ssd_preprocess(img)

        # Change the type
        img = torch.from_numpy(img)

        heat_mask = torch.from_numpy(
            heat_mask.transpose((2, 0, 1)).astype(np.float32))

        HeatMap = torch.from_numpy(
            HeatMap.transpose((2, 0, 1)).astype(np.float32))

        paf_mask = torch.from_numpy(
            paf_mask.transpose((2, 0, 1)).astype(np.float32))

        PAF = torch.from_numpy(
            PAF.transpose((2, 0, 1)).astype(np.float32))

        return img, HeatMap, heat_mask, PAF, paf_mask

    def __len__(self):
        return self.number








