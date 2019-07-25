# -*- coding:utf-8 -*-
# @TIME     :2019/3/8 15:25
# @Author   :Fan
# @File     :COCO_eval.py

import os
import cv2
import numpy as np
import json
import pandas as pd
import torch
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from Pretrain.Preprocess import rtpose_preprocess, vgg_preprocess
from Eval.Parser import decode_pose, people_to_pose, plot_pose
#from Eval.cpm.cpm_layer import rtpose_postprocess_np, rtpose_postprocess
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
ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
COCO_num_joints = 17

limb_1 = [ 1,  8,  9,  1, 11, 12,  1,  2,  3,
           2,  1,  5,  6,  5,  1,  0,  0, 14, 15]

limb_2 = [ 8,  9, 10, 11, 12, 13,  2,  3,  4,
          16,  5,  6,  7, 17,  0, 14, 15, 16, 17]

num_joints = 18
num_limbs = 2 * (num_joints + 1)


def crop_image(img, dst_size=None, factor=32, is_ceil=True):
    img_shape = img.shape
    img_size_min = np.min(img_shape[0:2])
    scale = float(dst_size) / img_size_min
    img = cv2.resize(img, None, fx=scale, fy=scale)

    if is_ceil:
        height = int(np.ceil(float(img.shape[0]) / factor)) * factor
        width = int(np.ceil(float(img.shape[1]) / factor)) * factor
    else:
        height = int(np.floor(float(img.shape[0]) / factor)) * factor
        width = int(np.floor(float(img.shape[1]) / factor)) * factor

    img_cropped = np.zeros([height, width, img.shape[2]], dtype=img.dtype)
    img_cropped[0:img.shape[0], 0:img.shape[1], :] = img

    return img_cropped, scale, img.shape


def eval_coco(outputs, dataDir, imgIds):
    """Evaluate images on Coco test set
    :param outputs: list of dictionaries, the models' processed outputs
    :param dataDir: string, path to the MSCOCO data directory
    :param imgIds: list, all the image ids in the validation set
    :returns : float, the mAP score
    """
    with open('results.json', 'w') as f:
        json.dump(outputs, f)
    annType = 'keypoints'
    prefix = 'person_keypoints'

    # initialize COCO ground truth api
    dataType = 'val2014'
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)

    # load annotations
    cocoGt = COCO(annFile)
    # load model outputs
    cocoDt = cocoGt.loadRes('results.json')
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    os.remove('results.json')
    # return Average Precision
    return cocoEval.stats[0]


def get_coco_val(file_path):
    """
    Reads MSCOCO validation information
    :param file_path: string, the path to the MSCOCO validation file
    :returns : list of image ids, list of image file paths, list of widths,
               list of heights
    """
    val_coco = pd.read_csv(file_path, sep='\s+', header=None)
    image_ids = list(val_coco[1])
    file_paths = list(val_coco[2])
    heights = list(val_coco[3])
    widths = list(val_coco[4])

    return image_ids, file_paths, heights, widths


def get_multiplier(img, size):
    """
    Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    #scale_search = [0.5, 1., 1.5, 2., 2.5]
    scale_search = [0.5, 1., 1.5, 2.]
    return [x * float(size) / float(img.shape[0]) for x in scale_search]

def get_output(img, model, size=256, preprocess='rtpose'):
    im_croped, im_scale, real_shape = crop_image(
        img, size, factor=8, is_ceil=True)

    if preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_croped)
    else:
        im_data = vgg_preprocess(im_croped)
    im_data = np.expand_dims(im_data, 0)
    img_input = torch.from_numpy(im_data).cuda().float()
    outputs, _ = model(img_input)
    paf, heatmap = outputs[-2], outputs[-1]
    return paf, heatmap, im_scale

def get_multi_scale_output(multiplier, img, model, stride, preprocess='rtpose'):
    """
    Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param img: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], num_joints+1), dtype=np.float32)
    paf_avg = np.zeros((img.shape[0], img.shape[1], num_limbs), dtype=np.float32)
    max_scale = multiplier[-1]
    max_size = max_scale * img.shape[0]
    # padding
    max_cropped, _, _ = crop_image(img, max_size, factor=8, is_ceil=True)
    batch_images = np.zeros(
        (len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        img_size = scale * img.shape[0]

        # padding
        im_croped, im_scale, real_shape = crop_image(
            img, img_size, factor=8, is_ceil=True)

        if preprocess == 'rtpose':
            im_data = rtpose_preprocess(im_croped)
        else:
            im_data = vgg_preprocess(im_croped)

        batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
    pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        img_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = crop_image(
            img, img_size, factor=8, is_ceil=True)
        heatmap = heatmaps[m, :int(im_cropped.shape[0] / stride),
                            :int(im_cropped.shape[1] / stride), :]
        heatmap = cv2.resize(heatmap, None, fx=stride, fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
        heatmap = cv2.resize(
            heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = pafs[m, :int(im_cropped.shape[0] / stride),
                    :int(im_cropped.shape[1] / stride), :]
        paf = cv2.resize(paf, None, fx=stride, fy=stride,
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[0:real_shape[0], 0:real_shape[1], :]
        paf = cv2.resize(
            paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    return paf_avg, heatmap_avg


def get_flipped_output(normal_heat, flipped_heat, normal_paf, flipped_paf):
    """
    Compute the average of normal and flipped heatmap and paf
    :param normal_heat: numpy array, the normal heatmap
    :param normal_paf: numpy array, the normal paf
    :param flipped_heat: numpy array, the flipped heatmap
    :param flipped_paf: numpy array, the flipped  paf
    :returns: numpy arrays, the averaged paf and heatmap
    """

    swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                          13, 8, 9, 10, 15, 14, 17, 16, 18))

    swap_paf = np.array((6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 20, 21, 22, 23,
                         24, 25, 26, 27, 12, 13, 14, 15, 16, 17, 18, 19, 28,
                         29, 32, 33, 30, 31, 36, 37, 34, 35))

    flipped_paf = flipped_paf[:, ::-1, :]

    # The pafs are unit vectors, The x will change direction after flipped.
    flipped_paf[:, :, swap_paf[1::2]] = flipped_paf[:, :, swap_paf[1::2]]
    flipped_paf[:, :, swap_paf[::2]] = -flipped_paf[:, :, swap_paf[::2]]

    averaged_paf = (normal_paf + flipped_paf[:, :, swap_paf]) / 2.
    averaged_heatmap = (normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.

    return averaged_paf, averaged_heatmap


def append_result(image_id, person_to_joint_assoc, joint_list, outputs):
    """
    Build the outputs to be evaluated
    :param image_id: int, the id of the current image
    :param person_to_joint_assoc: numpy array of joints associations
    :param joint_list: list, list of joints
    :param outputs: list of dictionaries with the following keys: image_id,
                    category_id, keypoints, score
    """

    for person in range(len(person_to_joint_assoc)):
        one_result = {
            "image_id": 0,
            "category_id": 1,
            "keypoints": [],
            "score": 0
        }

        one_result["image_id"] = image_id
        keypoints = np.zeros((COCO_num_joints, 3))

        for part in range(COCO_num_joints):
            ind = ORDER_COCO[part]
            index = int(person_to_joint_assoc[person, ind])

            if -1 == index:
                keypoints[part, 0] = 0
                keypoints[part, 1] = 0
                keypoints[part, 2] = 0

            else:
                keypoints[part, 0] = joint_list[index, 0] + 0.5
                keypoints[part, 1] = joint_list[index, 1] + 0.5
                keypoints[part, 2] = 1

        one_result["score"] = person_to_joint_assoc[person, -2] * \
                              person_to_joint_assoc[person, -1]
        one_result["keypoints"] = list(keypoints.reshape(COCO_num_joints * 3))

        outputs.append(one_result)


def run_eval(image_dir, anno_dir, store_dir, image_list_txt, model,
             preprocess='rtpose', size=256, stride=4):
    """
    Run the evaluation on the test set and report mAP score
    :param model: the model to test
    :returns: float, the reported mAP score
    """
    img_ids, img_paths, img_heights, img_widths = get_coco_val(
        image_list_txt)
    print("Total number of validation images {}".format(len(img_ids)))

    # iterate all val images
    outputs = []
    total_time = 0.
    for i in range(len(img_ids)):
        # if i % 100 == 0 and i != 0:
        #     print("Processed {} images".format(i))

        oriImg = cv2.imread(os.path.join(image_dir, 'val2014/' + img_paths[i]))
        begin = time.time()
        # Get results of original imagess
        multiplier = get_multiplier(oriImg, size=size)
        orig_paf, orig_heat = get_multi_scale_output(
            multiplier, oriImg, model, stride=stride, preprocess=preprocess)
        
        # Get results of flipped image
        swapped_img = oriImg[:, ::-1, :]
        flipped_paf, flipped_heat = get_multi_scale_output(multiplier, swapped_img,
                                               model, preprocess=preprocess, stride=stride)
        
        # compute averaged heatmap and paf
        paf, heatmap = get_flipped_output(
            orig_heat, flipped_heat, orig_paf, flipped_paf)
        
         # choose whether use the flipped prediction
         #paf, heatmap = orig_paf, orig_heat
        
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        to_plot, image, joint_list, person_to_joint_assoc = decode_pose(
            oriImg, param, heatmap, paf)
        
        # heatmap = np.expand_dims(heatmap.transpose([2, 0, 1]), 0)
        # paf = np.expand_dims(paf.transpose([2, 0, 1]), 0)
        # people_list = rtpose_postprocess_np(heatmap, paf, num_part=18, param=param)
        # people_list = people_list[0]
        # im_scale = heatmap.shape[2] / float(oriImg.shape[0])
        # people_list[:, :, 0:2] /= im_scale
        # joint_list, person_to_joint_assoc = people_to_pose(people_list)
        # to_plot, image = plot_pose(oriImg, joint_list, person_to_joint_assoc)
        
        # 1 scale 
        #paf, heatmap, scale = get_output(oriImg, model, size=size, preprocess=preprocess)
        #people_list = rtpose_postprocess(heatmap, paf, stride)
        #people_list = people_list[0]
        #people_list[:, :, 0:2] /= scale
        #joint_list, person_to_joint_assoc = people_to_pose(people_list)
        #to_plot, image = plot_pose(oriImg, joint_list, person_to_joint_assoc)

        end = time.time() - begin
        total_time += end
        
        # store_path = os.path.join(store_dir, img_paths[i])
        # cv2.imwrite(store_path, to_plot)

        # person_to_joint_assoc indicated how many peoples found in image.
        append_result(img_ids[i], person_to_joint_assoc, joint_list, outputs)
    
    print("Average time is %.3f" %(total_time/len(img_ids)))
    return eval_coco(outputs=outputs, dataDir=anno_dir, imgIds=img_ids)
