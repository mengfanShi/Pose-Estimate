# -*- coding:utf-8 -*-
# @TIME     :2019/3/11 16:02
# @Author   :Fan
# @File     :eval.py

import torch
import os
import sys
sys.path.insert(0, '..')
import argparse
from Eval.COCO_eval import run_eval
from Train.Network.Hourglass import Get_Hourglass
from Train.Network.rtpose_vgg import get_model

Parse = argparse.ArgumentParser(description='Type of image')
Parse.add_argument('--img_size', type=int, default=256)
Parse.add_argument('--stride', type=int, default=4)
args = Parse.parse_args()

gpu_id = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# Load Hourglass Network
def load_model(num_stack=4, num_block=3, depth=4):
	weight_name = sys.path[0] + '/Data/Stored_model/' + \
	              'pose_Hourglass_{}_{}_{}.pth'.format(num_stack, num_block, depth)
	model = Get_Hourglass(num_stacks=num_stack, num_blocks=num_block,
	                      paf_classes=38, ht_classes=19, depth=depth)
	print('pose_Hourglass_{}_{}_{}'.format(num_stack, num_block, depth))
	model.load_state_dict(torch.load(weight_name))
	model.eval()
	model.float()
	model = model.cuda()
	return model

def load_vgg():
	weight_name = sys.path[0] + '/Data/Stored_model/'+'pose_model.pth'

	model = get_model('vgg19')     
	model.load_state_dict(torch.load(weight_name), strict=False)
	model.float()
	model.eval()
	model = model.cuda()
	return model

if __name__ == '__main__':
    model = load_model()
    with torch.no_grad():
        AP = run_eval(image_dir=sys.path[0] + '/Data/COCO/image',
                 anno_dir=sys.path[0] + '/Data/COCO',
                 store_dir=sys.path[0] + '/Data/Stored_image',
                 image_list_txt=sys.path[0] + '/Data/COCO/image_info_val2014_1k.txt',
                 model=model, preprocess='rtpose',
                 size=args.img_size, stride=args.stride)
        print('\nThe Average Precision is %.3f' % AP)
