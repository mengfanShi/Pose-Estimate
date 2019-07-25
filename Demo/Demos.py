# -*- coding: utf-8 -*-
# @Time     : 2019/4/24 10:13
# @Author   : Fan
# @File     : Demos.py

import torch
import os
import cv2
import time
import sys
sys.path.insert(0, '..')
import threading
from Train.Network.Hourglass import Get_Hourglass
from Train.Network.rtpose_vgg import get_model

from Eval.COCO_eval import get_output
from Eval.Parser import plot_pose, people_to_pose
from Eval.cpm.cpm_layer import rtpose_postprocess
from Demo.gui import Gui

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load Hourglass Network
def load_model(num_stack=3, num_block=3, depth=4):
	weight_name = sys.path[0] + '/Data/Stored_model/' + \
	              'pose_Hourglass_plus_{}_{}_{}.pth'.format(num_stack, num_block, depth)
	model = Get_Hourglass(num_stacks=num_stack, num_blocks=num_block,
	                      paf_classes=38, ht_classes=19, depth=depth)
	model.load_state_dict(torch.load(weight_name), strict=False)
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
	size = 256
	mygui = Gui()
	model = load_model()
	mygui.begin()

	path = mygui.filepath
	id = mygui.id
	if id == 0:
		img = cv2.imread(path)
		start = time.time()
		with torch.no_grad():
			paf, heatmap, scale = get_output(img, model, size=size, preprocess='rtpose')
			people_list = rtpose_postprocess(heatmap, paf, 4, nms_thr=0.1)
			people_list = people_list[0]
			people_list[:, :, 0:2] /= scale
			joint_list, person_to_joint_assoc = people_to_pose(people_list)
			to_plot, image = plot_pose(img, joint_list, person_to_joint_assoc)
		used_time = time.time() - start
		print('used_time:%.5f' %used_time)
		cv2.imshow('result', to_plot)
		key = cv2.waitKey(0)
		if key == 27:
			cv2.destroyAllWindows()
		# cv2.imwrite(sys.path[0] + '/Demo/to_plot.jpg', to_plot)
	if id == 1:
		Video = cv2.VideoCapture(path)
		while True:
			ret, frame = Video.read()
			with torch.no_grad():
				paf, heatmap, scale = get_output(frame, model, size=size, preprocess='rtpose')
				people_list = rtpose_postprocess(heatmap, paf, 4, nms_thr=0.1)
				people_list = people_list[0]
				people_list[:, :, 0:2] /= scale
				joint_list, person_to_joint_assoc = people_to_pose(people_list)
				to_plot, image = plot_pose(frame, joint_list, person_to_joint_assoc)
			cv2.imshow('Video', to_plot)
			key = cv2.waitKey(1)
			if key == 27:
				break
		Video.release()
		cv2.destroyAllWindows()
