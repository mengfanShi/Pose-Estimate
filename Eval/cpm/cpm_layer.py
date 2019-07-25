import numpy as np
import torch
import torch.nn.init
from torch.autograd import Function

from ._ext import cpm


class ResizeFucntion(Function):

    def __init__(self, factor, start_scale, scale_gap):
        self.factor = factor
        self.start_scale = start_scale
        self.scale_gap = scale_gap

    def forward(self, input):

        assert input.is_cuda, 'only support cuda tensor'

        height, width = input.size()[2:4]
        target_height, target_width = height * self.factor, width * self.factor

        output = torch.FloatTensor().cuda(input.get_device())
        cpm.resize_forward(input, output, target_height, target_width, self.start_scale, self.scale_gap)

        return output

    def backward(self, grad_top):
        assert False


class NMSFunction(Function):
    def __init__(self, num_parts, max_peaks, threshold):
        self.num_parts = num_parts
        self.max_peaks = max_peaks
        self.threshold = threshold

    def forward(self, input):
        assert input.is_cuda, 'only support cuda tensor'
        output = torch.FloatTensor().cuda(input.get_device())
        cpm.cpm_nms_forward(input, output, self.num_parts, self.max_peaks, self.threshold)

        return output


class ConnectLimbsFuntion(Function):
    def __init__(self, min_subset_score=0.05, min_subset_cnt=3, inter_threshold=0.050, inter_min_above_threshold=9):
        self.min_subset_cnt = min_subset_cnt
        self.min_subset_score = min_subset_score

        self.inter_threshold = inter_threshold
        self.inter_min_above_threshold = inter_min_above_threshold

    def forward(self, heatmap, peak):
        assert not heatmap.is_cuda, 'ConnectFunction only support cpu tensor'
        joints = torch.FloatTensor()
        num_joints = torch.IntTensor()
        cpm.limbs_coco_cpu(heatmap, peak, joints, num_joints,
                           self.min_subset_cnt, self.min_subset_score,
                           self.inter_threshold, self.inter_min_above_threshold)

        return num_joints, joints


def rtpose_postprocess(pred_heatmap, pred_paf, feat_stride=4, scale_gap=0.1, num_part=18, nms_thr=0.1):
    """
    postprocess based on caffe rtpose
    :param pred_heatmap: cuda Variable
    :param pred_paf: cuda Variable
    :param feat_stride:
    :param num_part:
    :param nms_thr:
    :return:
    people_list: a list of numpy array [batch_size, num_people, num_part, 3] => [x, y, score] relative to heatmap size
    """

    predict_concat = torch.cat([pred_heatmap, pred_paf], 1)

    # cuda tensor
    resized_map = ResizeFucntion(feat_stride, start_scale=1, scale_gap=scale_gap)(predict_concat)
    peaks = NMSFunction(num_part, 200, nms_thr)(resized_map)
    
    # cpu tensor
    num_joints, joints = ConnectLimbsFuntion()(resized_map.cpu(), peaks.cpu())
    
    num_joints_np = num_joints.data.numpy()
    joints_np = joints.data.numpy()
    people_list = []
    for i in range(len(num_joints_np)):
        people_list.append(joints_np[i, 0:num_joints_np[i], 0:18])

    return people_list


def rtpose_postprocess_np(heatmap, paf, num_part=18, gpu=0, param=None):
    if param is None:
        # nms threshold and connect threshold
        param = {'thre1': 0.1, 'thre2': 0.05}

    heatmap_data = torch.from_numpy(heatmap.astype(np.float32)).cuda(gpu)
    paf_data = torch.from_numpy(paf.astype(np.float32)).cuda(gpu)

    # rtpose_postprocess(heatmap_data, paf_data, 8)

    predict_concat = torch.cat([heatmap_data, paf_data], 1)

    peaks = NMSFunction(num_part, 200, param['thre1']).forward(predict_concat)

    num_joints, joints = ConnectLimbsFuntion(inter_threshold=param['thre2']).forward(predict_concat.cpu(), peaks.cpu())

    num_joints_np = num_joints.numpy()
    joints_np = joints.numpy()
    people_list = []
    for i in range(len(num_joints_np)):
        people_list.append(joints_np[i, 0:num_joints_np[i], 0:18])

    return people_list
