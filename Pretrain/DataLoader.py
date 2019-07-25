# -*- coding:utf-8 -*-
# @TIME     :2019/2/25 17:06
# @Author   :Fan
# @File     :DataLoader.py
"""
Purpose: Generate COCO DataLoader
"""

import json
import logging
import sys

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from Pretrain.Data import COCOkeypoints

# print the special control information
logger = logging.getLogger(__name__)


class sDataLoader(DataLoader):
    def get_stream(self):
        # Return a Generator that can yield endless data

        while True:
            for data in iter(self):
                yield data

    @staticmethod
    def copy(loader):
        # Init a sDataLoader from an existing DataLoader

        if not isinstance(loader, DataLoader):
            logger.warning('loader should be an instance of Dataloader,'
                           'but get {}'.format(type(loader)))
            return loader

        new_loader = sDataLoader(loader.dataset)
        for k, v in loader.__dict__.items():
            setattr(new_loader, k, v)
        return new_loader


def get_index(json_path):
    with open(sys.path[0] + json_path) as file:
        reader = json.load(file)
        data = reader['root']

    num_samples = len(data)
    train_index = []
    val_index = []

    # load the train and val data
    for i in range(num_samples):
        if data[i]['isValidation'] != 0.:
            val_index.append(i)
        else:
            train_index.append(i)

    return train_index, val_index, data


def get_loader(train_index, val_index, data, data_dir, mask_dir, img_size, stride, Preprocess,
               batch_size, Param_Trans, train=True, shuffle=True, num_worker=0):

    coco_data = COCOkeypoints(root=data_dir, mask_dir=mask_dir,
                              index_list=train_index if train else val_index,
                              data=data, img_size=img_size, stride=stride,
                              Preprocess=Preprocess, transform=ToTensor(),
                              Param_Trans=Param_Trans)

    data_loader = sDataLoader(coco_data, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_worker)

    return data_loader


