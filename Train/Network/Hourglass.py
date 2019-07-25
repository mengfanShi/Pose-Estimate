# -*- coding:utf-8 -*-
# @TIME     :2019/2/26 9:54
# @Author   :Fan
# @File     :Hourglass.py
"""
Purpose: Build a Hourglass network
"""

import torch.nn as nn
import torch.nn.functional as func

class Bottleneck(nn.Module):
    def __init__(self, channel_in, channel_mid, stride=1, expansion=2):
        super(Bottleneck, self).__init__()

        self.BN1 = nn.BatchNorm2d(channel_in)

        # self.Conv1 = nn.Conv2d(channel_in, channel_mid,
        #                        kernel_size=1, bias=True)
        self.Conv1 = nn.Conv2d(channel_in, channel_mid,
                               kernel_size=1, bias=False)
        
        self.BN2 = nn.BatchNorm2d(channel_mid)


        self.Conv2 = nn.Conv2d(channel_mid, channel_mid, bias=False,
                               kernel_size=3, stride=stride, padding=1)

        # Depth-wise Convolution
        #self.Conv2 = nn.Conv2d(channel_mid, channel_mid, bias=False,
        #                       kernel_size=3, stride=stride, padding=1,
        #                       groups=channel_mid)

        self.BN3 = nn.BatchNorm2d(channel_mid)

        self.Conv3 = nn.Conv2d(channel_mid, channel_mid * expansion,
                               kernel_size=1, bias=True)

        self.ReLU = nn.ReLU(inplace=True)

        # make sure the channels are the same
        self.identity = (channel_in == channel_mid * expansion) and (stride == 1)
        # self.Conv = nn.Conv2d(channel_in, channel_mid * expansion,
        #                       kernel_size=1, bias=True, stride=stride)
        self.Conv = nn.Conv2d(channel_in, channel_mid * expansion,
                              kernel_size=1, bias=False, stride=stride)

    def shuffle(self, x, group):
        x = x.reshape(x.shape[0], group, x.shape[1] // group, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x

    def forward(self, x):
        if self.identity:
            residual = x
        else:
            residual = self.Conv(x)

        out = self.BN1(x)
        out = self.ReLU(out)

        out = self.Conv1(out)
        out = self.BN2(out)
        out = self.ReLU(out)

        # out = self.shuffle(out, group=3)
        out = self.Conv2(out)
        out = self.BN3(out)
        # if use depth-wise, abandon the next relu
        out = self.ReLU(out)

        out = self.Conv3(out)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_block, channels, depth, expansion=2):
        super(Hourglass, self).__init__()

        self.depth = depth
        self.expansion = expansion
        self.dowmsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hourglass = self.get_hourglass(block, num_block,
                                            channels, depth)

    def get_residual(self, block, num_block, channels):
        # add Bottleneck block to build residual block
        layers = []
        for i in range(num_block):
            layers.append(block(channels * self.expansion, channels))
        return nn.Sequential(*layers)

    def get_hourglass(self, block, num_blocks, channels,
                      depth, num_residual=3):
        hourglass = []
        for i in range(depth):
            residual = []
            for j in range(num_residual):
                residual.append(
                    self.get_residual(block, num_blocks, channels))

            if i == 0:
                residual.append(
                    self.get_residual(block, num_blocks, channels))

            hourglass.append(nn.ModuleList(residual))

            del residual

        return nn.ModuleList(hourglass)

    def hourglass_forward(self, depth, x):
        identity = self.hourglass[depth - 1][0](x)

        downsample = self.dowmsample(x)
        downsample = self.hourglass[depth - 1][1](downsample)

        # Do the recurrence to build the Hourglass formation
        if depth > 1:
            downsample2 = self.hourglass_forward(depth - 1, downsample)
        else:
            downsample2 = self.hourglass[depth - 1][3](downsample)
            # the number is 3 means only the 1st layer exists 4 residual

        downsample3 = self.hourglass[depth - 1][2](downsample2)

        upsample = func.interpolate(downsample3,
                                    size=(identity.shape[-2], identity.shape[-1]))
        #upsample = func.upsample(downsample3,
        #                            size=(identity.shape[-2], identity.shape[-1]),
        #                            mode='bilinear', align_corners=True)
        out = identity + upsample
        return out

    def forward(self, x):
        return self.hourglass_forward(self.depth, x)


class HourglassNet(nn.Module):
    # Project conclude 19 parts of limbs and 18 joints
    # num_stack means the count of the structure of Hourglass
    # num_block means the count of the structure of bottleneck
    # depth means the count of the recurrence Hourglass

    def __init__(self, block, channel_in=32, features=64, num_stack=2, num_block=4,
                 depth=2, paf_class=19*2, heatmap_class=18+1, expansion=2):
        super(HourglassNet, self).__init__()

        #self.channel_in = 64
        #self.features = 128
        self.channel_in = 32
        self.features = 64
        self.num_stack = num_stack
        self.depth = depth
        self.expansion = expansion

        # Quick dimension reduction
        self.Conv1 = nn.Conv2d(3, self.channel_in, kernel_size=7,
                               stride=2, padding=3, bias=True)
        self.BN1 = nn.BatchNorm2d(self.channel_in)
        self.Layer1 = self.expand_features(block, channels=self.channel_in,
                                           channel_in=self.channel_in)
        self.Layer2 = self.expand_features(block, channels=self.channel_in*self.expansion,
                                           channel_in=self.channel_in*self.expansion)
        self.Layer3 = self.expand_features(block, channels=self.features,
                                           channel_in=self.channel_in*self.expansion*self.expansion)

        self.ReLU = nn.ReLU(inplace=True)
        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Get Hourglass Module
        hourglass, residual, fc, score_paf, score_heatmap, fc_, \
        mid_paf, mid_heatmap = self.get_hourglass(block, num_stack,
                                                  num_block, paf_class,
                                                  heatmap_class)

        self.hourglass = nn.ModuleList(hourglass)
        self.residual = nn.ModuleList(residual)
        self.fc = nn.ModuleList(fc)
        self.score_paf = nn.ModuleList(score_paf)
        self.score_heatmap = nn.ModuleList(score_heatmap)
        self.fc_ = nn.ModuleList(fc_)
        self.mid_paf = nn.ModuleList(mid_paf)
        self.mid_heatmap = nn.ModuleList(mid_heatmap)

        self.init_weight()


    def get_residual(self, block, num_block, channels):
        layers = []

        for i in range(num_block):
            layers.append(block(channels * self.expansion, channels))

        return nn.Sequential(*layers)

    def expand_features(self, block, channel_in, channels, stride=1):
        layers = []
        layers.append(block(channel_in, channels, stride))
        return nn.Sequential(*layers)

    def get_fc(self, channel_in, channel_out):
        Conv = nn.Conv2d(channel_in, channel_out, kernel_size=1, bias=True)
        BN = nn.BatchNorm2d(channel_out)
        ReLU = nn.ReLU(inplace=True)
        fc = nn.Sequential(Conv, BN, ReLU)
        return fc

    # Build Hourglass Module
    def get_hourglass(self, block, num_stack, num_block, paf_class, heatmap_class):
        hourglass, residual, fc, score_paf, score_heatmap, \
        fc_, mid_paf, mid_heatmap = [], [], [], [], [], [], [], []

        for i in range(num_stack):
            hourglass.append(Hourglass(block, num_block, self.features, self.depth))
            residual.append(self.get_residual(block, num_block, self.features))
            fc.append(self.get_fc(self.features * self.expansion,
                                  self.features * self.expansion))
            score_paf.append(nn.Conv2d(self.features * self.expansion, paf_class,
                                       kernel_size=1, bias=True))
            score_heatmap.append(nn.Conv2d(self.features * self.expansion, heatmap_class,
                                           kernel_size=1, bias=True))

            # intermediate supervision
            if i < num_stack - 1:
                fc_.append(nn.Conv2d(self.features * self.expansion,
                                     self.features * self.expansion,
                                     kernel_size=1, bias=True))
                mid_paf.append(nn.Conv2d(paf_class, self.features * self.expansion,
                                         kernel_size=1, bias=True))
                mid_heatmap.append(nn.Conv2d(heatmap_class, self.features * self.expansion,
                                             kernel_size=1, bias=True))

        return hourglass, residual, fc, score_paf, score_heatmap, fc_, mid_paf, mid_heatmap

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        mediate_pred = []
        score_paf = 0
        score_heatmap = 0

        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLU(x)

        x = self.Layer1(x)
        x = self.Pool(x)
        x = self.Layer2(x)
        x = self.Layer3(x)

        for i in range(self.num_stack):
            y = self.hourglass[i](x)
            y = self.residual[i](y)
            y = self.fc[i](y)
            score_paf = self.score_paf[i](y)
            score_heatmap = self.score_heatmap[i](y)

            if i < self.num_stack - 1:
                fc_ = self.fc_[i](y)
                mid_paf = self.mid_paf[i](score_paf)
                mid_heatmap = self.mid_heatmap[i](score_heatmap)
                x = x + fc_ + mid_paf + mid_heatmap

                mediate_pred.append(score_paf)
                mediate_pred.append(score_heatmap)

        pred = (score_paf, score_heatmap)

        return pred, mediate_pred


def Get_Hourglass(**kwargs):
    model = HourglassNet(block=Bottleneck, num_stack=kwargs['num_stacks'],
                         num_block=kwargs['num_blocks'],
                         paf_class=kwargs['paf_classes'],
                         heatmap_class=kwargs['ht_classes'],
                         depth=kwargs['depth'])

    return model
