# -*- coding:utf-8 -*-
# @TIME     :2019/2/28 11:22
# @Author   :Fan
# @File     :Train_hourglass.py
"""
Purpose: Train the Hourglass Network to get the prediction
"""
import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')
o_path = os.getcwd()
sys.path.append(o_path)

from Train.Network.Hourglass import Get_Hourglass
from Pretrain.DataLoader import get_loader, get_index

from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict
from tensorboardX import SummaryWriter

#img_size = 256
img_size = 384
#img_size = 512
# stride = img.shape[0] / HeatMap.shape[0]  Need to match the Network
stride = 4

# Get kinds of hyper-parameters
parser = argparse.ArgumentParser(description='Train with Hourglass Network')
parser.add_argument('--data_dir', type=str, default='/Data/COCO/image/')
parser.add_argument('--mask_dir', type=str, default='/Data/COCO/mask/')
parser.add_argument('--json_path', type=str, default='/Data/COCO/COCO.json')
parser.add_argument('--log_dir', type=str, default='Data/Log',
                    help='Where to store the visible tensorboard Log')
parser.add_argument('--model_dir', type=str, default='Data/Stored_model')

parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--gpu_id', type=int, default=[1], dest='gpu_id',
                    help='which gpu to use', nargs="+")
parser.add_argument('--print_frequence', type=int, default=500,
                    help='number of iterations to print the training data')
parser.add_argument('--nesterov', dest='nesterov', action='store_true')


parser.add_argument('--num_stack', type=int, default=3)
parser.add_argument('--num_block', type=int, default=4)
parser.add_argument('--depth', type=int, default=4)

args = parser.parse_args()

# Hourglass Hyper parameters
num_stack = args.num_stack
num_block = args.num_block
depth = args.depth

# Store the parameters
Param_Trans = dict()
# Image Augmentation
Param_Trans['scale_max'] = 1.1
Param_Trans['scale_min'] = 0.5
Param_Trans['target_dist'] = 0.6
Param_Trans['crop_disturb'] = 40
Param_Trans['rotate_degree'] = 40

# HeatMap
# Param_Trans['sigma'] = 4.416
Param_Trans['sigma'] = 7.0

# PAF
# Param_Trans['limb_width'] = 1.289
Param_Trans['limb_width'] = 1.0

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu_id)

# create the name space
def build_name(stack_num=num_stack):
    name = []
    for i in range(stack_num):
        for j in range(2):
            name.append('loss_stage%d_L%d' % (i+1, j+1))

    return name

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

# Calculate the loss during the training including intermediate supervision
def get_loss(mediat_pred, pred, heatmap, heatmap_mask, paf, paf_mask):
    if torch.cuda.is_available():
        loss_func = nn.MSELoss().cuda()
    else:
        loss_func = nn.MSELoss()

    total_loss = 0
    number = int(len(mediat_pred) / 2)
    log = OrderedDict()
    name = build_name(number+1)

    # Calculate the ground-truth
    gt_heatmap = heatmap * heatmap_mask
    gt_paf = paf * paf_mask

    # Calculate the mediate training loss
    for i in range(number):
        pred_paf = mediat_pred[2 * i] * paf_mask
        pred_heatmap = mediat_pred[2 * i + 1] * heatmap_mask

        loss_paf = loss_func(pred_paf, gt_paf)
        loss_heatmap = loss_func(pred_heatmap, gt_heatmap)

        log[name[2 * i]] = loss_paf.item()
        log[name[2 * i + 1]] = loss_heatmap.item()

        total_loss += loss_paf
        total_loss += loss_heatmap

    # Calculate the last loss
    pred_paf = pred[0] * paf_mask
    pred_heatmap = pred[1] * heatmap_mask
    loss_paf = loss_func(pred_paf, gt_paf)
    loss_heatmap = loss_func(pred_heatmap, gt_heatmap)

    log[name[2 * number]] = loss_paf.item()
    log[name[2 * number + 1]] = loss_heatmap.item()

    log['max_ht'] = torch.max(pred[1].data).item()
    log['min_ht'] = torch.min(pred[1].data).item()
    log['max_paf'] = torch.max(pred[0].data).item()
    log['min_paf'] = torch.min(pred[0].data).item()

    total_loss += loss_paf
    total_loss += loss_heatmap

    del loss_func

    return total_loss, log

def find_lr(train_loader, model, init_value=1e-8, final_value=1.):
    num = 5000
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    loss = AverageMeter()

    location = 'lr_{}_{}_{}'.format(num_stack, num_block, depth)
    writer = SummaryWriter(log_dir=os.path.join(sys.path[0], args.log_dir, location))
    model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i, (img, HeatMap, heat_mask, PAF, paf_mask) in enumerate(train_loader):
        if (i + 1) % (num + 1) == 0:
            break
        img = img.cuda()
        HeatMap = HeatMap.cuda()
        heat_mask = heat_mask.cuda()
        PAF = PAF.cuda()
        paf_mask = paf_mask.cuda()

        pred, mediate_pred = model(img)
        total_loss, _ = get_loss(mediate_pred, pred, HeatMap, heat_mask,
                                   PAF, paf_mask)

        loss.update(float(total_loss), img.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalars('data/scalar_group', {'loss':loss.avg}, np.log10(lr))
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    writer.close()
    return lr

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    meter_dict = {}
    for name in build_name():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()
    meter_dict['max_paf'] = AverageMeter()
    meter_dict['min_paf'] = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, HeatMap, heat_mask, PAF, paf_mask) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        HeatMap = HeatMap.cuda()
        heat_mask = heat_mask.cuda()
        PAF = PAF.cuda()
        paf_mask = paf_mask.cuda()

        # compute output
        try:
            pred, mediate_pred = model(img)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        total_loss, log = get_loss(mediate_pred, pred, HeatMap, heat_mask,
                                             PAF, paf_mask)

        for name, _ in meter_dict.items():
            meter_dict[name].update(log[name], img.size(0))
        losses.update(float(total_loss), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # refresh the storage
        del total_loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_frequence == 0:
            print_string = 'Epoch: [{}][{}/{}]\t'.format(epoch, i, len(train_loader))
            print_string += 'Data time {data_time.value:.3f} ({data_time.avg:.3f})\t\t'.format(data_time=data_time)
            print_string += 'Loss {loss.value:.4f} ({loss.avg:.4f})\n'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string += '{name}: {loss.value:.4f} ({loss.avg:.4f})\n'.format(name=name, loss=value)
            print(print_string + '\n')
    return losses.avg


def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, HeatMap, heat_mask, PAF, paf_mask) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            img = img.cuda()
            HeatMap = HeatMap.cuda()
            heat_mask = heat_mask.cuda()
            PAF = PAF.cuda()
            paf_mask = paf_mask.cuda()

            # compute output
            try:
                pred, mediate_pred = model(img)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            total_loss, log = get_loss(mediate_pred, pred, HeatMap, heat_mask,
                                                 PAF, paf_mask)

            losses.update(float(total_loss), img.size(0))

            # refresh the storage
            del total_loss

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_frequence == 0:
                print_string = 'Epoch: [{}][{}/{}]\t'.format(epoch, i, len(val_loader))
                print_string += 'Data time {data_time.value:.3f} ({data_time.avg:.3f})\t\t'.format(data_time=data_time)
                print_string += 'Loss {loss.value:.4f} ({loss.avg:.4f})\n'.format(loss=losses)

                # for name, value in meter_dict.items():
                #     print_string += '{name}: {loss.value:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
                print(print_string)

    return losses.avg

# Get index
train_index, val_index, data = get_index(args.json_path)

# train data
train_data = get_loader(train_index, val_index, data, args.data_dir, args.mask_dir,
                        img_size=img_size, stride=stride, Preprocess='rtpose',
                        batch_size=args.batch_size, Param_Trans=Param_Trans,
                        shuffle=True, train=True, num_worker=2)

print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = get_loader(train_index, val_index, data, args.data_dir, args.mask_dir,
                        img_size=img_size, stride=stride, Preprocess='rtpose',
                        batch_size=args.batch_size, Param_Trans=Param_Trans,
                        shuffle=False, train=False, num_worker=2)

print('val dataset len: {}'.format(len(valid_data.dataset)))

del train_index
del val_index
del data

# model
model = Get_Hourglass(num_stacks=num_stack, num_blocks=num_block, paf_classes=38,
                      ht_classes=19, depth=depth)

# lr = find_lr(train_data, model)
# print('lr:%.3f' % lr)
model_save_filename = sys.path[0] + \
                      '/Data/Stored_model/pose_Hourglass_{}_{}_{}.pth'.format(num_stack, num_block, depth)

if os.path.exists(model_save_filename):
    model.load_state_dict(torch.load(model_save_filename))
    print('Exist such Network Hourglass_{}_{}_{}, Load it'.format(num_stack, num_block, depth))
# model = torch.nn.DataParallel(model).cuda()
model = model.cuda()

location = 'Net_{}_{}_{}'.format(num_stack, num_block, depth)
writer = SummaryWriter(log_dir=os.path.join(sys.path[0], args.log_dir, location))

trainable_vars = [param for param in model.parameters() if param.requires_grad]

#optimizer = torch.optim.SGD(trainable_vars, lr=args.learning_rate,
#                            momentum=args.momentum,
#                            weight_decay=args.weight_decay,
#                            nesterov=args.nesterov)

optimizer = torch.optim.Adam(trainable_vars, lr=args.learning_rate, weight_decay=args.weight_decay)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True,
                                 threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=1e-08)


# Load best val loss

json_file = os.path.join(sys.path[0], args.model_dir,
            "tensorboard/all_scalars_{}_{}_{}.json".format(num_stack, num_block, depth))

if not os.path.exists(json_file):
    with open(json_file, 'w') as f:
        f.write('\n')
    #best_val_loss = 0.0070145      # 324
    #best_val_loss = 0.0063671      # 344
    best_val_loss = 0.0078487
    #best_val_loss = np.inf
    print('Load best val loss: %f \n' % best_val_loss)
else:
    with open(json_file, 'r') as f:
        loss = json.loads(f.read())
        log_dir = os.path.join(sys.path[0], args.log_dir, location)
        val = loss['{}/data/scalar_group/val loss'.format(log_dir)]
        val.sort(key=lambda k: k[-1])
        min_val = val[0][-1]
    best_val_loss = float(min_val)
    print('Load best val loss: %f \n' % best_val_loss)


for epoch in range(args.epoch):
    # train for one epoch
    train_loss = train(train_data, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(valid_data, model, epoch)

    writer.add_scalars('data/scalar_group', {'train loss': train_loss,
                                             'val loss': val_loss,
                                             'learning_rate': optimizer.param_groups[0]['lr']}, epoch)

    lr_scheduler.step(val_loss)

    is_best = float(val_loss) < best_val_loss
    best_val_loss = min(float(val_loss), best_val_loss)

    if is_best:
        torch.save(model.state_dict(), model_save_filename)
    
    if (epoch + 1) % 70 == 0:
        lr = optimizer.param_groups[0]['lr'] / 2.
        optimizer.param_groups[0]['lr'] = lr
        
    del train_loss
    del val_loss


writer.export_scalars_to_json(json_file)
writer.close()



