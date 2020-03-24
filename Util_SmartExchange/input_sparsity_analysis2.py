import numpy as np
import torch
import torch.nn as nn
import sys
import os
import argparse
from tqdm import tqdm

from data import *
from ThiNet import *
import resnet
import resnet_cifar
import vgg
import vgg_cifar
from efficientnet import *
from mobilenet_v1 import *
from mobilenet_v2 import *


parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10 training with gating')
parser.add_argument('--arch', default='', help='select a model')
parser.add_argument('--dataset', '-d', type=str, default='imagenet',
                    choices=['cifar10', 'cifar100','imagenet'],
                    help='dataset choice')
parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str,
                    help='path to  latest checkpoint (default: None)')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size')
parser.add_argument('--save_folder', default='./',
                    type=str,
                    help='folder to save the extracted info')
args = parser.parse_args()

def turn_binary(a):
    b = bin(a)[2:]

    pad = ''
    for _ in range(8-len(b)):
        pad += '0'

    return pad + b


def bit_sparsity(a):
    if type(a) != 'str':
        a = turn_binary(int(a))

    cnt = 0
    for bit in a:
        if bit == '0':
            cnt += 1
    return cnt


def bit_sparsity_booth(a):
    if type(a) != 'str':
        a = turn_binary(int(a))

    a =  a + '0' 

    booth_list = [a[0:3], a[2:5], a[4:7], a[6:9]]

    cnt = 0
    for item in booth_list:
        if '0' not in item or '1' not in item:
            cnt += 1

    return cnt


def layer_bit_sparsity(act):
    act = np.reshape(act, [-1]).astype(np.int32)
    cnt = 0
    for element in act:
        cnt += bit_sparsity(element)

    return cnt, len(act) * 8


def layer_bit_sparsity_booth(act):
    act = np.reshape(act, [-1]).astype(np.int32)
    cnt = 0
    for element in act:
        cnt += bit_sparsity_booth(element)

    return cnt, len(act) * 4


def layer_vector_sparsity(act, kernel_size, stride, padding, vector_length=8):
    act = act.astype(np.int32)

    if padding:
        act_pad = np.zeros([act.shape[0], act.shape[1], act.shape[2] + 2*padding, act.shape[3] + 2*padding])
        act_pad[:,:,padding:-padding, padding:-padding] = act
        act = act_pad

    vector_num_per_row = act.shape[2] // (vector_length * stride)

    shape3 = vector_num_per_row if vector_num_per_row * vector_length * stride == act.shape[2] else vector_num_per_row + 1

    total_vector_num = act.shape[0]*act.shape[1]*act.shape[2]*vector_num_per_row*shape3

    cnt = 0

    if kernel_size != 1:
        total_vector_num = act.shape[0] * act.shape[1] * act.shape[2] * vector_num_per_row * shape3

        for img_id in range(act.shape[0]):
            for channel in range(act.shape[1]):
                for row in range(act.shape[2]):                        
                    for n in range(vector_num_per_row):
                        lower_bound = n * 8 * stride
                        upper_bound = min((n+1) * 8 * stride + kernel_size - 1, act.shape[3])
                        vector = act[img_id, channel, row, lower_bound : upper_bound]
                        
                        if len(vector):
                            cnt += 1
                            for element in vector:
                                if element != 0:
                                    cnt -= 1
                                    break
    else:
        channel_group_num = act.shape[1] // 3
        shape1 = channel_group_num if channel_group_num * 3 == act.shape[1] else channel_group_num + 1

        total_vector_num = act.shape[0] * shape1 * act.shape[2] * vector_num_per_row * shape3

        for img_id in range(act.shape[0]):
            for channel_group in range(channel_group_num):
                for row in range(act.shape[2]):                        
                    for n in range(vector_num_per_row):
                        width_lower_bound = n * 8 * stride
                        width_upper_bound = min((n+1) * 8 * stride + kernel_size - 1, act.shape[3])
                        channel_lower_bound = channel_group * 3
                        channel_upper_bound = min((channel_group+1) * 3, act.shape[1]) 

                        vector = act[img_id, channel_lower_bound : channel_upper_bound, row, width_lower_bound : width_upper_bound]
                        vector = np.reshape(vector, [-1])

                        if len(vector):
                            cnt += 1
                            for element in vector:
                                if element != 0:
                                    cnt -= 1
                                    break

    return cnt, total_vector_num


if args.arch == 'vgg11':
    cfg = [38, 'M', 127, 'M', 232, 255, 'M', 491, 511, 'M', 512, 501, 'M']
    model = vgg.vgg11_bn(pretrained=False, config=cfg)

elif args.arch == 'thinet50':
    model = thinet50()

elif args.arch == 'thinet70':
    model = thinet70()

elif args.arch == 'resnet50':
    model = resnet.resnet50()

elif args.arch == 'vgg19':
    checkpoint_cfg = torch.load('/home/yw68/network-slimming/pruned_vgg19_75/pruned.pth.tar')
    model = vgg_cifar.vgg19(pretrained=False, cfg_vgg19=checkpoint_cfg['cfg'])

elif args.arch == 'mobilenet_v1':
    model = mobilenet_v1()   

elif args.arch == 'mobilenet_v2':
    model = mobilenet_v2()   

elif args.arch == 'resnet164':
    checkpoint_cfg = torch.load('/home/yw68/network-slimming/pruned_resnet164_50/pruned.pth.tar')
    model = resnet_cifar.ResNet(cfg=checkpoint_cfg['cfg'], num_classes=10)

else:
    print('No such model.')
    sys.exit()


if os.path.isfile(args.resume):
    print('=> loading checkpoint `{}`'.format(args.resume))
    checkpoint = torch.load(args.resume)
    state_dict = {k.replace('module.','') : v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    print('=> loaded checkpoint `{}` (epoch: {})'.format(
        args.resume, checkpoint['epoch']
    ))
else:
    print('=> no checkpoint found at `{}`'.format(args.resume))


test_loader = prepare_test_data(dataset=args.dataset,
                                  datadir=args.datadir+'/val' if 'imagenet' in args.dataset else args.datadir,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1)

modules = list(model.modules())

input_conv_list = []
def forward_hook_conv(module, input, output):
    input_max = input[0].max()
    input_min = input[0].min()
    input_quant = ((input[0]-input_min)/(input_max-input_min)*(2**8-1)).clamp_(0, 2**8-1).round_().cpu().data.numpy()
    input_conv_list.append(input_quant)

input_linear_list = []
def forward_hook_linear(module, input, output):
    input_max = input[0].max()
    input_min = input[0].min()
    input_quant = ((input[0]-input_min)/(input_max-input_min)*(2**8-1)).clamp_(0, 2**8-1).round_().cpu().data.numpy()
    input_linear_list.append(input_quant)


conv_list = []
weight_conv_list = []
linear_list = []
weight_linear_list = []
hook_list = []

for module in modules:
    if isinstance(module, nn.Conv2d):
        conv_list.append(module)
        weight_conv_list.append(module.weight.cpu().data.numpy())
        hook_list.append(module.register_forward_hook(forward_hook_conv))
    if isinstance(module, nn.Linear):
        linear_list.append(module)
        weight_linear_list.append(module.weight.cpu().data.numpy())
        hook_list.append(module.register_forward_hook(forward_hook_linear))

for i, (input_data, target) in enumerate(test_loader):
    _ = model(input_data)
    break

for hook in hook_list:
    hook.remove()

HW = []
Padding = []
C = []
B = []
RS = []
M = []
U = []
group = []


for i, item in enumerate(conv_list):

    print('layer:', i+1)

    HW.append(input_conv_list[i].shape[2])
    print('input size:', input_conv_list[i].shape[2])

    Padding.append(item.padding[0])
    print('padding:',item.padding[0])

    C.append(item.weight.size(1))
    print('input channel:',item.weight.size(1))

    B.append(1)

    RS.append(item.kernel_size[0])
    print('kernel size:',item.kernel_size[0])

    M.append(item.weight.size(0))
    print('output channel:',item.weight.size(0))

    U.append(item.stride[0])
    print('stride:', item.stride[0])

    group.append(item.groups)
    print('group',item.groups)

    print('####')

conv_info = {'HW':HW, 'Padding':Padding, 'C':C, 'B':B, 'RS':RS, 'M':M, 'U':U, 'group':group}


bit_level = []
bit_level_total = []

booth = []
booth_total = []

vector8 = []
vector8_total = []

vector4 = []
vector4_total = []

for i, act in enumerate(input_conv_list):
    print('dealing with layer ', i)
    cnt_sparsity, total = layer_bit_sparsity(act)
    bit_level.append(cnt_sparsity)
    bit_level_total.append(total)

    cnt_sparsity, total = layer_bit_sparsity_booth(act)
    booth.append(cnt_sparsity)
    booth_total.append(total)

    cnt_sparsity, total = layer_vector_sparsity(act, kernel_size=conv_info['RS'][i], stride=conv_info['U'][i], padding=conv_info['Padding'][i], vector_length=8)
    vector8.append(cnt_sparsity)
    vector8_total.append(total)

    cnt_sparsity, total = layer_vector_sparsity(act, kernel_size=conv_info['RS'][i], stride=conv_info['U'][i], padding=conv_info['Padding'][i], vector_length=4)
    vector4.append(cnt_sparsity)
    vector4_total.append(total)


bit_level = np.array(bit_level) / args.batch_size 
bit_level_total = np.array(bit_level_total) / args.batch_size 
booth = np.array(booth) / args.batch_size 
booth_total = np.array(booth_total) / args.batch_size 
vector8 = np.array(vector8) / args.batch_size 
vector8_total = np.array(vector8_total) / args.batch_size 
vector4 = np.array(vector4) / args.batch_size 
vector4_total = np.array(vector4_total) / args.batch_size 


sparsity_dict = {'bit_level':bit_level, 'bit_level_total':bit_level_total, 'booth':booth, 'booth_total':booth_total, 
                'vector8': vector8, 'vector8_total':vector8_total, 'vector4':vector4, 'vector4_total':vector4_total}                 


for subdir in args.resume.split('/'):
    if 'bit' in subdir:
        savedir = subdir

np.save(os.path.join('cam_yue', savedir, 'input_sparsity_info.npy'), sparsity_dict)






 