import numpy as np
import torch
import torch.nn as nn
import sys
import os
import argparse

from data import *
from models.efficientnet import EfficientNet
from models.mobilenet_v1 import mobilenet_v1
from models.mobilenet_v2 import mobilenet_v2
from models.resnet import resnet18, resnet50, resnet152
from models.resnet164 import resnet_164
from models.vgg import vgg11, vgg19
from utils import unstructure_prune
from utils import vector_prune
from utils import filter_prune

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10 training with gating')
parser.add_argument('--arch', default='efficientnet_b0', help='select a model')
parser.add_argument('--dataset', '-d', type=str, default='imagenet',
                    choices=['cifar10', 'cifar100','imagenet'],
                    help='dataset choice')
parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                    help='path to dataset')
parser.add_argument('--resume', default='', type=str,
                    help='path to  latest checkpoint (default: None)')
parser.add_argument('--prune_type', default='unstructure', type=str,
                    help='prune method')
parser.add_argument('--prune_ratio', default=0, type=float,
                    help='unstructure pruning ratio (how many weights are left)')
parser.add_argument('--quantize', default=False, action='store_true',
                    help='quantize weight and activation')
parser.add_argument('--pretrained', default=False, action='store_true',
                    help='load official pretrained ckpt')
parser.add_argument('--save_folder', default='./',
                    type=str,
                    help='folder to save the extracted info')
args = parser.parse_args()


if 'cifar10' in args.dataset:
    num_classes = 10
elif 'cifar100' in args.dataset:
    num_classes = 100
elif 'imagenet' in args.dataset:
    num_classes = 1000

if args.arch == 'efficientnet_b0':
    model = EfficientNet.from_name("efficientnet-b0", quantize=args.quantize, override_params={'num_classes': num_classes})
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'mobilenet_v1':
    model = mobilenet_v1(quantize=args.pretrained, num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'mobilenet_v2':
    model = mobilenet_v2(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'resnet18':
    model = resnet18(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'resnet50':
    model = resnet50(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'resnet152':
    model = resnet152(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'resnet164':
    model = resnet_164(num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'vgg11':
    model = vgg11(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

elif args.arch == 'vgg19':
    model = vgg19(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
    model = torch.nn.DataParallel(model).cuda()

else:
    print('No such model.')
    sys.exit()


if args.resume:
    if os.path.isfile(args.resume):
        print('=> loading checkpoint `{}`'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print('=> loaded checkpoint `{}` (epoch: {})'.format(
            args.resume, checkpoint['epoch']
        ))
    else:
        print('=> no checkpoint found at `{}`'.format(args.resume))

if args.prune_type == 'unstructure':
    prune = unstructure_prune
elif args.prune_type == 'vector':
    prune = vector_prune
elif args.prune_type == 'filter':
    prune = filter_prune
else:
    print("Wrong Prune Type")
    sys.exit()

if args.prune_ratio > 0:
    masks = prune.get_masks(model, args.prune_ratio)
    prune.set_masks(model, masks)

train_loader = prepare_train_data(dataset=args.dataset,
                                  datadir=args.datadir+'/train' if 'imagenet' in args.dataset else args.datadir,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=1)

modules = list(model.modules())

input_conv_list = []
def forward_hook_conv(module, input, output):
    if args.quantize:
        input_max = input[0].max()
        input_min = input[0].min()
        input_quant = ((input[0]-input_min)/(input_max-input_min)*(2**8-1)).clamp_(0, 2**8-1).round_().cpu().data.numpy()
        input_conv_list.append(input_quant)
    else:
        input_conv_list.append(input[0].cpu().data.numpy())

input_linear_list = []
def forward_hook_linear(module, input, output):
    if args.quantize:
        input_max = input[0].max()
        input_min = input[0].min()
        input_quant = ((input[0]-input_min)/(input_max-input_min)*(2**8-1)).clamp_(0, 2**8-1).round_().cpu().data.numpy()
        input_linear_list.append(input_quant)
    else:
        input_linear_list.append(input[0].cpu().data.numpy())

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
   
model.eval()
input_data, target = next(iter(train_loader))
_ = model(input_data)

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

dict_conv = {'HW':HW, 'Padding':Padding, 'C':C, 'B':B, 'RS':RS, 'M':M, 'U':U, 'group':group}
np.save(os.path.join(args.save_folder, args.arch)+'_'+str(args.prune_ratio)+'_conv_info.npy', dict_conv)

dict_input_weight_conv = {'input': input_conv_list, 'weight': weight_conv_list}
np.save(os.path.join(args.save_folder, args.arch)+'_'+str(args.prune_ratio)+'_input_weight_conv.npy', dict_input_weight_conv)

dict_input_weight_linear = {'input': input_linear_list, 'weight': weight_linear_list}
np.save(os.path.join(args.save_folder, args.arch)+'_'+str(args.prune_ratio)+'_input_weight_linear.npy', dict_input_weight_linear)



 