""" This file is for training original model without routing modules.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import os
import sys
import shutil
import argparse
import time
import logging
from tqdm import tqdm

from models.efficientnet import EfficientNet
from models.mobilenet_v1 import mobilenet_v1
from models.mobilenet_v2 import mobilenet_v2
from models.resnet import resnet18, resnet50, resnet152
from models.resnet164 import resnet_164
from models.vgg import vgg11, vgg19
from utils import unstructure_prune
from utils import vector_prune
from utils import filter_prune
from utils.data import *


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', default='efficientnet_b0', help='select a model')
    parser.add_argument('--dataset', '-d', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100','imagenet','writing'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=90, type=int,
                        help='number of epochs (default: 90)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr_schedule_cifar', default=False, action="store_true",
                        help='use cifar learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_decay_freq', default=30, type=float,
                        help='num epochs to decay lr once')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--proceed', default=False, action="store_true",
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='use official pretrained model')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save_folder', default='logs',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--prune_type', default='unstructure', type=str,
                        help='prune method')
    parser.add_argument('--prune_ratio', default=0, type=float,
                        help='unstructure pruning ratio (how many weights are left)')
    parser.add_argument('--prune_delay', default=0, type=float,
                        help='number of epochs starting to prune')
    parser.add_argument('--quantize', default=False, action='store_true',
                        help='quantize weight and activation')
    parser.add_argument('--num_bits', default=8, type=int,
                        help='bits for activation')
    parser.add_argument('--num_bits_weight',default=8, type=int,
                        help='bits for weight')   
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):

    num_classes = 2

    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained, num_classes=num_classes, quantize=args.quantize)
        model = torch.nn.DataParallel(model).cuda()
    else:
        logging.info('No such model.')
        sys.exit()


    best_prec1 = 0
    best_epoch = 0

    if args.resume and not args.pretrained:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.proceed:
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']

            res = model.load_state_dict(checkpoint['state_dict'], strict=False)
            for missing_key in res.missing_keys:
                assert 'quantize' in missing_key

            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir+'/train' if 'imagenet' in args.dataset else args.datadir,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir+'/val' if 'imagenet' in args.dataset else args.datadir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    print_sparsity = True

    global adjust_learning_rate
    if args.lr_schedule_cifar:
        adjust_learning_rate = adjust_learning_rate_cifar

    if args.prune_type == 'unstructure':
        prune = unstructure_prune
    elif args.prune_type == 'vector':
        prune = vector_prune
    elif args.prune_type == 'filter':
        prune = filter_prune
    else:
        logging.info("Wrong Prune Type")
        sys.exit()

    for _epoch in tqdm(range(args.start_epoch, args.epoch)):
        lr = adjust_learning_rate(args, optimizer, _epoch)
        print('Learning Rate:', lr)

        if args.prune_ratio > 0 and args.prune_delay == _epoch:
            masks = prune.get_masks(model, args.prune_ratio)
        print(train_loader.dataset.files)
        for i, (input, target) in enumerate(train_loader):
            # measuring data loading time       
            data_time.update(time.time() - end)
            model.train()

            target = target.squeeze().long().cuda()
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            if args.prune_ratio > 0 and args.prune_delay <= _epoch :
                prune.set_masks(model, masks)
                if print_sparsity:
                    prune.count_sparsity(model)
                    print_sparsity = False

            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if i % args.print_freq == 0:
                logging.info("Iter: [{0}][{1}/{2}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                _epoch,
                                i,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                top1=top1)
                )

        with torch.no_grad():
            if args.prune_ratio > 0 and args.prune_delay <= _epoch :
                prune.set_masks(model, masks)
            prec1 = validate(args, test_loader, model, criterion, _epoch)

        is_best = prec1 > best_prec1
        if is_best:
            best_epoch = _epoch + 1
            best_prec1 = prec1

        print("Current Best Prec@1: ", best_prec1, "Best Epoch: ", best_epoch)
        
        checkpoint_path = os.path.join(args.save_path, 'checkpoint_{:05d}_{:.2f}.pth.tar'.format(_epoch, prec1))
        save_checkpoint({
            'epoch': _epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        },
            is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                      'checkpoint_latest'
                                                      '.pth.tar'))



def validate(args, test_loader, model, criterion, _epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    logging.info('Epoch {} * Prec@1 {top1.avg:.3f}'.format(_epoch, top1=top1))
    return top1.avg



def test_model(args):
    # create model
    num_classes = 2
    if args.arch == 'efficientnet_b0':
        if args.pretrained:
            model = EfficientNet.from_pretrained("efficientnet-b0", quantize=args.quantize, num_classes=num_classes)
        else:
            model = EfficientNet.from_name("efficientnet-b0", quantize=args.quantize, override_params={'num_classes': num_classes})
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'mobilenet_v1':
        model = mobilenet_v1(quantize=args.quantize, num_classes=num_classes)
        model = torch.nn.DataParallel(model).cuda()

        if args.pretrained:
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']

            if num_classes != 1000:
                new_dict = {k:v for k,v in state_dict.items() if 'fc' not in k}
                state_dict = new_dict

            res = model.load_state_dict(state_dict, strict=False)

            for missing_key in res.missing_keys:
                assert 'quantize' in missing_key or 'fc' in missing_key

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
        logging.info('No such model.')
        sys.exit()

    if args.resume and not args.pretrained:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()
    
    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, 0)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, _epoch):
    lr = args.lr * (args.step_ratio ** (_epoch // args.lr_decay_freq))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_cifar(args, optimizer, _epoch):
    if _epoch <= 80:
        lr = args.lr 
    elif _epoch <= 120:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    print(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
