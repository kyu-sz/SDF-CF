import argparse
import os
import random
import shutil
import time
import os.path as osp

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import visdom

from models.vgg_m_2048 import VGG_M_2048
from utils.imagenet_video import ImageNetVideoDataset
from utils.imagenet import ImageNetDataset
from utils.logger import Logger
from utils.masked_smoothness_loss import MaskedSmoothnessLoss
from utils.eco_eval import eco_eval

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--imagenet-dir', type=str, metavar='N',
                    help='directory of the ImageNet dataset')
parser.add_argument('--imagenet-video-dir', type=str, metavar='N',
                    help='directory of the ImageNet Video dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--vis-freq', '-v', default=10, type=int,
                    metavar='N', help='visualization frequency (default: 10)')
parser.add_argument('--eval-freq', default=10, type=int,
                    metavar='N', help='evaluation frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--gpu', default="0,1,2,3", type=str,
                    help='GPUs for training')
parser.add_argument('--vis', action='store_true')

best_loss = 0

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

INPUT_MEAN = [123.6591, 116.7663, 103.9318]
INPUT_STD = [1, 1, 1]


def main():
    global args, best_loss
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load data
    normalize = transforms.Normalize(mean=INPUT_MEAN,
                                     std=INPUT_STD)
    train_sampler = None
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    smoothness_train_loader = torch.utils.data.DataLoader(
        ImageNetVideoDataset(osp.join(args.imagenet_video_dir, 'ILSVRC'), 'train', img_transform),
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    smoothness_val_loader = torch.utils.data.DataLoader(
        ImageNetVideoDataset(osp.join(args.imagenet_video_dir, 'ILSVRC'), 'val', img_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    cls_train_loader = torch.utils.data.DataLoader(
        ImageNetDataset(args.imagenet_dir, img_transform),
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # create model
    model = VGG_M_2048(model_path='models/imagenet_vgg_m_2048.mat')
    print(model)

    model = torch.nn.DataParallel(model)
    model.cuda()

    smoothness_criterion = MaskedSmoothnessLoss().cuda()
    bbox_criterion = nn.MSELoss().cuda()
    cls_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # initialize loggers.
    logger = Logger('tflogs', name='sdf-cf')
    vis = visdom.Visdom(port=7236)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train smoothness and classification iteratively in each epoch.
        train_classfication(cls_train_loader, model, cls_criterion, bbox_criterion, optimizer, epoch,
                            logger, vis)
        train_smoothness(smoothness_train_loader, model, smoothness_criterion, bbox_criterion, optimizer, epoch,
                         logger, vis)

        # evaluate smoothness on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(smoothness_val_loader, model, smoothness_criterion, bbox_criterion, epoch, logger)
            # remember best prec@1 and save checkpoint
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best)

            if is_best:
                eco_eval(model)


def to_np(x):
    return x.data.cpu().numpy()


def train_classfication(train_loader, model, cls_criterion, bbox_criterion, optimizer, epoch, logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    bbox_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (image, cid, bbox) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image_var = torch.autograd.Variable(image, requires_grad=True)
        cid_var = torch.autograd.Variable(cid, requires_grad=False).cuda(async=True)
        bbox_var = torch.autograd.Variable(bbox, requires_grad=False).cuda(async=True)

        # compute output
        output = model.forward(image_var, ['fc8ext', 'bbox_reg'])

        # Compute losses.
        cls_loss = cls_criterion(output['fc8ext'], cid_var)
        bbox_loss = bbox_criterion(output['bbox_reg'], bbox_var)
        total_loss = cls_loss + bbox_loss

        # measure metrics and record loss
        cls_losses.update(cls_loss.data[0], image_var.size(0))
        bbox_losses.update(bbox_loss.data[0], image_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Classification loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Bounding box regression loss {bbox_loss.val:.4f} ({bbox_loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, cls_loss=cls_losses, bbox_loss=bbox_losses))

        logger.scalar_summary('training/classification_losses', cls_loss.data[0], epoch * len(train_loader) + i)
        logger.scalar_summary('training/bbox_losses', bbox_loss.data[0], epoch * len(train_loader) + i)


def train_smoothness(train_loader, model, smoothness_criterion, bbox_criterion, optimizer, epoch, logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    smoothness_losses = AverageMeter()
    pos_losses = AverageMeter()
    neg_losses = AverageMeter()
    bbox_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (target, pos_sample, neg_sample, bbox, pos_bbox) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_var = torch.autograd.Variable(target, requires_grad=True).cuda(async=True)
        pos_var = torch.autograd.Variable(pos_sample, requires_grad=True).cuda(async=True)
        neg_var = torch.autograd.Variable(neg_sample, requires_grad=True).cuda(async=True)
        bbox_var = torch.autograd.Variable(bbox, requires_grad=False).cuda(async=True)

        # compute output
        target_output = model.forward(target_var, ['relu5', 'bbox_reg'])
        pos_output = model.forward(pos_var, ['relu5'])['relu5']
        neg_output = model.forward(neg_var, ['relu5'])['relu5']

        # Compute losses.
        pos_loss, neg_loss = smoothness_criterion(target_output['relu5'], pos_output, neg_output, bbox, pos_bbox)
        smoothness_loss = pos_loss + neg_loss
        bbox_loss = bbox_criterion(target_output['bbox_reg'], bbox_var)
        total_loss = smoothness_loss + bbox_loss

        # measure metrics and record loss
        smoothness_losses.update(smoothness_loss.data[0], target_var.size(0))
        pos_losses.update(pos_loss.data[0], target_var.size(0))
        neg_losses.update(neg_loss.data[0], target_var.size(0))
        bbox_losses.update(bbox_loss.data[0], target_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Smoothness epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Positive loss {pos_loss.val:.4f} ({pos_loss.avg:.4f})\t'
                  'Negative loss {neg_loss.val:.4f} ({neg_loss.avg:.4f})\t'
                  'Overall loss {smoothness_loss.val:.4f} ({smoothness_loss.avg:.4f})\t'
                  'Bounding box regression loss {bbox_loss.val:.4f} ({bbox_loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                pos_loss=pos_losses, neg_loss=neg_losses, smoothness_loss=smoothness_losses, bbox_loss=bbox_losses))

        if i % args.vis_freq == 0:
            concat_imgs = torch.zeros((target.shape[0], target.shape[1], target.shape[2], target.shape[3] * 3 + 4))

            for c in range(target.shape[1]):
                concat_imgs[:, c, :, :target.shape[3]] = target[:, c, ...] * INPUT_STD[c] + INPUT_MEAN[c]
                concat_imgs[:, c, :, target.shape[3] + 2: target.shape[3] * 2 + 2] = \
                    pos_sample[:, c, ...] * INPUT_STD[c] + INPUT_MEAN[c]
                concat_imgs[:, c, :, target.shape[3] * 2 + 4:] = neg_sample[:, c, ...] * INPUT_STD[c] + INPUT_MEAN[c]

            sample_list = [concat_imgs[b, ...] for b in range(concat_imgs.shape[0])]

            logger.image_summary('training/samples', sample_list, epoch * len(train_loader) + i)

        logger.scalar_summary('training/smoothness_losses', smoothness_loss.data[0], epoch * len(train_loader) + i)
        logger.scalar_summary('training/pos_losses', pos_loss.data[0], epoch * len(train_loader) + i)
        logger.scalar_summary('training/neg_losses', neg_loss.data[0], epoch * len(train_loader) + i)
        logger.scalar_summary('training/bbox_losses', bbox_loss.data[0], epoch * len(train_loader) + i)


def validate(val_loader, model, smoothness_criterion, bbox_criterion, epoch, logger):
    batch_time = AverageMeter()
    pos_losses = AverageMeter()
    neg_losses = AverageMeter()
    bbox_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (target, pos_sample, neg_sample, bbox, pos_bbox) in enumerate(val_loader):
        target_var = torch.autograd.Variable(target, requires_grad=True).cuda(async=True)
        pos_var = torch.autograd.Variable(pos_sample, requires_grad=True).cuda(async=True)
        neg_var = torch.autograd.Variable(neg_sample, requires_grad=True).cuda(async=True)
        bbox_var = torch.autograd.Variable(bbox, requires_grad=False).cuda(async=True)

        target_output = model.forward(target_var, ['relu5', 'bbox_reg'])
        pos_output = model.forward(pos_var, ['relu5'])['relu5']
        neg_output = model.forward(neg_var, ['relu5'])['relu5']

        pos_loss, neg_loss = smoothness_criterion(target_output['relu5'], pos_output, neg_output, bbox, pos_bbox)
        bbox_loss = bbox_criterion(target_output['bbox_reg'], bbox_var)

        # measure metrics and record loss
        pos_losses.update(pos_loss.data[0], target_var.size(0))
        neg_losses.update(neg_loss.data[0], target_var.size(0))
        bbox_losses.update(bbox_loss.data[0], target_var.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.scalar_summary('validation/smoothness_pos_losses', pos_losses.avg, epoch * len(val_loader) + i)
    logger.scalar_summary('validation/smoothness_neg_losses', neg_losses.avg, epoch * len(val_loader) + i)
    logger.scalar_summary('validation/bbox_losses', bbox_losses.avg, epoch * len(val_loader) + i)

    print(' * Smoothness losses {smoothness_losses.avg:.3f}'
          .format(smoothness_losses=(pos_losses.avg + neg_losses.avg)))

    return pos_losses.avg + neg_losses.avg + bbox_losses.avg


def save_checkpoint(state, is_best, state_fn='checkpoint.pth.tar', model_fn='model_best.pth.tar'):
    torch.save(state, state_fn)
    if is_best:
        shutil.copyfile(state_fn, model_fn)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
