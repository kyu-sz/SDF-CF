import argparse
import os
import random
import shutil
import time

import numpy as np
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from models.vgg_m_2048 import VGG_M_2048
from utils.eco_eval import eco_eval
from utils.imagenet import ImageNetDataset
from utils.imagenet_video import ImageNetVideoDataset
from utils.logger import Logger
from utils.smoothness_loss import SmoothnessLoss

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
parser.add_argument('--gpus', default="0,1,2,3", type=str,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # load data
    normalize = transforms.Normalize(mean=INPUT_MEAN,
                                     std=INPUT_STD)
    train_sampler = None
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    imagenet = ImageNetDataset(args.imagenet_dir, img_transform)
    train_loader = torch.utils.data.DataLoader(
            ImageNetVideoDataset(osp.join(args.imagenet_video_dir, 'ILSVRC'), 'train', img_transform) + imagenet,
            batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
            ImageNetVideoDataset(osp.join(args.imagenet_video_dir, 'ILSVRC'), 'val', img_transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # create model
    model = VGG_M_2048(num_classes=imagenet.num_classes, model_path='models/imagenet_vgg_m_2048.mat')
    print(model)
    model.to(torch.device('cuda'))  # Move to GPU.
    model = torch.nn.DataParallel(model)  # Enable multi-GPU training.

    smoothness_criterion = SmoothnessLoss().cuda()
    bbox_criterion = nn.MSELoss().cuda()
    cls_criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # initialize loggers.
    logger = Logger('tflogs', name='sdf-cf')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, optimizer, epoch,
              smoothness_criterion, bbox_criterion, cls_criterion,
              logger)
        save_checkpoint({
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'best_loss':  best_loss,
            'optimizer':  optimizer.state_dict(),
        }, model, False)

        # evaluate smoothness on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            # remember best prec@1 and save checkpoint
            loss = validate(val_loader, model, smoothness_criterion, bbox_criterion, epoch, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint({
                'epoch':      epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss':  best_loss,
                'optimizer':  optimizer.state_dict(),
            }, model, is_best)
            if is_best:
                eco_eval(model)
        else:
            save_checkpoint({
                'epoch':      epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss':  best_loss,
                'optimizer':  optimizer.state_dict(),
            }, model, False)


def to_np(x):
    return x.data.cpu().numpy()


def train(train_loader: torch.utils.data.DataLoader,
          model: nn.Module,
          optimizer: torch.optim.SGD,
          epoch: int,
          smoothness_criterion: nn.Module,
          bbox_criterion: nn.Module,
          cls_criterion: nn.Module,
          logger: Logger) -> None:
    batch_time = AverageMeter()
    data_time = AverageMeter()
    smoothness_losses = AverageMeter()
    pos_losses = AverageMeter()
    neg_losses = AverageMeter()
    bbox_losses = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (target, pos_sample, neg_sample, cid, bbox, pos_bbox) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Move the tensors to GPU.
        target = target.cuda(non_blocking=True)
        pos_sample = pos_sample.cuda(non_blocking=True)
        neg_sample = neg_sample.cuda(non_blocking=True)
        cid = cid.float().cuda(non_blocking=True)
        bbox = bbox.cuda(non_blocking=True)
        pos_bbox = pos_bbox.cuda(non_blocking=True)

        # compute output
        target_output = model.forward(target, ['relu5', 'bbox_reg', 'fc8ext'])
        pos_output = model.forward(pos_sample, ['relu5'])['relu5']
        neg_output = model.forward(neg_sample, ['relu5'])['relu5']

        # Compute losses.
        pos_loss, neg_loss = smoothness_criterion(target_output['relu5'], pos_output, neg_output, bbox, pos_bbox)
        smoothness_loss = pos_loss + neg_loss
        bbox_loss = bbox_criterion(target_output['bbox_reg'], bbox.view(target_output['bbox_reg'].shape))
        cls_loss = cls_criterion(target_output['fc8ext'], cid.view(target_output['fc8ext'].shape))
        total_loss = smoothness_loss + bbox_loss + cls_loss

        # measure metrics and record loss
        smoothness_losses.update(smoothness_loss.item(), target.size(0))
        pos_losses.update(pos_loss.item(), target.size(0))
        neg_losses.update(neg_loss.item(), target.size(0))
        bbox_losses.update(bbox_loss.item(), target.size(0))

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

            logger.image_summary('training/smoothness_samples', sample_list, epoch * len(train_loader) + i)

        logger.scalar_summary('training/smoothness_losses', smoothness_loss.item(), epoch * len(train_loader) + i)
        logger.scalar_summary('training/pos_losses', pos_loss.item(), epoch * len(train_loader) + i)
        logger.scalar_summary('training/neg_losses', neg_loss.item(), epoch * len(train_loader) + i)
        logger.scalar_summary('training/cls_losses', cls_loss.item(), epoch * len(train_loader) + i)
        logger.scalar_summary('training/bbox_losses', bbox_loss.item(), epoch * len(train_loader) + i)


def validate(val_loader, model, smoothness_criterion, bbox_criterion, epoch, logger):
    batch_time = AverageMeter()
    pos_losses = AverageMeter()
    neg_losses = AverageMeter()
    bbox_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (target, pos_sample, neg_sample, _, bbox, pos_bbox) in enumerate(val_loader):
        # Move the tensors to GPU.
        target = target.cuda(non_blocking=True)
        pos_sample = pos_sample.cuda(non_blocking=True)
        neg_sample = neg_sample.cuda(non_blocking=True)
        bbox = bbox.cuda(non_blocking=True)
        pos_bbox = pos_bbox.cuda(non_blocking=True)

        target_output = model.forward(target, ['relu5', 'bbox_reg'])
        pos_output = model.forward(pos_sample, ['relu5'])['relu5']
        neg_output = model.forward(neg_sample, ['relu5'])['relu5']

        pos_loss, neg_loss = smoothness_criterion(target_output['relu5'], pos_output, neg_output, bbox, pos_bbox)
        bbox_loss = bbox_criterion(target_output['bbox_reg'], bbox.view(target_output['bbox_reg'].shape))

        # measure metrics and record loss
        pos_losses.update(pos_loss.item(), target.size(0))
        neg_losses.update(neg_loss.item(), target.size(0))
        bbox_losses.update(bbox_loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.scalar_summary('validation/smoothness_pos_losses', pos_losses.avg, epoch)
    logger.scalar_summary('validation/smoothness_neg_losses', neg_losses.avg, epoch)
    logger.scalar_summary('validation/cls_losses', bbox_losses.avg, epoch)
    logger.scalar_summary('validation/bbox_losses', bbox_losses.avg, epoch)

    print(' * Smoothness losses {smoothness_losses.avg:.3f}'
          .format(smoothness_losses=(pos_losses.avg + neg_losses.avg)))

    return pos_losses.avg + neg_losses.avg + bbox_losses.avg


def save_checkpoint(state, model, is_best, state_path='checkpoint.pth.tar'):
    model.module.save('checkpoint.mat')
    torch.save(state, state_path)
    if is_best:
        shutil.copyfile(state_path, 'model_best.pth.tar')
        model.module.save('model_best.mat')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
