import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import visdom

from models.vgg_m_2048 import VGG_M_2048
from utils.imagenet_video import ImageNetVideoDataset
from utils.logger import Logger
from utils.smoothness_loss import SmoothnessLoss

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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
parser.add_argument('--eval-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--vis', action='store_true')

best_loss = 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]


def main():
    global args, best_loss
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    model = VGG_M_2048(model_path='models/imagenet_vgg_m_2048.mat')
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    criterion = SmoothnessLoss().cuda()
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

    # Data loading codes
    normalize = transforms.Normalize(mean=INPUT_MEAN,
                                     std=INPUT_STD)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        ImageNetVideoDataset('../datasets/ILSVRC', 'train', transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ImageNetVideoDataset('../datasets/ILSVRC', 'val', transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    logger = Logger('tflogs', name='smooth')
    vis = visdom.Visdom(port=7236)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger, vis)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, criterion, epoch, logger)
            # remember best prec@1 and save checkpoint
            is_best = loss > best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def to_np(x):
    return x.data.cpu().numpy()


def train(train_loader, model, smoothness_criterion, optimizer, epoch, logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (target, pos_peer, neg_peer) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_var = torch.autograd.Variable(target, requires_grad=True)
        pos_peer_var = torch.autograd.Variable(pos_peer, requires_grad=True)
        neg_peer_var = torch.autograd.Variable(neg_peer, requires_grad=True)

        # compute output
        target_output = model.forward(target_var, ['relu5', 'prob', 'bbox_reg'])
        pos_output = model.forward(pos_peer_var, ['relu5'])['relu5']
        neg_output = model.forward(neg_peer_var, ['relu5'])['relu5']

        smoothness_loss = smoothness_criterion(target_output['relu5'], pos_output, neg_output)

        # measure metrics and record loss
        losses.update(smoothness_loss.data[0], target_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        smoothness_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        logger.scalar_summary('training/losses', smoothness_loss.data[0], epoch * len(train_loader) + i)
        for value in model.parameters():
            logger.histo_summary('weights', to_np(value), epoch * len(train_loader) + i)
            logger.histo_summary('gradients', to_np(value.grad), epoch * len(train_loader) + i)


def validate(val_loader, model, smoothness_criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (target, pos_peer, neg_peer) in enumerate(val_loader):
        target_var = torch.autograd.Variable(target, requires_grad=True)
        pos_peer_var = torch.autograd.Variable(pos_peer, requires_grad=True)
        neg_peer_var = torch.autograd.Variable(neg_peer, requires_grad=True)

        target_output = model.forward(target_var, ['relu5', 'prob', 'bbox_reg'])
        pos_output = model.forward(pos_peer_var, ['relu5'])['relu5']
        neg_output = model.forward(neg_peer_var, ['relu5'])['relu5']

        smoothness_loss = smoothness_criterion(target_output['relu5'], pos_output, neg_output)

        # measure metrics and record loss
        losses.update(smoothness_loss.data[0], target_var.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.scalar_summary('validation/losses', losses.avg, epoch)

    print(' * Losses {losses.avg:.3f}'
          .format(losses=losses))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
