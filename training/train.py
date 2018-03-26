import argparse
import os
import shutil
import time
import sys
import random
import matplotlib.pyplot as plt

import sklearn
import sklearn.metrics

import cv2
import numpy as np
import visdom

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils.logger import Logger

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

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

best_prec1 = 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    else:
        model = None
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO: define loss function (criterion) and optimizer
    criterion = nn.SoftMarginLoss(size_average=False).cuda()
    # criterion = nn.SoftMarginLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading codes
    # TODO: Write codes for IMDBDataset in custom.py
    trainval_imdb = get_imdb('voc_2007_trainval')
    test_imdb = get_imdb('voc_2007_test')

    normalize = transforms.Normalize(mean=INPUT_MEAN,
                                     std=INPUT_STD)
    train_dataset = IMDBDataset(
        trainval_imdb,
        transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for visdom and tboard
    # TODO: You can pass the logger objects to train(), make appropriate modifications to train()
    logger = Logger(args.arch + '_logs', name='freeloc')
    vis = visdom.Visdom(port=3627)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger, vis)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch, logger)
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    # At the end of training, use Visdom to plot 20 randomly chosen images and corresponding heatmaps (similar to above)
    # from the validation set.
    final_loader = torch.utils.data.DataLoader(
        IMDBDataset(test_imdb, transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=20, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    model.eval()
    input, target = final_loader.__iter__().__next__()
    input_var = torch.autograd.Variable(input, volatile=True)
    response_maps = model.forward(input_var)
    heatmaps = F.sigmoid(response_maps)
    for b in xrange(heatmaps.size()[0]):
        img = input[b]
        for c in xrange(img.shape[0]):
            img[c] = img[c] * INPUT_STD[c] + INPUT_MEAN[c]
        vis.image(input[b],
                  opts={'title': 'final_' + str(b) + '_image'})
        for c in np.where(target[b] == 1)[0]:
            heatmap = cv2.resize(np.array(heatmaps.data[b, c, :, :]), input.size()[2:])
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(heatmap)
            rgb_map = np.transpose(np.delete(rgba_img, 3, 2), (2, 0, 1))

            classname = train_loader.dataset.classes[c]
            vis.image(rgb_map,
                      opts={'title': 'final_' + str(b) + '_heatmap_' + classname})


def to_np(x):
    return x.data.cpu().numpy()


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, logger, vis):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, requires_grad=True)
        target_var = torch.autograd.Variable(target)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        # compute output
        response_maps = model.forward(input_var)
        imoutput = F.max_pool2d(response_maps, response_maps.size()[2:])
        heatmaps = F.sigmoid(response_maps)
        loss = criterion(imoutput, target_var) / input.size()[0]
        # loss = criterion(imoutput, target_var)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # TODO: compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, avg_m1=avg_m1,
                avg_m2=avg_m2))

        # TODO: Visualize things as mentioned in handout
        # TODO: Visualize at appropriate intervals
        logger.scalar_summary('training/metric1', m1[0], epoch * len(train_loader) + i)
        logger.scalar_summary('training/metric2', m2[0], epoch * len(train_loader) + i)
        logger.scalar_summary('training/losses', loss.data[0], epoch * len(train_loader) + i)
        for value in model.parameters():
            logger.histo_summary('weights', to_np(value), epoch * len(train_loader) + i)
            logger.histo_summary('gradients', to_np(value.grad), epoch * len(train_loader) + i)
        if i % int(len(train_loader) / 4) == 0:
            image_list = []
            heatmap_list = []
            for b in xrange(heatmaps.size()[0]):
                img = input[b]
                for c in xrange(img.shape[0]):
                    img[c] = img[c] * INPUT_STD[c] + INPUT_MEAN[c]
                vis.image(img,
                          opts={'title': str(epoch) + '_' + str(i) + '_' + str(b) + '_image'})
                image_list.append(img)
                for c in np.where(target[b] == 1)[0]:
                    heatmap = cv2.resize(np.array(heatmaps.data[b, c, :, :]), input.size()[2:])
                    cmap = plt.get_cmap('jet')
                    rgba_img = cmap(heatmap)
                    rgb_map = np.transpose(np.delete(rgba_img, 3, 2), (2, 0, 1))

                    heatmap_list.append(rgb_map)
                    classname = train_loader.dataset.classes[c]
                    vis.image(rgb_map,
                              opts={'title': str(epoch) + '_' + str(i) + '_' + str(b) + '_heatmap_' + classname})
            logger.image_summary('training/images', image_list, epoch * len(train_loader) + i)
            logger.image_summary('training/heatmaps', heatmap_list, epoch * len(train_loader) + i)


def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.type(torch.FloatTensor).cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion`` compute output
        resp_maps = model.forward(input_var)
        imoutput = nn.functional.max_pool2d(resp_maps, resp_maps.size()[2:])
        loss = criterion(imoutput, target_var)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.data[0], input.size(0))
        avg_m1.update(m1[0], input.size(0))
        avg_m2.update(m2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                avg_m1=avg_m1, avg_m2=avg_m2))

        # TODO: Visualize things as mentioned in handout
        # TODO: Visualize at appropriate intervals
        logger.scalar_summary('validation/metric1', m1[0], epoch)
        logger.scalar_summary('validation/metric2', m2[0], epoch)

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'
          .format(avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
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


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    # Mean accuracy.
    return [float(sum(sum((output.view(target.size()) > 0) == (target > 0)))) / target.nelement()]


def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed
    return [float(sum(sum((output.view(target.size()) > 0) + (target > 0) == 2)))
            / sum(sum(target > 0)) if sum(sum(output.view(target.size()) > 0)) > 0 else 0]


if __name__ == '__main__':
    main()
