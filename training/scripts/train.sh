#!/usr/bin/env bash
BASEDIR=$(dirname "$0")
cd $BASEDIR/..
python train.py --imagenet-dir ../../datasets/ImageNet --imagenet-video-dir ../../datasets/ImageNetVideo -b 64 --lr 0.001 \
    --epochs 1000 -j 8 --gpus 2,3
