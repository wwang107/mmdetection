#!/usr/bin/env bash
sudo docker run --gpus all\
  --privileged\
  -it\
  -v "/media/weiwang/Elements/human36m":/mmdetection/data/human36m\
  -v "/home/weiwang/Desktop/master-thesis/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth":/mmdetection/checkpoints\
  -p 8888:8888\
  weiwang/bbextractor\
  /bin/bash