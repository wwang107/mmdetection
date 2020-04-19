#!/usr/bin/env bash
sudo docker run --gpus all\
  --privileged\
  -it\
  -v "/media/weiwang/Elements/human36m":/mmdetection/data/human36m\
  -v "/home/weiwang/Desktop/master-thesis/mmdetection/checkpoints/":/mmdetection/checkpoints\
  -p 8888:8888\
  weiwang/bbextractor\
  /bin/bash