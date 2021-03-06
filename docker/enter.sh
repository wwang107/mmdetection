#!/usr/bin/env bash
sudo docker run --gpus all\
  --privileged\
  -it\
  -v "/media/weiwang/Elements/human36m":/mmdetection/data/human36m\
  -v "/home/weiwang/Desktop/master-thesis/mmdetection/checkpoints/":/mmdetection/checkpoints\
  -v "/home/weiwang/Desktop/master-thesis/mmdetection/extractbbox.py":/mmdetection/extractbbox.py\
  -v "/home/weiwang/Desktop/master-thesis/mmdetection/continue.py":/mmdetection/continue.py\
  -v "/home/weiwang/Desktop/master-thesis/mmdetection/mycode":/mmdetection/mycode\
  -p 8888:8888\
  weiwang/bbextractor\
  /bin/bash