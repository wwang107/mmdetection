from __future__ import division
from mmdet.apis import init_detector, inference_detector
from mycode.visualization import show_bbox
import mmcv
import numpy as np
import os
import json
import torch
import visdom

IMAGE_LIST = os.path.join(
    os.getcwd(), "data/human36m/processed/image-list/train-images-list.txt")
HUMAN36M = os.path.join(os.getcwd(), "data/human36m/processed")
OUT_FILEE = os.path.join(
    os.getcwd(), "data/human36m/extra/myBBox/fasterRCNN.json")


def filter_dectection(detection) -> list:
    detection = detection[0]
    if detection.shape[0] > 1:
        detection = detection[detection[:, 4].argsort()]
        if detection[-1, 4] > 0.9:
            return [detection[-1, :][np.newaxis, :]]
        else:
            return [detection[-1:-2, :]]
    else:
        return [detection]


train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
test_subjects = ['S9', 'S11']
print("Is CUDA available: ", torch.cuda.is_available())
# build the model from a config file and a checkpoint file
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

vis = visdom.Visdom(port=8888)
wind = None
with open(IMAGE_LIST, "r") as imageList:
    total = sum(1 for line in imageList)

count = 0
boxes = []
with open(OUT_FILEE, "w") as outfile:
    with open(IMAGE_LIST, "r") as imageList:
        for item in imageList:
            count += 1
            if count % 100 == 1:
                print(f"[{count}/{total}]\n {item}")

            subject, action, cameraId, frameId = item.rstrip("\n").split("/")
            img = os.path.join(
                HUMAN36M, f"{subject}/{action}/imageSequence/{cameraId}/{frameId}")
                # HUMAN36M, f"S1/Eating-1/imageSequence/58860488/img_001462.jpg")
            bbox = filter_dectection(inference_detector(model, img))
            num_box = bbox[0].shape[0]
            idx = 0
            if num_box > 1:
            # if there are more than 2 detections, we should list out and choose
                wind = vis.image(
                    show_bbox(img, bbox),
                    win=wind)
                idx = input("Choose a box to keep: ")
                bbox = [bbox[0][int(idx),:]]
                l, t, r, b = bbox[0][idx, 0].item(), bbox[0][idx, 1].item(),bbox[0][idx, 2].item(), bbox[0][idx, 3].item()
                json.dump([l,t,r,b], outfile)
            elif num_box == 0:
                json.dump([], outfile)
            else:
                l, t, r, b = bbox[0][0, 0].item(), bbox[0][0, 1].item(), bbox[0][0, 2].item(), bbox[0][0, 3].item()
                json.dump([l, t, r, b], outfile)
            outfile.write("\n")
