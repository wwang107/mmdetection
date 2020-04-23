from mmdet.apis import init_detector, inference_detector
from visualization import show_result_vsidom

config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# or img = mmcv.imread(img), which will only load it once
img = './data/human36m/processed/S1/Directions-1/imageSequence/54138969/img_000001.jpg'
result = inference_detector(model, img)
# visualize the results in a new window
show_result_vsidom(img, result, model.CLASSES,port=8888)
# show_result(img, result, model.CLASSES)
# or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')
