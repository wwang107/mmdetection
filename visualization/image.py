import visdom
import cv2
import numpy as np
import mmcv
from mmdet.apis import show_result

def show_result_vsidom(img,
                       result,
                       class_names,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       port=8097):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    vis = visdom.Visdom(port=port)
    vis.text("detection")

    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    vis.image(np.transpose(mmcv.bgr2rgb(img), (2, 0, 1)))

