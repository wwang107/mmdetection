from mmdet.apis import show_result
import mmcv
import visdom
import numpy as np
import cv2


def show_result_vsidom(img,
                       result,
                       class_names,
                       score_thr=0.0):
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
    img = show_result(
        img, result, class_names, score_thr=score_thr, show=False)
    return np.transpose(mmcv.bgr2rgb(img), (2, 0, 1))


def show_bbox(img, bbox):
    if type(img) is not np.ndarray:
        img = cv2.imread(img)

    bboxes = bbox[0]
    num_box = bboxes.shape[0]
    for i in range(0, num_box):
        l, t, r, b = bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]
        print(t, l, b, r)
        confidence = bboxes[i, 4]
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
        put_text(img, str(i), (l,t))

    return np.transpose(mmcv.bgr2rgb(img), (2, 0, 1))


def put_text(img, text, bl):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (0, 255, 0)
    lineType = 2
    cv2.putText(img, text, 
                bl,
                font,
                fontScale,
                fontColor,
                lineType)