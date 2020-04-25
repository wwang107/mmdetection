"""
this script generate a txt file containing a list of image,
each line is a path to the image with following formate:

subject_name/action_name/camera_name/frame_idx
"""

import os

retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'action_names': [
        'Directions-1', 'Directions-2',
        'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2',
        'Greeting-1', 'Greeting-2',
        'Phoning-1', 'Phoning-2',
        'Posing-1', 'Posing-2',
        'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2',
        'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2',
        'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2',
        'Walking-1', 'Walking-2',
        'WalkingDog-1', 'WalkingDog-2',
        'WalkingTogether-1', 'WalkingTogether-2']
}
ROOT_HUMAN36M_PATH = "/home/weiwang/Desktop/master-thesis/mmdetection/data/human36m/processed"
TRAIN_FILE_OUT = "/home/weiwang/Desktop/master-thesis/mmdetection/data/human36m/processed/image-list/train-images-list.txt"
TEST_FILE_OUT = "/home/weiwang/Desktop/master-thesis/mmdetection/data/human36m/processed/image-list/test-images-list.txt"

with open(TRAIN_FILE_OUT,"w") as train_file:
    for subject in retval["subject_names"]:
        for action in retval["action_names"]:
            for camera in retval["camera_names"]:
                try:
                    files = os.listdir(os.path.join(ROOT_HUMAN36M_PATH, subject, action, "imageSequence",camera))
                    for file in files:
                        if file.split(".")[-1] == "jpg":
                            train_file.write(f"{subject}/{action}/{camera}/{file}\n")
                except os.error as error:
                    print(error)




