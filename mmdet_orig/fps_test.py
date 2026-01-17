from mmdet.apis import init_detector, inference_detector #, show_result
import torch
import mmcv
import time

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# # for centernet
# config_file     = '/content/drive/MyDrive/IEEE_Access/code/mmdet/configs/_base_/default_centernet.py'
# checkpoint_file = '/content/drive/MyDrive/IEEE_Access/code/weights/centernet_epoch_20.pth'

# for faster rcnn
config_file     = '/content/drive/MyDrive/IEEE_Access/code/mmdet/configs/_base_/default_runtime_fasterrcnn.py'
checkpoint_file = '/content/drive/MyDrive/IEEE_Access/code/weights/fasterrcnn_epoch_20.pth' # D:\Github\mmdetection\work_dirs_fasterRCNN\default_runtime

# build the model from a config file and a checkpoint file
t0 = time_synchronized()
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img    = r'/content/data_mmdet/one_image/val/images/4_000720.jpg'  # or img = mmcv.imread(img), which will only load it once

t1 = time_synchronized()
result = inference_detector(model, img)
t2 = time_synchronized()
# print('result =' ,result)
t3 = time_synchronized()
print('inference time =', t2-t1)
print(t3-t0)
# # visualize the results in a new window
# show_result(img, result, model.CLASSES)
# # or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     show_result(frame, result, model.CLASSES, wait_time=1)