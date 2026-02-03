# from mmdet.apis import init_detector, inference_detector #, show_result
# import torch
# import mmcv
# import time

# def time_synchronized():
#     # pytorch-accurate time
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     return time.time()

# # # for centernet
# # config_file     = '/content/drive/MyDrive/IEEE_Access/code/mmdet/configs/_base_/default_centernet.py'
# # checkpoint_file = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/centernet_r18_TrainVal_5_times/seed0/epoch_50.pth'

# # for faster rcnn
# config_file     = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/_config_2026/UDP_faster_rcnn_TranVal_test.py'
# checkpoint_file = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r18_TrainVal_17e_5_times/seed0/epoch_17.pth'

# # build the model from a config file and a checkpoint file
# t0 = time_synchronized()
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img    = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/4_025950.jpg'

# t1 = time_synchronized()
# result = inference_detector(model, img)
# t2 = time_synchronized()
# # print('result =' ,result)
# t3 = time_synchronized()
# print('inference time =', t2-t1)
# print(t3-t0)
# # # visualize the results in a new window
# # show_result(img, result, model.CLASSES)
# # # or save the visualization results to image files
# # show_result(img, result, model.CLASSES, out_file='result.jpg')

# # test a video and show the results
# # video = mmcv.VideoReader('video.mp4')
# # for frame in video:
# #     result = inference_detector(model, frame)
# #     show_result(frame, result, model.CLASSES, wait_time=1)





from mmdet.apis import init_detector, inference_detector
import torch
import mmcv
import time
import numpy as np

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# # # for centernet
config_file     = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/_config_2026/UDP_centernet_TrainVal_test.py'
checkpoint_file = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/centernet_r18_TrainVal_5_times/seed0/epoch_50.pth'



# Configuration
# config_file     = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/_config_2026/UDP_faster_rcnn_TranVal_test.py'
# checkpoint_file = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r18_TrainVal_17e_5_times/seed0/epoch_17.pth'
img_path        = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/4_025950.jpg'

# 1. Build the model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 2. Pre-load the image (Optional but recommended)
# Loading it into memory once ensures you are testing the MODEL speed, 
# not your hard drive's read speed.
img = mmcv.imread(img_path)

# 3. WARM-UP Phase
# Run inference a few times to initialize CUDA buffers (results are discarded)
print("Warming up...")
for _ in range(10):
    _ = inference_detector(model, img)

# 4. TIMING Phase
iterations = 100  # Number of times to average
print(f"Starting timing loop for {iterations} iterations...")

t_start = time_synchronized()
for _ in range(iterations):
    _ = inference_detector(model, img)
t_end = time_synchronized()

# 5. Calculate and Print Results
total_time = t_end - t_start
avg_time = total_time / iterations
fps = 1.0 / avg_time

print(f'\n--- Results ---')
print(f'Total time for {iterations} runs: {total_time:.4f}s')
print(f'Average inference time: {avg_time:.4f}s')
print(f'FPS: {fps:.2f}')