# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:56:15 2023

@author: user

It calculate uncertainty expained in my paper.

First, it takes one bounding box in one txt file.
It calculates 'd' between this bbox and all gt bboxes.
Then, it takes the minimum 'd's.
Finally, it calculates the standard deviation of all distances.

It doesn't care the class or iou.
It only calculates the nearest box distance from all gt boxes.

In short, it is a code that find the std of all the minimum distances for each predicted bbox.

* Check!!
If the d_i == 0, I excluded this value.
What if I include these values?? The result would be different?
"""

import os
import math
import numpy


# labels_pred = r'D:\Github\DUP\final_results\v7\pred_labels_val_over_conf_30'  # pascal format
# labels_pred = r'D:\Github\DUP\final_results\centernet\pred_labels_val_over_conf_30'
labels_pred = r'E:\gdrive\My Drive\IEEE_Access\code\metric\uncertainty\pascal_val_pred_v9_conf30'
labels_gt   = r'D:\Github\DUP\dataset\val_labels_pascal'   # pascal format

distance = [] # all distance having the minimal distance of each bbox of each image.

i=0
# Looping all pred txt files 
for file_pred in os.listdir(labels_pred):
    if file_pred.endswith('.txt'):
        i += 1
        print(i, ", " ,file_pred)
        file_pred_path = os.path.join(labels_pred, file_pred)
        file_gt_path   = os.path.join(labels_gt,   file_pred)
        # print(file_gt_path)
        
        

        # opening each txt file.
        with open(file_pred_path, 'r') as f:
            lines_pred = f.readlines()
        
        # Looping each line in prediction txt file.
        for pred_line in lines_pred:
            # print(pred_line)
            l_pred_line = int(pred_line.split()[2])
            t_pred_line = int(pred_line.split()[3])
            # print(lt_pred_line)
            
            # So far, we got the left top point of one bbox of each predicted bbox in one txt file.
            # Now, compare it to every gt box left top points.
            with open(file_gt_path, 'r') as gt:
                lines_gt   = gt.readlines()
                # print(lines)
                
            # Looping all GT lines. and took left top point.    
            d_i = []
            for gt_line in lines_gt:
                l_gt_line = int(gt_line.split()[1])
                t_gt_line = int(gt_line.split()[2])
                
                temp = math.sqrt( (l_gt_line - l_pred_line)**2 + (t_gt_line - t_pred_line)**2 )
                d_i.append(temp)
                # print('d_i =', d_i)
                
                # print(l_gt_line)
                # print(t_gt_line)
            # print('min =', min(d_i))
            
            # if min(d_i) > 0:
            distance.append(min(d_i))
            # print('distance =', distance)

print(numpy.std(distance, ddof=1))        


