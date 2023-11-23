# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 00:59:47 2022

@author: mrinal

v2:
    It is using postProcessingFuncV2

* It calculates following metrics for multi-label instance segmentation - 
        Precision, Recall, Sorensen Dice Score, and Mean IoU
        
* Metrics are calculated label-wise and image-wise, meaning that it calculates
metrics for each labels individually as well as metrics for an entire image that
may contain multiple labels.

Reference;
    https://stackoverflow.com/questions/21448310/how-do-you-find-common-sublists-between-two-lists

v3:
    OBB added

"""
import numpy as np
import os
import pandas as pd
import json
import cv2
from postProcessingFuncV2 import post_process2D, bbox2D
from color import color
import matplotlib.pyplot as plt
from my_utils import get_cords, label2obb
from shapely.geometry import Polygon

# Get color for each label
color = color()

# Load directories
gt_dir = r'.\dns-panoramic-images-ivisionlab\instance-segmentation\folds\fold5\masks'
pred_dir = r'.\SMP\prediction'

gt_obb_txt_dir = r'.\dns-panoramic-images-ivisionlab\instance-segmentation\folds\fold5\obb_mask'

# Save directories
save_dir_pred = os.path.join(pred_dir, 'pp_v4')
if not os.path.exists(save_dir_pred): os.makedirs(save_dir_pred)

save_dir_for_colored_label = os.path.join(pred_dir, 'original_colored_prediction') # stores original prediction with colored labels
if not os.path.exists(save_dir_for_colored_label): os.makedirs(save_dir_for_colored_label)

# Read names of images
all_names = os.listdir(gt_dir)
names = [name for name in all_names if '.png' in name.lower()]
#names = [names[0], names[1], names[2]]

# Parameters
EPSILON = 1e-6

SAVE_EXL = True # image-wise result stores in exel files
SAVE_JSON = True
IMSAVE = True # save post-processed image
STORE_DETAIL = True # store detailed result for each label in each image

PP = True # post process. It removes small objects and keeps the largest one only.
DILATION = True # performs dilation
KERNEL_SIZE = (5,5) # required for dilation

LABELING = True # color each tooth
TEXT = True # write the label number
BBOX = True # calculate coordinates for bounding boxes
DRAW_BBOX = True # draw bounding boxes
SAVE_OBB = True # save OBB info in a text file
#IMSHOW = True # enables image visualization

# Create dataframe to store records for each image. RIoU -> Rotated IoU
df = pd.DataFrame(index=[], columns = [
    'Name', 'Precision', 'Recall', 'Dice', 'IoU', 'RIoU'], dtype='object')

# Create dataframe to store records for all labels in each image
if STORE_DETAIL:
    df_detailed = pd.DataFrame(index=[], columns = [
        'SN', 'Name', 'Label', 'Precision', 'Recall', 'Dice', 'IoU', 'RIoU'], dtype='object')

# Create a dictionary to store metrics
metric = {} # Nested metric format: metric[image_name][label] = [precision, recall, dice, iou]

# Store bbox info in a list
bbox_info = []

for cnt, name in enumerate(names):
    name_only = os.path.splitext(name)[0] # remove extension
    
    print(f'Processing {name_only}')
    
    metric[name_only] = {} # Creating nested dictionary    
    
    'Create a text file to store obb_info'
    # DOTA format is followed
    if SAVE_OBB: f_obb = open(os.path.join(save_dir_pred, name_only + '.txt'), 'w')
    
    # Image-wise mean of metrics
    i_mp, i_mr, i_mdice, i_miou, i_mriou = [], [], [], [], []
    
    # Read images
    gt = cv2.imread(os.path.join(gt_dir, name))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) # Remove 3rd axis
    pred = cv2.imread(os.path.join(pred_dir, name_only + '.jpg'))    
    
    # Save colored label
    # Convert to RGB image. It is required to draw rectangles using cv2
    if (len(pred.shape) < 3) or (len(pred.shape) == 3 and pred.shape[2] == 1): 
        im_cLabel = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) # max intensity is 1
    else: im_cLabel = pred.copy()
    
    lbl_pred_before_pp = np.unique(im_cLabel)
    lbl_pred_before_pp = lbl_pred_before_pp[1:]
    
    # lbl_pred_before_pp = [lbl for lbl in lbl_pred_before_pp if lbl<33] ######## temp line
    
    for lbl in lbl_pred_before_pp:        
        # Set desired color for the RGB image for better visualization. 
        im_cLabel[np.all(im_cLabel == (lbl, lbl, lbl), axis=-1)] = color[lbl] # Set desire color
        
    cv2.imwrite(os.path.join(save_dir_for_colored_label, name), im_cLabel)
    
    'Post processing'
    # Post process: remove small objects
    if PP:
        pred = post_process2D(pred, name_only, thresh=0.9, DILATION=DILATION, KERNEL_SIZE=KERNEL_SIZE)

    if BBOX: 
        bbox = bbox2D(pred, name_only, thresh=0.9)
        bbox_info.extend(bbox)

#    # Save post-processed image
#    if PP and IMSAVE:
#        cv2.imwrite(os.path.join(save_dir_pred, 'pp_' + name), pred)  
    
    # Find labels in gt and prediction
    lbl_gt = set(np.unique(gt))
    lbl_gt.remove(0) # remove 0. It is background
    lbl_pred = set(np.unique(pred))
    lbl_pred.remove(0) # remove 0. It is background
    
    # All labels
    all_lbls = lbl_gt.union(lbl_pred)
    
    # Find labels that are not common in both gt and prediction. For such cases. IoU = 0
    diff1 = lbl_gt - lbl_pred
    diff2 = lbl_pred - lbl_gt
    diffs = diff1.union(diff2) # labels that do not exist in either gt or prediction
    
    # Set IoU == 0 for such labels
    if not len(diffs) == 0:
        for diff in diffs:
            p, r, dice, iou, riou = 0, 0, 0, 0, 0
            metric[name_only][str(diff)] = [p, r, dice, iou]
            print("%d %s: label: %s; Precision: %3.2f; Recall: %3.2f; Dice: %3.2f; IoU: %3.2f; RIoU: %3.2f"%
                  (cnt+1, name_only, diff, p, r, dice, iou, riou))
            
            # Store detailed result
            if STORE_DETAIL:
                df_detailed = df_detailed.append(
                        pd.Series([cnt+1, name_only, diff, p, r, dice, iou, riou], index=['SN', 'Name', 'Label', 'Precision', 'Recall', 'Dice', 'IoU', 'RIoU']), 
                        ignore_index = True)
    
    # Create blank image for obb        
    imc_obb = np.zeros(pred.shape, dtype=pred.dtype) # pred.shape = (1127, 1991)
    imc_obb = cv2.cvtColor(imc_obb, cv2.COLOR_GRAY2BGR)
    
    # Create color image of the prediction. Note that if PP is True, then the 
    # colored image will be generated for the post-processed prediction.
    
    # Convert to RGB image. It is required to draw rectangles using cv2
    if (len(pred.shape) < 3) or (len(pred.shape) == 3 and pred.shape[2] == 1): 
        imc = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) # max intensity is 1
    else: imc = pred.copy()
    
    for lbl in lbl_pred:
        # Set desired color for the RGB image for better visualization. 
        imc[np.all(imc == (lbl, lbl, lbl), axis=-1)] = color[lbl] # Set desire color
 
    # Find labels that are common in both gt and prediction.
    cmns = lbl_gt.intersection(lbl_pred)
    
    # Iterate over labels
    for cmn in cmns:
        gt_idx = np.where(gt == cmn)
        pred_idx = np.where(pred == cmn)
        
        # Convert to [(x1,y1), (x2,y2), ...]
        gt_lidx, pred_lidx = [], [] # List index        
        for i in range(len(gt_idx[0])): gt_lidx.append((gt_idx[0][i], gt_idx[1][i]))            
        for i in range(len(pred_idx[0])): pred_lidx.append((pred_idx[0][i], pred_idx[1][i]))
        
        # Get OBB
        obb_cord, iml_c, iml_c_no_tooth = label2obb(pred, cmn, DRAW_OBB=True)
        
        if SAVE_OBB:
            # Save in DOTA format (x1, y1, x2, y2, x3, y3, x4, y4, category, difficult)
            f_obb.write("%d %d %d %d %d %d %d %d %s %d\n" % (obb_cord[0][1], obb_cord[0][0], 
                                                            obb_cord[1][1], obb_cord[1][0],
                                                            obb_cord[2][1], obb_cord[2][0],
                                                            obb_cord[3][1], obb_cord[3][0],
                                                            str(cmn), 1))        
        'Combine output'
        imc_obb = imc_obb + iml_c_no_tooth
                
        # Calculate metrics
        gt_tidx = tuple(gt_lidx) # convert to tuple
        pred_tidx = tuple(pred_lidx) # convert to tuple 
        tp = set(gt_tidx).intersection(pred_tidx) # set operation 
        fp = set(pred_tidx).difference(gt_tidx) # set operation 
        fn = set(gt_tidx).difference(pred_tidx) # set operation 
        
        n_tp = len(tp)
        n_fp = len(fp)
        n_fn = len(fn)
        
        p = (n_tp/(n_tp + n_fp + EPSILON)) * 100
        r = (n_tp/(n_tp + n_fn + EPSILON)) * 100
        dice = (2 * n_tp / (2 * n_tp + n_fp + n_fn + EPSILON)) * 100
        iou = (n_tp/(n_tp + n_fp + n_fn + EPSILON)) * 100
        
        # Calculate rotated iou (riou)
        gt_obb_cords = get_cords(gt_obb_txt_dir, name_only, cmn)
        
        poly1 = Polygon([gt_obb_cords[0], gt_obb_cords[1], gt_obb_cords[2], gt_obb_cords[3]])
        poly2 = Polygon([obb_cord[0], obb_cord[1], obb_cord[2], obb_cord[3]])

        riou = (poly1.intersection(poly2).area / poly1.union(poly2).area) * 100
              
        print("%d %s: label: %s; Precision: %3.2f; Recall: %3.2f; Dice: %3.2f; IoU: %3.2f, RIoU: %3.2f"%
              (cnt+1, name_only, cmn, p, r, dice, iou, riou))
        
        metric[name_only][str(cmn)] = [p, r, dice, iou]
        
        # Store detailed result
        if STORE_DETAIL:
            df_detailed = df_detailed.append(
                    pd.Series([cnt+1, name_only, cmn, p, r, dice, iou, riou], index=['SN', 'Name', 'Label', 'Precision', 'Recall', 'Dice', 'IoU', 'RIoU']), 
                    ignore_index = True)
        
        # Keep appending metrics for all labels for the current image
        i_mp.append(p)
        i_mr.append(r)
        i_mdice.append(dice)
        i_miou.append(iou)
        i_mriou.append(riou)
    
    # Calculate mean of metrics for the current image
    i_mp = np.mean(i_mp)
    i_mr = np.mean(i_mr)
    i_mdice = np.mean(i_mdice)
    i_miou = np.mean(i_miou)
    i_mriou = np.mean(i_mriou)
    
    # Store results in the data frame
    tmp = pd.Series([name_only, i_mp, i_mr, i_mdice, i_miou, i_mriou], 
                    index=['Name', 'Precision', 'Recall', 'Dice', 'IoU', 'RIoU'])
    df = df.append(tmp, ignore_index = True)
    
    'Final combined output'      
    combined = imc_obb + imc
    
    f_obb.close()
    
    if IMSAVE:
        cv2.imwrite(os.path.join(save_dir_pred, 'obb_' + name), combined)
    
    # Save post-processed image
    if PP and IMSAVE:
        if LABELING: # color each tooth        
            if DRAW_BBOX:
                for bb in bbox:
                    bb = bb[2:] # removing name and label, keeping coordinates only
                    cv2.rectangle(imc, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 0, 255), thickness=2)
            
            for lbl in lbl_pred:
                # Write labels as text
                if TEXT:
                    cord = np.argwhere(pred == lbl)
                    cv2.putText(imc, str(lbl), (cord[0][1], cord[0][0]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
            cv2.imwrite(os.path.join(save_dir_pred, 'pp_' + name), imc)
        else:
            cv2.imwrite(os.path.join(save_dir_pred, 'pp_' + name), pred) 

        
    # break #name
    
# Print overall mean of metrics
print("Over all precision: %3.2f" % df["Precision"].mean())
print("Over all Recall: %3.2f" % df["Recall"].mean())
print("Over all dice: %3.2f" % df["Dice"].mean())
print("Over all IoU: %3.2f" % df["IoU"].mean())
print("Over all RIoU: %3.2f" % df["RIoU"].mean())

# Save results in an exel file
if SAVE_EXL:
    df.to_excel(os.path.join(save_dir_pred, 'result_image_wise.xlsx'), index=False)
    if STORE_DETAIL: df_detailed.to_excel(os.path.join(save_dir_pred, 'result_detailed.xlsx'), index=False)

# create json object from dictionary
if SAVE_JSON:
    json_write = json.dumps(metric)
    f = open(os.path.join(save_dir_pred, "metric.json"), "w")
    f.write(json_write)
    f.close()
    
    if BBOX:
        json_write = json.dumps(bbox_info)
        f = open(os.path.join(save_dir_pred, "bbox_info.json"), "w")
        f.write(json_write)
        f.close()

# Uncomment to read json file
#f = open(r"F:\Research\a_panoramic_teeth_segmentation\@DNS_PANORAMIC\EXP\prediction\metric.json")
#data = json.load(f) 
#f.close()

