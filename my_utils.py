# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:21:51 2021

@author: mrinal
"""
import numpy as np
import os
from PIL import Image
from patch import Patch
import cv2
from pcaFunc import pca

def create_patch(base_loc, base_patch_dir, patch_size, overlap, mode, names,  csv_output):
    
    img_patch_dir = os.path.join(base_patch_dir, mode, 'images')
    mask_patch_dir = os.path.join(base_patch_dir, mode, 'mask')
    
    if not os.path.exists(img_patch_dir): os.makedirs(img_patch_dir)
    if not os.path.exists(mask_patch_dir): os.makedirs(mask_patch_dir)
    
    for i, name in enumerate(names):
        print('%d. Processing: %s' %(i+1, name))
        # Read image and mask
        img = Image.open(os.path.join(base_loc, 'images', name + '.jpg'))
        mask = Image.open(os.path.join(base_loc, 'masks', name + '.png'))
        # Create patches
        patch = Patch(patch_size, overlap, patch_name=name, csv_output=csv_output)
        patches_im, _, _ = patch.patch2d(np.array(img))    
        patches_mask, _, _ = patch.patch2d(np.array(mask))    
        # Save patches
        patch.save2d(patches_im, save_dir=img_patch_dir, ext = '.png')
        patch.save2d(patches_mask, save_dir=mask_patch_dir, ext = '.png')


def get_cords(file_dir, file_name, lbl):
    '''
    This function is written explicitly for DNS panaromic project where we need
    to get the coordinates of a given label from a text file. 
    
    Inputs
    ---------    
    file_dir: file directory
    file_name: text file name without extension
    lbl: label number
    
    Outputs
    ---------
    Returns a list that contains coordinates. Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    '''

    text = []
    
    f = open(os.path.join(file_dir, file_name + '.txt'), 'r')
    
    for line in f: 
        text.append([int(x) for x in line.split()])
        
    f.close()
    
    text = np.array(text)
    
    lbls = text[:,8] # 8th pythonic column has the label value
    
    idx = int(np.argwhere(lbls==lbl)) # idx is the row number that has lbl in text array
    
    cords = text[idx][0:8].tolist() # get the row indicated by the idx. Only collect first 8 elements.
                                   # Rest of them are not coordinates.    
    
    # Rearrange to [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    cords = [[cords[1], cords[0]], [cords[3], cords[2]], [cords[5], cords[4]], [cords[7], cords[6]]]
    
    return cords


# =============================================================================
# # Example
# if __name__ == '__main__':
#     
#     file_dir = r'F:\Research\a_panoramic_teeth_segmentation\@DNS_PANORAMIC\predictions_3090\recurrent_resunet_a\2022-06-10_11-29-39\cp-0050\pp_v4'
#     
#     file_name = 'cate1-00001'
#     
#     lbl = 10
#     
#     crds = get_crds(file_dir, file_name, lbl)
#     
#     print(crds)
# 
# =============================================================================

def rotateImage(image, pivot, angle):
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    row,col = image.shape
    center=tuple(pivot)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def label2obb(im_org, lbl, DRAW_OBB=False):
    '''
    This function generates obb coordinates for a given label.
    
    Input
    ---------
    im_org: Original image
    lbl: (int) label
    DRAW_OBB: (bool) Whether to draw obb
    
    Output
    ---------
    Returns obb cooridinates. If DRAW_OBB is True, then also returns -
            iml_c: obb drawn with teeth 
            iml_c_no_tooth: obb drawn without teeth
            
            Format of obb is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
    '''
    
    
    iml = np.zeros((im_org.shape[0], im_org.shape[1]), dtype=im_org.dtype) # image for current label
    iml[np.where(im_org == lbl)] = 255
    
    'Calculate PCA to find PCs'
    pca_obj = pca
    pca_points = pca_obj.pc(iml, n_components=2)
    pc1_theta1 = pca_obj.angle(pca_points[0])  # Angle between PC1 and x-axis
    
    'PCA points'
    x0 = pca_points[0][0] # pca mean
    y0 = pca_points[0][1]
    
    x1 = pca_points[0][2]
    y1 = pca_points[0][3]
    
    dx, dy = x1 - x0, y1 - y0
    
    x2 = pca_points[0][0] - dx
    y2 = pca_points[0][1] - dy
    
    pivot = [x0, y0]
    
    'Rotate image to make the tooth straight'
    if pc1_theta1 < 0: rot_angle = 180+(90-pc1_theta1)
    else: rot_angle = 90-pc1_theta1
    
    iml_r = rotateImage(iml, pivot, rot_angle)
    iml_r[np.where(iml_r > 0)] = 255
    
    'Detect contour points'
    cntrs, hierarchy = cv2.findContours(iml_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Sort contours from left to right
    cntrs = sorted(cntrs, key=lambda x: cv2.boundingRect(x)[0])
       
    cord_list = []
    for cntr in cntrs:
        x,y,w,h = cv2.boundingRect(cntr)
        cord_list.append([x,y,w,h]) # Append info of the current contour
        # bbox_info.append([name_only, int(lbl), int(x), int(y), int(x+w), int(y+h)]) # Append bbox info
        'set bbox info outside of the function'
    
    # Four points of the bbox: P1(x1,y1), P2(x2,y1), P3(x2,y2), P4(x1,y2)    
    x1, y1, x2, y2 = x, y, x+w, y+h
    
    # x and y cooridnates of the points        
    p_x = [x1, x2, x2, x1]
    p_y = [y1, y1, y2, y2]
              
    'Rotate back the corner points'
    obb_cord = []
    for i in range(len(p_x)):
        corner_img = np.zeros(im_org.shape, dtype=im_org.dtype)
        corner_img[p_y[i], p_x[i]] = 255
        rot_corner_img = rotateImage(corner_img, pivot, -rot_angle) # rotating back
        idx = np.where(rot_corner_img == np.max(rot_corner_img)) # index of the max intensity
        obb_cord.append([idx[0][0], idx[1][0]])
    
    #obb_info.append([name_only] + [int(lbl)] + obb_cord[0] + obb_cord[1] + obb_cord[2] + obb_cord[3])
    
    'Draw obb'
    if DRAW_OBB:
        # With tooth
        iml_c = cv2.cvtColor(iml, cv2.COLOR_GRAY2BGR) # create color image
        for cord in obb_cord:
            cv2.circle(iml_c, (cord[1], cord[0]), 5, [255,0,0], -1)
            
        cv2.line(iml_c, (obb_cord[0][1], obb_cord[0][0]), (obb_cord[1][1], obb_cord[1][0]), [0,0,255], 2)
        cv2.line(iml_c, (obb_cord[1][1], obb_cord[1][0]), (obb_cord[2][1], obb_cord[2][0]), [0,0,255], 2)
        cv2.line(iml_c, (obb_cord[2][1], obb_cord[2][0]), (obb_cord[3][1], obb_cord[3][0]), [0,0,255], 2)
        cv2.line(iml_c, (obb_cord[3][1], obb_cord[3][0]), (obb_cord[0][1], obb_cord[0][0]), [0,0,255], 2)
        
        # Without tooth
        iml_c_no_tooth = np.zeros(im_org.shape, im_org.dtype)
        iml_c_no_tooth = cv2.cvtColor(iml_c_no_tooth, cv2.COLOR_GRAY2BGR) # create color image
        for cord in obb_cord:
            cv2.circle(iml_c_no_tooth, (cord[1], cord[0]), 5, [255,0,0], -1)
            
        cv2.line(iml_c_no_tooth, (obb_cord[0][1], obb_cord[0][0]), (obb_cord[1][1], obb_cord[1][0]), [0,0,255], 2)
        cv2.line(iml_c_no_tooth, (obb_cord[1][1], obb_cord[1][0]), (obb_cord[2][1], obb_cord[2][0]), [0,0,255], 2)
        cv2.line(iml_c_no_tooth, (obb_cord[2][1], obb_cord[2][0]), (obb_cord[3][1], obb_cord[3][0]), [0,0,255], 2)
        cv2.line(iml_c_no_tooth, (obb_cord[3][1], obb_cord[3][0]), (obb_cord[0][1], obb_cord[0][0]), [0,0,255], 2)
    
    if DRAW_OBB: return obb_cord, iml_c, iml_c_no_tooth
    else: return obb_cord
    

# =============================================================================
# # How to use
# obb_cord, iml_c, iml_c_no_tooth = label2obb(im_org, lbl, DRAW_OBB=True)
# =============================================================================
