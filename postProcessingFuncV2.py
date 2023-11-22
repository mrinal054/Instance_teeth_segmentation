# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:40:56 2022

@author: mrinal
"""
import numpy as np
import cv2
from morph import get_ccomps

def post_process2D(im, name, thresh = 0.9, DILATION=False, KERNEL_SIZE=(5,5), DEL=30): 
    print("Post processing %s ... ... " % name, end='')
    # Output image
    pp_im = np.zeros((im.shape[0], im.shape[1]), dtype=im.dtype)
 
    # Find labels in the current image
    lbls = set(np.unique(im))
    lbls.remove(0) # remove label 0. It is background
    
    # lbls = [lbl for lbl in lbls if lbl<33] ######## temp line
    
    #  # Convert to gray image
    if len(im.shape) == 3 and im.shape[2] == 3: 
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert to gray. Required to create binary image
   
    # Create dict to store ccomps which are not largest
    not_lrg_ccomps = dict()
    for lbl in lbls: not_lrg_ccomps[lbl] = []
    not_lrg_ccomps = []

    lrg_ccomps = []
    lrg_img = np.zeros(im.shape, dtype='uint8')
    for lbl in lbls:
#        print('Processing image: %s, label: %d ...' % (name, lbl))
        
        # Keep the current labeled tooth and remove other labeled teeth.        
        m_im = np.zeros(im.shape, dtype='uint8') # mask prediction
        m_idx = np.where(im == lbl) # indices where current label exists
        m_im[m_idx] = lbl
        
        # Convert to binary
        ret, bin_img = cv2.threshold(m_im, thresh, 255, cv2.THRESH_BINARY)
        
        # Dilate image
        if  DILATION:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
            bin_img = cv2.dilate(bin_img, kernel, iterations=1)
            
        ccomps, num = get_ccomps(bin_img)
        
        # Find the largest ccomp
        lrg_size = 0
        for i in range(num):
            if len(ccomps[i]) > lrg_size: 
                lrg_size = len(ccomps[i])
                lrg_idx = i
        
        lrg_ccomp = ccomps[lrg_idx]
        lrg_ccomps.append([lrg_ccomp, lbl])
        lrg_img[lrg_ccomp[:,0], lrg_ccomp[:,1]] = lbl
        pp_im[lrg_ccomp[:,0], lrg_ccomp[:,1]] = lbl
        # Store ccomps that are not largest
        for k, v in ccomps.items():
            if k != lrg_idx:
                not_lrg_ccomps.append(v)
        
    # Sort not_lrg_ccomps according to the size
    sorted_not_lrg_ccomps = sorted(not_lrg_ccomps, key=lambda x:len(x), reverse=True)    
        
    # Start checking non-largest ccomps
    # Iterate over each ccomp
    for val in sorted_not_lrg_ccomps: 
        if not len(val) == 0:
            # Create binary mask
            m_not_lrg = np.zeros(im.shape, dtype='uint8')
            m_not_lrg[val[:,0], val[:,1]] = 255
            
            # Find contours
            cntrs, hierarchy = cv2.findContours(m_not_lrg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find neighbor pixels
            nb_pxls_x = []
            nb_pxls_y = []
            
            x = cntrs[0][:,0][:,0]
            y = cntrs[0][:,0][:,1]
            
            nxt_x = [i+1  if i+1 < im.shape[1] else i for i in x]
            nxt_y = [i+1  if i+1 < im.shape[0] else i for i in y]
            prev_x = [i-1  if i-1 >= 0 else i for i in x]
            prev_y = [i-1  if i-1 >= 0 else i for i in y]
            
            # 8 neighborhood
            nb_pxls_x.extend(nxt_x)
            nb_pxls_y.extend(y)
            
            nb_pxls_x.extend(x)
            nb_pxls_y.extend(nxt_y)
    
            nb_pxls_x.extend(prev_x)
            nb_pxls_y.extend(y)
    
            nb_pxls_x.extend(x)
            nb_pxls_y.extend(prev_y)        
            
            nb_pxls_x.extend(nxt_x)
            nb_pxls_y.extend(nxt_y)
    
            nb_pxls_x.extend(nxt_x)
            nb_pxls_y.extend(prev_y)  
            
            nb_pxls_x.extend(prev_x)
            nb_pxls_y.extend(nxt_y)      
            
            nb_pxls_x.extend(prev_x)
            nb_pxls_y.extend(prev_y)     
            
            curr_lbl = im[y[0], x[0]] # label of the current ccomp

            # Get neighbor pixel intensity
            intensities = pp_im[nb_pxls_y, nb_pxls_x]
            intensities = intensities.tolist()
                      
            # Remove 0 if there exists other labels
            if len(np.unique(intensities)) > 1:
                intensities = [i for i in intensities if i != 0]
            
            # Remove current label if there exists other labels
            if not np.all(intensities == curr_lbl):
                intensities = [i for i in intensities if i != curr_lbl]
                
            # Find the intesity with highest frequency 
            highest_freq = max(set(intensities), key = intensities.count)
            
            pp_im[val[:,0], val[:,1]] = highest_freq
    

#    if DEL not None:
                
    print("Done")
    
    return pp_im


#%% Implement bounding box

def bbox2D(im, name, thresh=0.9):
    # Convert to gray image
    if len(im.shape) == 3 and im.shape[2] == 3: 
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert to gray. Required to create binary image
    
    # Find labels in the current image
    lbls = set(np.unique(im))
    lbls.remove(0) # remove label 0. It is background
    
    # Store bbox info in a list
    bbox_info = []
    
    for lbl in lbls:        
        # Keep the current labeled tooth and remove other labeled teeth.        
        m_im = np.zeros(im.shape, dtype=im.dtype) # mask prediction
        m_idx = np.where(im == lbl) # indices where current label exists
        m_im[m_idx] = lbl
        
        # Convert to binary        
        ret, bin_img = cv2.threshold(m_im, thresh, 255, cv2.THRESH_BINARY)
        
        # Find contours
        cntrs, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right
        cntrs = sorted(cntrs, key=lambda x: cv2.boundingRect(x)[0])
        
        # Append bbox info
        for cntr in cntrs:
            x,y,w,h = cv2.boundingRect(cntr)
            bbox_info.append([name, int(lbl), int(x), int(y), int(x+w), int(y+h)]) # Append info of the current contour
            
    return bbox_info
