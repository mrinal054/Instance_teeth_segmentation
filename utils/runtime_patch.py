from jenti.patch import Patch
import numpy as np
import random

def choose_fg_idx(
        patch_mask, # an nd-array with size patch_H x patch_W or patch_H x patch_W x patch_Ch
        fg_idx:list, # list of foreground indices. e.g. [2, 4, 7, 9]
        MAX_ROI:bool=True, # if true and the returned patch is a foreground patch, then it
                           # returns the patch that has maximum info or region of interest (roi) 
        ):
    """ 
    It is a helper function that picks a foreground index. If MAX_ROI is True,
    then it returns the index of that patch that has max info or roi in it. Otherwise,
    it returns a randomly chosen foreground index.
    
    Return
    --------
    It returns a foreground index. 
    """
    if MAX_ROI: # pick the index of the foreground patch that has maximum roi     
    
        max_nonzeros = 0 # Maximum no. of nonzeros. Initially set it to 0.
        final_fg_idx: int # index of the patch that has maximum roi
        
        for idx in fg_idx:
            
            x = patch_mask[idx] # get the foreground patch mask
            
            n_nonzero = np.count_nonzero(x) # no. of nonzeros in the patch mask
            
            # Compare with current max no. of nonzeros
            if n_nonzero > max_nonzeros: 
                final_fg_idx = idx # update index if new count is higher than the previous count
                max_nonzeros = n_nonzero # update max no. of nonzeros
                
        return final_fg_idx 
        
    else: # randomly pick a foreground index
        return random.choice(fg_idx)
        

def runtime_patch(
        image, # an nd-array with size H x W or H x W x Ch
        mask, # an nd-array with size H x W or H x W x Ch
        patch_shape:tuple=(256,256), # patch size
        overlap:tuple=(0,0), # overlap between adjacent patches
        FG_PROB:float=0.9, # probability of choosing a foreground
        MAX_ROI:bool=True, # if true and the returned patch is a foreground patch, then it
                           # returns the patch that has maximum info or region of interest (roi) 
        ):
    
    """
    This function returns an image patch and the corresponding mask patch. The patch
    can be a background patch or a foreground patch.
    
        foreground patch: It contains information or region of interest (roi)
        background patch: It does not contain any info or roi
        
    Return
    --------
    It returns an image patch and the corresponding mask patch. 
    Size of image/mask patch: (patch_H, patch_W, ch) or (patch_H, patch_W)
    """
    
    patch = Patch(patch_shape, overlap, patch_name='patch2d', csv_output=False)
    patch_img, _, _ = patch.patch2d(image)
    patch_mask, _, _ = patch.patch2d(mask)
    
    # Separate foreground (fg) and background
    fg_idx, bg_idx = [], []
    
    for i,x in enumerate(patch_mask):
        if np.sum(x) > 0: fg_idx.append(i) # fg
        else: bg_idx.append(i)  # background
    
    # If no foreground, then randomly return a background patch
    if len(fg_idx) == 0: 
        
        # Randomly choose a bg index
        final_bg_idx = random.choice(bg_idx)
        
        return patch_img[final_bg_idx], patch_mask[final_bg_idx]
    
    # If no background, then randomly return a foreground patch
    if len(bg_idx) == 0:
        
        final_fg_idx = choose_fg_idx(patch_mask, fg_idx, MAX_ROI)
        
        return patch_img[final_fg_idx], patch_mask[final_fg_idx]
       
    # Choose foreground or background based on a probability distribution
    fg_flag: bool
    fg_flag = True if np.random.uniform(low=0, high=1, size=1) <= FG_PROB else False
        
    if fg_flag: # pick a foreground patch
    
        final_fg_idx = choose_fg_idx(patch_mask, fg_idx, MAX_ROI)
            
        return patch_img[final_fg_idx], patch_mask[final_fg_idx]
        
    else: # pick a background

        final_bg_idx = random.choice(bg_idx)
        
        return patch_img[final_bg_idx], patch_mask[final_bg_idx]        

# =============================================================================
# # Example 
#         
# import cv2
# import matplotlib.pyplot as plt
# import os
# import random
# 
# # Parameters
# FG_PROB = 0.9 # probability of selecting a foreground image
# MAX_ROI = True # select the patch that has maximum roi 
# 
# # Directory
# img_dir = r'.\dataset\test\images'
# mask_dir = r'.\dataset\test\labels'
# 
# # List of images
# names = os.listdir(img_dir)
# 
# name = random.choice(names)
# 
# # Read image
# image = cv2.imread(os.path.join(img_dir, name))[:,:,::-1]
# mask = cv2.imread(os.path.join(mask_dir, name), 0)
# 
# mask = np.expand_dims(mask, axis=-1)
# 
# # Create patches
# patch_shape = [256, 256]
# overlap = [10,10] # overlap between two adjacent patches along both axes
#         
# ip, mp = runtime_patch(
#         image, 
#         mask, 
#         patch_shape=(256,256), 
#         overlap=(0,0), 
#         FG_PROB=0.9, 
#         MAX_ROI=True)  
# 
# 
# 
# fig, ax = plt.subplots(2,1, figsize=(15,7))
# ax[0].imshow(ip)
# ax[1].imshow(mp, cmap='gray')
# 
# =============================================================================

