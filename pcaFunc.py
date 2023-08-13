# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 23:19:54 2022

@author: mrinal

Reference:
     https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

"""
import cv2
import numpy as np
from sklearn.decomposition import PCA
import math

class pca:

    def pc(img, n_components):
        '''
            - It returns start and end points of eigen vectors
        Input
        ------
            img: Image
            n_components: no. of principal components
            
        Output
        -------
            pca_points: A list containing eigen vectors. 
                        If n_components=2, then pca_points will be [[x01, y01, x02, y02], [x11, y11, x12, y12]]
        '''
        # Convert to gray image
        if len(img.shape) == 3 and img.shape[2] == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray. Required to create binary image
        X = np.argwhere(img == 255)
        
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        # Store points of eigen vector. Format: [x1, y1, x2, y2]
        pca_points = [] 
        
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)        
            pca_points.append([int(pca.mean_[1]), int(pca.mean_[0]), int((pca.mean_ + v)[1]), int((pca.mean_ + v)[0])])
        
        return pca_points
            
    
    def angle(pca_points):
        # Calculate angle between the PC1 and x-axis
        '''
        Formula:
            slope, m = (y2 - y1)/(x2 - x1)
            angle = atan((m1 - m2)/(1 + m1m2)
            We will measure angle w.r.to the x-axis. So, m1 = 0.
            Finally, angle = atan(-m2)
        '''
        
#        m = (pca_points[3] - pca_points[1])/(pca_points[2] - pca_points[0] + 1e-6) # Slope
#        angle = math.degrees(math.atan(-m))
        angle = math.degrees(math.atan2(-(pca_points[3] - pca_points[1]), pca_points[2] - pca_points[0] + 1e-6))
        
        return angle
    
    def draw(imc, pca_points, angle):
        '''
        Draw PCs and angle on the image
        
        Input
        ------
        imc: Image
        pca_points: Terminal points of eigenvectors (or PCs). It is a list.
        angle: Angle between PC1 and x-axis
        
        Output
        -------
        imc: Image containing PCs and angle 
        '''
        # Convert to BGR, if not
        if (len(imc.shape) < 3) or (len(imc.shape) == 3 and imc.shape[2] == 1): 
            imc = cv2.cvtColor(imc, cv2.COLOR_GRAY2BGR) 
        
        for i in range(len(pca_points)):
            cv2.circle(imc, (pca_points[i][0], pca_points[i][1]), 5, [255,0,0], -1)
            cv2.circle(imc, (pca_points[i][2], pca_points[i][3]), 5, [255,0,0], -1)
            cv2.line(imc, (pca_points[i][0], pca_points[i][1]), (pca_points[i][2], pca_points[i][3]), [0,0,255], 2)
            if i == 0: # for pc1
                dx = pca_points[i][2] - pca_points[i][0]
                dy = pca_points[i][3] - pca_points[i][1]
                cv2.line(imc, (pca_points[i][0], pca_points[i][1]), (pca_points[i][0] - dx, pca_points[i][1] - dy), [0,255,0], 2)
                cv2.circle(imc, (pca_points[i][0] - dx, pca_points[i][1] - dy), 5, [0,255,0], -1)               
                        
        # Write text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 0, 255) # BGR, not RGB in open-cv
        thickness = 2
        font_pos = (pca_points[0][0]+15, pca_points[0][1]+15)
        text = str(round(angle, 2))
        
        imc = cv2.putText(imc, text, font_pos, font, 
                          font_scale, font_color, thickness, cv2.LINE_AA)  

        return imc
        
