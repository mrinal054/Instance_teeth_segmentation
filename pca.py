# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:31:43 2022

@author: mrina
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:09:51 2022

@author: mrinal
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import math

#rng = np.random.RandomState(1)
#X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
#plt.scatter(X[:, 0], X[:, 1])
#plt.axis('equal')
#
#pca = PCA(n_components=2)
#pca.fit(X)

loc = r'.\EXP\single_tooth'
names = os.listdir(loc)
name = names[5]
# Loading the image 
img = cv2.imread(os.path.join(loc, name)) #you can use any image you want.
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
X = np.argwhere(img == 255)

# Source: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
pca = PCA(n_components=2)
pca.fit(X)

X_pca = pca.transform(X)

print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

imc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 

# plot data
pca_points = [] # Store points of eigen vector. Format: [x1, y1, x2, y2]

plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    print(pca.mean_, pca.mean_ + v)
    
    pca_points.append([int(pca.mean_[1]), int(pca.mean_[0]), int((pca.mean_ + v)[1]), int((pca.mean_ + v)[0])])
    
    draw_vector(pca.mean_, pca.mean_ + v)
    
    cv2.circle(imc, (int(pca.mean_[1]), int(pca.mean_[0])), 5, [255,0,0], -1)
    cv2.circle(imc, (int((pca.mean_ + v)[1]), int((pca.mean_ + v)[0])), 5, [255,0,0], -1)
    cv2.line(imc, (int(pca.mean_[1]), int(pca.mean_[0])), (int((pca.mean_ + v)[1]), int((pca.mean_ + v)[0])), [0,0,255], 2)
    
    
plt.axis('equal');

# Calculate angle between the PC1 and x-axis
'''
Formula:
    slope, m = (y2 - y1)/(x2 - x1)
    angle = atan((m1 - m2)/(1 + m1m2)
    We will measure angle w.r.to the x-axis. So, m1 = 0.
    Finally, angle = atan(-m2)
'''

m = (pca_points[0][3] - pca_points[0][1])/(pca_points[0][2] - pca_points[0][0] + 1e-6) # Slope
angle = math.degrees(math.atan(-m))

# Write text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 0, 255) # BGR, not RGB in open-cv
thickness = 2
font_pos = (pca_points[0][0]+15, pca_points[0][1]+15)
text = str(round(angle, 2))

imc = cv2.putText(imc, text, font_pos, font, 
                  font_scale, font_color, thickness, cv2.LINE_AA)  
cv2.imshow("a", imc)
