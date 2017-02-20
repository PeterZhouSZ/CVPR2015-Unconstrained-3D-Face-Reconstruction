# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:04:07 2016

@author: hks
"""

import OBJ
import numpy as np
from os.path import join
from PIL import Image, ImageDraw

dataDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\warpData'
objPath = join(dataDir, 'warp.obj')

def getSurfNorm(template, P, imgPath):
    img = Image.open(imgPath)
    norm = template.vn
    (sizeA, sizeB) = img.size
    surfNorm = np.zeros((sizeA, sizeB, 3))
    point2D = project(template.v, P)
    point2D = point2D.astype('int')
    for index,(p1,p2) in enumerate(point2D):
        surfNorm[p1,p2,:] = norm[index]
    
def project(point3D, P):
    lm = np.c_[point3D, np.ones((len(point3D), 1))]
    return P.dot(lm.T).T[:,0:2]        

im = Image.open(imgPath)
imDraw = ImageDraw.Draw(im)
radius = 1
for (x,y) in point2D:
    imDraw.ellipse((x - radius,y -radius,x + radius, y + radius), fill = 'red', outline = 'red')

template = OBJ.obj(objPath)
template.load()
norm = template.vn