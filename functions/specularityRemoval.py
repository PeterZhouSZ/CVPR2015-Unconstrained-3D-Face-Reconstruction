# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:33:42 2016

@author: hks
"""

import specularity_removal as sp
import os


imgDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\testData\specularityRemove\stk' 
imgList = []
for i in os.listdir(imgDir):
    tempPath = os.path.join(imgDir, i)
    if os.path.isfile(tempPath):
        imgList.append(tempPath)
resultDir = os.path.join(imgDir, 'result')
if not os.path.exists(resultDir):
    os.mkdir(resultDir)
sp.remove_specularity(imgList, resultDir)