# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:58:38 2016

@author: hks
"""
import OBJ
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import os
from PIL import Image,ImageDraw

def getM(template, pSet, imgSetDir):
    M = []
    X = np.array(template.v)
    Xh = np.c_[X, np.ones((len(X),1))]
    imgList = os.listdir(imgSetDir)
    for (p, imgName) in zip(pSet, imgList):
        imgPath = os.path.join(imgSetDir, imgName)
        im = np.array(Image.open(imgPath).convert('L'))
        proPts = np.round(map(lambda x:np.dot(p, x), Xh))[:,:2]
        tempM = map(lambda x:im[x[1], x[0]], proPts)
        M.append(tempM)
    return np.array(M)
    
def getNorm(template, M):
    vNum = len(template.v)
    imgNum = len(M)
    iniN = np.c_[np.ones((vNum, 1)), np.array(template.vn)].T
    iniL = M.dot(np.linalg.pinv(iniN))
    Norm = iniN
    L = iniL
    rho = updateRho(M, L, Norm.reshape((4,Norm.size/4)))
    for i in range(3):
        if i!=0:
            rho = updateRho(M, L, Norm)
            L = updateL(M, rho, Norm)
        productRL = matrixL(L, vNum, imgNum).dot(matrixRho(rho[0]))
        vecNorm = computeN(productRL, M, Norm)
        Norm = vecNorm.reshape((4,vecNorm.size/4))
        print np.linalg.norm(M-rho*L.dot(Norm))
        
    return Norm

def computeN(productRL, M, N):
    left = productRL.T.dot(productRL) + csr_matrix(np.eye(productRL.shape[1]))
    right = productRL.T.dot(matrix2vector(M)) + matrix2vector(N)
    return spsolve(left, right)



def matrix2vector(matrix):
    return matrix.reshape((matrix.size,1))

def matrixRho(rho):
    rhoRepeat = np.repeat(rho, 4)
    return csr_matrix(np.diag(tuple(rhoRepeat)))

def matrixL(l, vNum, imgNum):
    matrixL = np.zeros((imgNum*vNum, 4*vNum))
    for i in range(imgNum):
        for j in range(vNum):
            matrixL[vNum*i+j,4*j:4*j+4] = l[i]
    return csr_matrix(matrixL)

#更新 L
def updateL(M, rho, N):
    return (M/rho).dot(np.linalg.pinv(N))

#更新每次迭代后的rho
def updateRho(M, l, N):
    rhoVal = M/(l.dot(N))
    meanRho = np.mean(rhoVal, axis = 0)
    meanRho = meanRho.reshape(meanRho.size, 1)
    rho = (np.repeat(meanRho, len(M), axis = 1)).T
    return rho
    
if __name__ == '__main__':
    pSet = pickle.load(open(r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\warpData\pMatrix','r'))
    imgSetDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\warpData\lp'
    template = OBJ.obj(r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\warpData\warp.obj')
    template.load()
    M = getM(template, pSet, imgSetDir)
    oriNorm = template.vn