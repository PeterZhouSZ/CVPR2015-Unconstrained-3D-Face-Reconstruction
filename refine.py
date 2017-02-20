# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 18:44:28 2017

@author: hks
"""

import copy
import OBJ
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import os
from PIL import Image,ImageDraw

from generate2 import getAllLandmark, calP, computeL

def getM(template, landmarkAll, landmarkIndex, imgSetDir):
    M = []
    X = np.array(template.v)
    landmark3D = X[landmarkIndex]
    Xh = np.c_[X, np.ones((len(X),1))]
    imgList = os.listdir(imgSetDir)
    for (l, imgName) in zip(landmarkAll, imgList):
        p = calP(l[19:], landmark3D)
        imgPath = os.path.join(imgSetDir, imgName)
        im = np.array(Image.open(imgPath).convert('L'))
        proPts = np.round(map(lambda x:np.dot(p, x), Xh))[:,:2]
        tempM = map(lambda x:im[x[1], x[0]], proPts)
        M.append(tempM)
    return np.array(M, dtype = np.float)
    
def getNorm(template, landmarkAll, landmarkIndex, imgSetDir):
    M = getM(template, landmarkAll, landmarkIndex, imgSetDir)
    vNum = len(template.v)
    imgNum = len(M)
    iniN = np.c_[np.ones((vNum, 1)), np.array(template.vn)].T
    Norm = iniN
    rho = np.ones((imgNum, vNum))
    for i in range(3):
        (rho, L) = updateLP(M, Norm, rho)
        Norm = computeN(L, rho, M)
        print ('{} times'.format(i))       
    return Norm

def computeN(L, rho, M):
    R = M/rho
    Rc = R - np.repeat(L[:,:1], len(M.T), axis=1)
    N = np.linalg.lstsq(L[:,1:], Rc)[0]
    return np.c_[np.ones((len(M.T), 1)), np.array(N.T)].T

def updateRho(M, l, N):
    R = l.dot(N)
    pointNum = len(M.T)
    rhoOne = np.zeros((pointNum, 1))
    for i in range(pointNum):
        rhoOne[i] = np.inner(M[:,i],R[:,i])/np.inner(R[:,i],R[:,i])
    rho = (np.repeat(rhoOne, len(M), axis = 1)).T
    return rho

#更新L跟Rho
def updateLP(M, Norm, rho):
    for i in range(3):
        #L = (M/rho).dot(np.linalg.pinv(N))
        L = (np.linalg.lstsq((Norm*rho[:4,:]).T, (M).T)[0]).T
        print np.linalg.norm(M-rho*(L.dot(Norm)))         
        rho = updateRho(M, L, Norm)
        #rho = M/(L.dot(N))
        print np.linalg.norm(M-rho*(L.dot(Norm)))
    return rho, L

def drawNorm(imgPath, template, Norm, P):
    radius = 0.5
    img = Image.open(imgPath)
    imDraw = ImageDraw.Draw(img)
    V = template.v
    for (point, n) in zip(V, Norm):
        point2D = P.dot(np.r_[point,[1]])[:2]
        point2DP = P.dot(np.r_[point + n,[1]])[:2]
        imDraw.line((tuple(point2D.astype(int)), tuple(point2DP.astype(int))),fill = 'red')
        imDraw.ellipse((point2DP[0]-radius, point2DP[1]-radius, point2DP[0]+radius, point2DP[1]+radius), fill = 'black')
    return img

def computeH(L, X, oriNorm):
    Lx = L.dot(X)
    Lx = Lx.reshape((len(Lx)/3,3))        
    H = []
    count = 0
    for (h, no) in zip(Lx, oriNorm):      
        if count in bIndex:
            H.append(h)
        else:
            hN = h.dot(no)/np.linalg.norm(h)
            if hN > 0:
                H.append(-np.linalg.norm(h)*no)
            else:
                H.append(-np.linalg.norm(h)*no)
        count += 1 
        
    return H

#load landmark index without contour
def loadLandmark(fp):
    text = open(fp, 'r')    
    landmarkIndex = []
    for line in text:
        line = line.split()[0]
        landmarkIndex.append(int(float(line))-1)
    return landmarkIndex[19:]
    
def computeX(template, L, H, D, landmarkAll, landmarkIndex):
    X = np.array(template.v)
    ptsNum = len(X)
    left = L.dot(L)
    right = L.dot(H)
    landmark3D = X[landmarkIndex]
    rowForP = np.array(map(lambda x:x, xrange(2*ptsNum))).repeat(3)
    colForP = (np.tile(np.array(map(lambda x:x, xrange(3*ptsNum))).reshape((ptsNum,3)),2)).flatten()
    for landmark2D in landmarkAll:
        P = calP(landmark2D[19:], landmark3D)
        valForP = np.tile(P[0:2,0:3].flatten(),ptsNum)
        Pplus = csr_matrix((valForP, (rowForP, colForP)), shape=(2*ptsNum, 3*ptsNum))
        W = np.zeros((2*vCount,1))
        for count, index in enumerate(landmarkIndex):
            W[2*index] = landmark2D[count, 0] - P[0, 3]
            W[2*index+1] = landmark2D[count, 1] - P[1, 3]
        tempL = Pplus.dot(D)
        left += lam*tempL.T.dot(tempL)
        right += lam*Pplus.T.dot(W)
    #return lsqr(left, np.array(right))[0]
    return spsolve(left, right)

            

if __name__ == '__main__':
    lam = 0.01
    rootDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\render'
    imgSetDir = os.path.join(rootDir, 'test')
    landmarkPath = os.path.join(rootDir, 'landmark.txt')
    neibPath = os.path.join(rootDir, 'neib')
    bIndex = pickle.load(open(os.path.join(rootDir, 'edgeSet'),'r'))
    template = OBJ.obj(os.path.join(rootDir, 'warping.obj'))    
    template.load()
    temp = copy.deepcopy(template)
    vCount = len(template.v)
    landmarkIndex = loadLandmark(landmarkPath)
    landmarkAll = getAllLandmark(imgSetDir)
    row = np.array(map(lambda x:[x*3,x*3+1, x*3+2], landmarkIndex))
    row = (row.reshape((1,row.size))).squeeze()
    dataD = np.ones(row.shape)
    D = csr_matrix((dataD, (row, row)), shape=(3*vCount, 3*vCount))
    Norm = getNorm(template, landmarkAll, landmarkIndex, imgSetDir)
    oriNorm = Norm[1:].T
    oriNormN = (np.array(map(np.linalg.norm, oriNorm))).reshape((vCount,1))
    oriNorm = oriNorm/oriNormN
    L = computeL(template, neibPath)
    X0 = np.array(template.v).reshape((3*vCount,1))
    X = X0
    Norm = oriNorm
    for i in range(1):
        H = computeH(L, X, Norm)
        H = L.dot(X)
        #H = np.array(H).reshape((3*vCount,1))
        #right = np.array(H).reshape((3*vCount,1))
        #solutionX = lsqr(L, right)[0]
        solutionX = computeX(temp, L, H, D, landmarkAll, landmarkIndex)
        #xx = inv(L.T.dot(L)).dot(L.T).dot(b)
        temp.v = solutionX.reshape((vCount,3))
        temp.save(r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data\tempResult\normRec{}.obj'.format(i))
        X = solutionX

    
    
    
    