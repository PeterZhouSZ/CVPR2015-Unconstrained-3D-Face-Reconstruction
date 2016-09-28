# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:30:56 2016

@author: 开圣
"""
import itertools
import numpy as np
from scipy.sparse import csr_matrix


    

def calCot(vec1, vec2):
    cos = vec1.dot(vec2)/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec2.dot(vec2)))
    sin = np.sqrt(1 - cos**2)
    cot = cos/sin
    return cot

def computeL(template):
    ptsNum = len(template.v)
    L = np.zeros((3*ptsNum, 3*ptsNum))
    for face in template.face:
        p = map(np.array, map(lambda x:template.v[x], face))
        perm = itertools.permutations((0,1,2))
        for (i,j,k) in perm:
            if L[3*face[i], 3*face[j]] == 0:
                val = calCot(p[k]-p[i], p[k]-p[j])
                L[3*face[i], 3*face[j]] = val
                L[3*face[i]+1, 3*face[j]+1] = val
                L[3*face[i]+2, 3*face[j]+2] = val
            else:
                val = 0.5 * (calCot(p[k]-p[i], p[k]-p[j]) + L[3*face[i], 3*face[j]])
                L[3*face[i], 3*face[j]] = val
                L[3*face[i]+1, 3*face[j]+1] = val
                L[3*face[i]+2, 3*face[j]+2] = val
    for i in range(ptsNum):
        tempSum = sum(L[3*i,:])
        L[3*i, 3*i] = -tempSum
        L[3*i+1, 3*i+1] = -tempSum
        L[3*i+2, 3*i+2] = -tempSum
    #LPlus = np.r_[np.ones((3,3*ptsNum)),L]
    L = csr_matrix(L)        
    return L


def computeTriangleArea(listVertex):
    vec1 = listVertex[0] - listVertex[1]
    vec2 = listVertex[0] - listVertex[2]
    return np.linalg.norm(np.cross(vec1, vec2))/2
    
"""
def computeH(template, L, norm):
    v = template.v
    ptsNum = len(v)
    cotVal = np.zeros((ptsNum, 1))
    areasVal = np.zeros((ptsNum, 1))
    for fa in template.face:
        p = map(lambda x:template.v[x], fa)
        area = computeTriangleArea(p)
        areasVal[fa] += area
        perm = itertools.permutations(fa, 2)
        for (i,j) in perm:
            #dotP = np.inner(v[i]-v[j], norm[j]-norm[i])
            dotP = np.inner(v[i]-v[j], -norm[i])
            cotVal[i] += 0.5 * L[3*i,3*j] * dotP
    h = cotVal / areasVal
    H = np.reshape(h * norm, (norm.size, 1))
    return H

def computeHN(template, L, norm):
    v = template.v
    ptsNum = len(v)
    cotVal = np.zeros((ptsNum, 3))
    areasVal = np.zeros((ptsNum, 1))
    for fa in template.face:
        p = map(lambda x:template.v[x], fa)
        area = computeTriangleArea(p)
        areasVal[fa] += area
        perm = itertools.permutations(fa, 2)
        for (i,j) in perm:
            vecT = v[j]-v[i]
            
            cotVal[i] += 0.5 * L[3*i,3*j] * vecT
    h = cotVal / areasVal
    H = np.reshape(h * norm, (norm.size, 1))
    return H
            
def getBoundaryIndex(filePath):
    bIndex = []
    for i in open(filePath, 'r'):
        bIndex.append(int(i))
    return bIndex
"""   

    
