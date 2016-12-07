# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:09:02 2016

@author: 开圣
"""
import urllib
from json import *
import os
import numpy as np
from PIL import Image,ImageDraw
from OBJ import obj
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import itertools
import time
#face++ api
from facepp import API,File
API_KEY = "b068f469bf92bbf202a2a351093f81c3"
API_SECRET = "9Bu9gj1RMTPM97htY7loSW-kxjXztZws"
api = API(API_KEY,API_SECRET)

def landmarkFromFacepp(imgPath):
    infor = api.detection.detect(img = File(imgPath))
    try:
        faceID = infor[u'face'][0][u'face_id']
    except:
        print ("no face detected")
    req_url = "http://api.faceplusplus.com/detection/landmark"
    params = urllib.urlencode({'api_secret':API_SECRET,'api_key':API_KEY,'face_id':faceID,'type':'83p'})
    req_rst = urllib.urlopen(req_url,params).read() # landmark data
    req_rst_dict = JSONDecoder().decode(req_rst)
    if len(req_rst_dict['result']) == 1:
        landmarkDict = req_rst_dict['result'][0]['landmark']
    imgInfo = Image.open(imgPath)
    xSize = imgInfo.size[0]
    ySize = imgInfo.size[1]
    landmarkList = sorted(list(landmarkDict))
    landmark_array = []
    for key in landmarkList:
        temp_xy = [landmarkDict[key]['x'],landmarkDict[key]['y']]
        landmark_array.append(np.array(temp_xy))
    landmarkArray = np.array(landmark_array)
    landmarkArray[:,0] = landmarkArray[:,0] * xSize / 100
    landmarkArray[:,1] = landmarkArray[:,1] * ySize / 100
    return landmarkArray

def getAllLandmark(imgDir):
    landmarkAll = []
    for i in os.listdir(imgDir):
        landmarkAll.append(landmarkFromFacepp(os.path.join(imgDir, i))[19:])
    return landmarkAll

#compute cot for L
def calCot(vec1, vec2):
    cos = vec1.dot(vec2)/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec2.dot(vec2)))
    sin = np.sqrt(1 - cos**2)
    cot = cos/sin
    return cot
# compute L
def computeL(template):
    ptsNum = len(template.v)
    L = np.zeros((3*ptsNum, 3*ptsNum))
    L = csr_matrix(L)
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
            
    return L


# compute P (weakly perspective)    
def calP(landmark2D, landmark3D):
    b1 = landmark2D[:,0]
    b2 = landmark2D[:,1]
    A = np.c_[landmark3D, np.ones((len(landmark3D), 1))]
    m1 = np.linalg.lstsq(A, b1)[0]
    m2 = np.linalg.lstsq(A, b2)[0]
    P = np.c_[m1, m2, np.array([0,0,0,1])]
    return P.T
    
# draw landmark     
def drawL(img, landmark):
    #img = Image.open(imgPath)
    imDraw = ImageDraw.Draw(img)
    radius = 20
    for p in landmark:
        x = p[0]
        y = p[1]
        imDraw.ellipse((x - radius,y -radius,x + radius, y + radius), fill = 'red', outline = 'red')
    return img

#3p vector turn to obj file

def projectL(P, landmark3D):
    lm = np.c_[landmark3D, np.ones((len(landmark3D), 1))]
    return P.dot(lm.T).T[:,0:2]    

#load landmark index without contour
def loadLandmark(fp):
    text = open(fp, 'r')    
    landmarkIndex = []
    for line in text:
        line = line.split()[0]
        landmarkIndex.append(int(float(line))-1)
    return landmarkIndex[19:]


    
def itera(template):
    L = computeL(template)
    vertex = np.array(template.v)
    X = vertex.reshape((3*vCount,1))
    landmark3D = vertex[landmarkIndex]
    Pset = []
    Wset = []
    for landmark2D in landmarkAll:
        P = calP(landmark2D, landmark3D)
        W = csr_matrix(np.zeros((2*vCount,1)))
        Pplus = csr_matrix(p.zeros((2*vCount, 3*vCount)))
        count = 0
        for index in landmarkIndex:
            Pplus[2*index:2*index+2, 3*index:3*index+3] = P[0:2,0:3]
            W[2*index] = landmark2D[count, 0] - P[0, 3]
            W[2*index+1] = landmark2D[count, 1] - P[1, 3]
            count = count + 1
        Pset.append(Pplus)
        Wset.append(W)     
    sumL = L.dot(L)    
    sumR = sumL.dot(X)
    for i in range(len(Pset)):
        tempL = Pset[i].dot(D)
        sumL = sumL + lamda * tempL.T.dot(tempL)
        sumR = sumR + lamda * (Pset[i].T).dot(Wset[i])
    newV = spsolve(sumL, sumR)
    template.v = newV.reshape((len(newV)/3, 3))
    template.vnCal()
    return template
    

    
#def main():
if __name__ == '__main__':
    lamda = 1
    time1 = time.time()
    rootDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data'
    imgSetDir = os.path.join(rootDir, 'test')
    landmarkPath = os.path.join(rootDir, 'landmark.txt')
    templatePath = os.path.join(rootDir, 'template.obj')
    tempPath = os.path.join(rootDir, 'tempResult')
    landmarkAll = getAllLandmark(imgSetDir)
    landmarkIndex = loadLandmark(landmarkPath)
    template = obj(templatePath)
    template.load()
    vCount = len(np.array(template.v))
    X0 = np.array(template.v).reshape((3*vCount,1))
    #selection matrix
    D = np.zeros((3*vCount, 3*vCount))
    D = csr_matrix(D)
    for index in landmarkIndex:
        for i in range(3):
            D[3*index+i, 3*index+i] = 1   
    for i in range(2):      
        template = itera(template)
        template.save(os.path.join(tempPath, 'iter{}.obj'.format(str(i))))
        print time.time() - time1
        time1 = time.time()
    
    
#    return template, pMatrix
    

        
            
        
    
    
    
    