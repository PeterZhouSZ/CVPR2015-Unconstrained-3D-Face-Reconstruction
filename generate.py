# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:09:02 2016

@author: 开圣
"""
import urllib
import pickle
from json import *
import os
import numpy as np
from PIL import Image,ImageDraw
from OBJ import obj
from functions import findContour
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import itertools
import time
#face++ api
from facepp import API,File
API_KEY = "*"
API_SECRET = "*"
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
        #landmarkAll.append(landmarkFromFacepp(os.path.join(imgDir, i))[19:])
        landmarkAll.append(landmarkFromFacepp(os.path.join(imgDir, i)))
    return landmarkAll

#compute cot for L
def calCot(vec):
    vec1 = vec[0]
    vec2 = vec[1]
    cos = vec1.dot(vec2)/(np.sqrt(vec1.dot(vec1))*np.sqrt(vec2.dot(vec2)))
    sin = np.sqrt(1 - cos**2)
    cot = cos/sin
    return cot
# compute L
def computeL(template):
    ptsNum = len(template.v)
    v = template.v
    rowSum = np.zeros((ptsNum, 1))
    row = []
    col = []
    valList = []
    nerb = pickle.load(open(neibPath, 'r'))
    for i in xrange(ptsNum):
        allNeib = set(nerb[i].nonzero()[1])
        tempNeib = filter(lambda x:x>i, allNeib)
        for n in tempNeib:
            common = allNeib.intersection(set(nerb[n].nonzero()[1]))
            allpair = map(lambda x:[v[x]-v[i], v[x]-v[n]], common)
            val = sum(map(calCot, allpair))/len(common)
            rowSum[i] = rowSum[i] + val
            rowSum[n] = rowSum[n] + val
            row.extend([i*3, i*3+1, i*3+2, n*3, n*3+1, n*3+2])
            col.extend([n*3, n*3+1, n*3+2, i*3, i*3+1, i*3+2])
            valList.extend([val, val, val, val, val, val])
    for idx, i in enumerate(rowSum):
        row.extend([idx*3, idx*3+1, idx*3+2])
        col.extend([idx*3, idx*3+1, idx*3+2])
        valList.extend([-i, -i, -i])
        
    return csr_matrix((np.array(valList), (np.array(row), np.array(col))), shape=(3*ptsNum, 3*ptsNum))
            
def neibInf(template):              
    row = []
    col = []
    boolList = []
    for face in template.face:
        for (i,j) in itertools.combinations(tuple(face), 2):
            row.extend([i, j])
            col.extend([j, i])
            boolList.extend([True, True])
    neib = csr_matrix((np.array(boolList, dtype=np.bool),(np.array(row), np.array(col))), shape=(ptsNum,ptsNum))
    return neib

    #LPlus = np.r_[np.ones((3,3*ptsNum)),L]      


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
    #return landmarkIndex[19:]
    return landmarkIndex

def itera(template):
    #L = computeL(template)
    vertex = np.array(template.v)
    ptsNum = len(vertex)
    X = vertex.reshape((3*vCount,1))
    landmark3D = vertex[landmarkIndex]
    sumL = L.dot(L)    
    sumR = sumL.dot(X)
    rowForP = np.array(map(lambda x:x, xrange(2*ptsNum))).repeat(3)
    colForP = (np.tile(np.array(map(lambda x:x, xrange(3*ptsNum))).reshape((ptsNum,3)),2)).flatten()
    for landmark2D in landmarkAll:
        P = calP(landmark2D, landmark3D)
        valForP = np.tile(P[0:2,0:3].flatten(),ptsNum)
        Pplus = csr_matrix((valForP, (rowForP, colForP)), shape=(2*ptsNum, 3*ptsNum))
        W = np.zeros((2*vCount,1))
        for count, index in enumerate(landmarkIndex):
            W[2*index] = landmark2D[count, 0] - P[0, 3]
            W[2*index+1] = landmark2D[count, 1] - P[1, 3]
        tempL = Pplus.dot(D)
        sumL = sumL + lamda * tempL.T.dot(tempL)
        sumR = sumR + lamda * Pplus.T.dot(W)
    newV = spsolve(sumL, sumR)
    template.v = newV.reshape((len(newV)/3, 3))
    return template

"""老版本
def itera(template):
    #L = computeL(template)
    vertex = np.array(template.v)
    X = vertex.reshape((3*vCount,1))
    landmark3D = vertex[landmarkIndex]
    sumL = L.dot(L)    
    sumR = sumL.dot(X)
    for landmark2D in landmarkAll:
        P = calP(landmark2D, landmark3D)
        W = np.zeros((2*vCount,1))
        Pplus = np.zeros((2*vCount, 3*vCount))
        count = 0
        for index in landmarkIndex:
            Pplus[2*index:2*index+2, 3*index:3*index+3] = P[0:2,0:3]
            W[2*index] = landmark2D[count, 0] - P[0, 3]
            W[2*index+1] = landmark2D[count, 1] - P[1, 3]
            count = count + 1
        tempL = csr_matrix(Pplus).dot(D)
        sumL = sumL + lamda * tempL.T.dot(tempL)
        sumR = sumR + lamda * (Pplus.T).dot(W)
    newV = spsolve(sumL, sumR)
    template.v = newV.reshape((len(newV)/3, 3))
    return template
"""   

    
#def main():
if __name__ == '__main__':
    lamda = 1
    time1 = time.time()
    rootDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data'
    imgSetDir = os.path.join(rootDir, 'test')
    landmarkPath = os.path.join(rootDir, 'landmark.txt')
    templatePath = os.path.join(rootDir, 'template.obj')
    neibPath = os.path.join(rootDir, 'neib')
    tempPath = os.path.join(rootDir, 'tempResult')
    landmarkAll = getAllLandmark(imgSetDir)
    landmarkIndex = loadLandmark(landmarkPath)
    template = obj(templatePath)
    template.load()
    vCount = len(np.array(template.v))
    X0 = np.array(template.v).reshape((3*vCount,1))
    #selection matrix
    row = np.array(map(lambda x:[x*3,x*3+1, x*3+2], landmarkIndex))
    row = (row.reshape((1,row.size))).squeeze()
    dataD = np.ones(row.shape)
    D = csr_matrix((dataD, (row, row)), shape=(3*vCount, 3*vCount))
    L = computeL(template)
    for i in range(2):      
        template = itera(template)
        template.save(os.path.join(tempPath, 'iter{}.obj'.format(str(i))))
        print time.time() - time1
        time1 = time.time()
"""
    lamda = 1
    rootDir = r'D:\WinPython-64bit-2.7.10.1\mine\Unconstrained 3D Face Reconstruction\data'
    imgSetDir = os.path.join(rootDir, 'test')
    landmarkPath = os.path.join(rootDir, 'landmark.txt')
    templatePath = os.path.join(rootDir, 'template.obj')
    neibPath = os.path.join(rootDir, 'neib')
    landmarkIndex = loadLandmark(landmarkPath)
    #selection matrix
    row = np.array(map(lambda x:[x*3,x*3+1, x*3+2], landmarkIndex))
    row = (row.reshape((1,row.size))).squeeze()
    dataD = np.ones(row.shape)
    D = csr_matrix((dataD, (row, row)), shape=(3*vCount, 3*vCount))
    for imgDir in os.listdir(imgSetDir):
        tempPath = os.path.join(imgSetDir, imgDir)
        landmarkAll = getAllLandmark(tempPath)
        template = obj(templatePath)
        template.load()
        vCount = len(np.array(template.v))
        X0 = np.array(template.v).reshape((3*vCount,1))
        L = computeL(template)
        template = itera(template)
        template.save(os.path.join(tempPath, 'warping.obj'.format(str(i))))

  
    import shutil    
    for img in os.listdir(imgSetDir):
        os.mkdir(os.path.join(imgSetDir, img.split('.')[0]))
        shutil.copy(os.path.join(imgSetDir, img), os.path.join(imgSetDir, img.split('.')[0]))
        os.remove(os.path.join(imgSetDir, img))
"""

        
            
        
    
    
    
    
