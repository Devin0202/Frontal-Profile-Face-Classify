# -*- coding: utf-8 -*-
"""@brief
Language:
Goal:
PS:
"""

import sys
import os
import time
import timeit
import re
import concurrent.futures

import shutil
import cv2
import re

"""Common utilities
Functions:
1. safeDirectory
2. makeAbsDirs
3. globalStart
4. globalEnd
5. traversFilesInDir
"""

def safeDirectory(fDir):
    if str == type(fDir):
        safeDir = os.path.normpath(fDir)
        safeDir += os.path.sep
    else:
        print("Error type of input!!!")
        sys.exit(0)
    return safeDir

def makeAbsDirs(fDir, fExistencePermitted = True):
    safeDir = safeDirectory(fDir)
    if os.path.isabs(safeDir):
        try:
            if not os.path.exists(safeDir):
                os.makedirs(safeDir)
            else:
                if not fExistencePermitted:
                    print("The folder had been existed!!!")
                    sys.exit(0)
                else:
                    pass
        except Exception as e:
            print(e)
            sys.exit(0)
        else:
            print("Create: " + safeDir + "    OK")
            return safeDir
    else:
        print("Please use absolute path!!!")
        sys.exit(0)

def globalStart():
    print("LocalSystem: " + os.name)
    print("Python Ver: " + sys.version)
    timeStampFormat = "%Y-%m-%d %H:%M:%S"
    print(time.strftime(timeStampFormat, time.localtime()))
    globalT = timeit.default_timer()
    print()
    return globalT

def globalEnd(fGlobalT):
    timeStampFormat = "%Y-%m-%d %H:%M:%S"
    globalElapsed = (timeit.default_timer() - fGlobalT) / 60
    print()
    print(time.strftime(timeStampFormat, time.localtime()))
    print("Finished in {:.2f}m".format(globalElapsed))

def concurrentWork(fMaxload, fFn, *fArgs, \
    isProcess = True, isConcurrent = True):
    if isConcurrent:
        if isProcess:
            executor = \
            concurrent.futures.ProcessPoolExecutor(max_workers = fMaxload)
        else:
            executor = \
            concurrent.futures.ThreadPoolExecutor(max_workers = fMaxload)
        results = list(executor.map(fFn, *fArgs))
    else:
        results = list(map(fFn, *fArgs))
    return results

def traversFilesInDir(fSrcRoot, fBlackList=[]):
    rtv = []
    srcRoot = safeDirectory(fSrcRoot)
    if os.path.exists(srcRoot):
        for rt, dirs, files in os.walk(srcRoot):
            for name in files:
                if rt in fBlackList:
                    continue
                else:
                    rtv.append(os.path.join(rt, name))
    else:
        print("Please use correct path!!!")
        sys.exit(0)
    return rtv
"""Definition region

Class:

Constants:

Functions:
1. 
"""

def extractCfpData(dstFile, dstFrontal, dstProfile, src):
    """Extract and divide frontal/profile face images from cfp-dataset.

    Arguments:
        dstFile {str} -- [in] Write the locations of images after processing.
        dstFrontal {str} -- [in] Where to save frontal face images.
        dstProfile {str} -- [in] Where to save profile face images.
        src {str} -- [in] Where to load cfp-dataset.
    """
    fCnt, pCnt = 0, 0
    dstF = makeAbsDirs(dstFrontal)
    dstP = makeAbsDirs(dstProfile)
    picList = traversFilesInDir(src)
    with open(dstFile, 'w') as fw:
        for i in picList:
            if "frontal" in i:
                fCnt += 1
                newOne = str(fCnt) + '.' \
                        + i.split(os.path.sep)[-1].split('.')[-1]
                newOne = dstF + newOne
                shutil.copy(i, newOne)
            elif "profile" in i:
                pCnt += 1
                newOne = str(pCnt) + '.' \
                        + i.split(os.path.sep)[-1].split('.')[-1]
                newOne = dstP + newOne
                shutil.copy(i, newOne)
            else:
                continue
            fw.write(newOne + os.linesep)
        
    print("Frontal num: ", fCnt)
    print("Profile num: ", pCnt)
    return

def extractAFLW(fileSrc, imageSrc):
    """Show face-rectangle from AFLW-dataset by landmarks.

    Arguments:
        fileSrc {str} -- [in] Folder which landmark-label files stored in.
        imageSrc {str} -- [in] Folder which images stored in.
    """
    fileSrc = safeDirectory(fileSrc)
    imageSrc = safeDirectory(imageSrc)
    fileList = traversFilesInDir(fileSrc)

    for i in fileList:
        lines = []
        l, t, r, b = sys.maxsize, sys.maxsize, 0, 0
        with open(i, 'r') as fr:
            lines = fr.readlines()
        for j in lines:
            l = min(l, float(j.split()[1]))
            t = min(t, float(j.split()[2]))
            r = max(r, float(j.split()[1]))
            b = max(b, float(j.split()[2]))
        imgName = os.path.basename(i)
        imgName = imgName.split('-')[0] + ".jpg"
        imgName = imageSrc + imgName

        halfEdge = int(max(r - l, b - t) / 2 * 1.5)
        centerX = int((l + r) / 2)
        centerY = int((t + b) / 2)
        if os.path.exists(imgName):
            img = cv2.imread(imgName)
            # img = cv2.rectangle(img, (centerX - halfEdge, centerY - halfEdge), (centerX + halfEdge, centerY + halfEdge), (0, 255, 0))

            img = img[centerX - halfEdge:centerX + halfEdge, centerY - halfEdge:centerY + halfEdge]
            cv2.imshow("show", img)
            cv2.waitKey()
        return

def extractWilderFace(fileSrc, imageSrc, dst):
    """Extract face-rectangle from WilderFace-dataset by labels.

    Arguments:
        fileSrc {str} -- [in] Path to label files.
        imageSrc {str} -- [in] Folder which images stored in.
        dst {[type]} -- [in] Where to save face-rectangle images.
    """
    cnt = 0
    dst = makeAbsDirs(dst)
    if not (os.path.exists(fileSrc) and os.path.exists(imageSrc)):
        print("Source error!!!")
        sys.exit(0)
    else:
        dst = safeDirectory(dst)
        lines = []
        with open(fileSrc, 'r') as fr:
            lines = fr.readlines()

        useEntry = [False, None]
        numEntry = None
        for i in lines:
            if useEntry[0]:
                numEntry = int(i)
                useEntry[0] = False
                # print(numEntry)
                continue

            if numEntry:
                roi = [int(j) for j in i.split()[:5]]
                numEntry -= 1
                # print(roi)
                if 1 == roi[4] and 50 < roi[3] and 50 < roi[2]:
                    l = roi[1]
                    t = roi[0]
                    r = roi[1] + roi[3]
                    b = roi[0] + roi[2]
                    halfEdge = int(max(r - l, b - t) / 2 * 1)
                    centerX = (l + r) // 2
                    centerY = (t + b) // 2
                    l = centerX - halfEdge
                    t = centerY - halfEdge
                    r = centerX + halfEdge
                    b = centerY + halfEdge
                    try:
                        img = useEntry[1][l:r, t:b]
                        savePath = dst + "1144-" + str(cnt) + ".jpg"
                        cv2.imwrite(savePath, img)
                        cnt += 1
                        # cv2.imshow("show", img)
                        # cv2.waitKey()
                    except BaseException as identifier:
                        print("Exception occured!!! Roi maybe over-ranged")
            else:
                useEntry = [False, None]

            # if re.match(".--(Handshaking|Dancing|Couple).+jpg$", i):
            if re.match(".--.+jpg$", i):
                imgPath = imageSrc + os.path.sep + i[:-1]
                useEntry = [True, cv2.imread(imgPath)]
                # print(useEntry)
        return

if "__main__" == __name__:
    globalT0 = globalStart()
### Parameters region
    src = "/home/devin/MyData/cfp-dataset/Data/Images/"
    dstF = "/home/devin/MyGit/TfLab/FrontalProfileFace/Data/f0"
    dstP = "/home/devin/MyGit/TfLab/FrontalProfileFace/Data/p0"

###Job region
    # extractCfpData("PicList.txt", dstF, dstP, src)

    # extractAFLW("/media/devin/OpenImage600/DataSet/AFLW-Style/AFLW-Convert/annotation/0/", "/media/devin/OpenImage600/DataSet/AFLW-Style/AFLW-Convert/aflw-Original/0/")

    extractWilderFace("/media/devin/OpenImage600/WildFace/wider_face_split/wider_face_train_bbx_gt.txt", "/media/devin/OpenImage600/WildFace/WIDER_train/images", "/home/devin/MyGit/TfLab/FrontalProfileFace/Data/f2")

    globalEnd(globalT0)