# -*- coding: utf-8 -*-
"""Brief
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

import random
import shutil

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
def divider(src, evaluateNum, evaluateName):
    """Seperate the dataset into training and evaluating part.

    Arguments:
        src {str} -- [in] Original images folder.
        evaluateNum {int} -- [in] Setting the number of evaluating images.
        evaluateName {str} -- [in] Setting the subname of evaluating folder.
    """
    if not os.path.exists(src):
        print("Src do not exist!!!")
        return
    else:
        imgList = traversFilesInDir(src)
        if 0 >= len(imgList) or evaluateNum > len(imgList):
            print("Invalid evaluate number for dividing!!!")
            return
        else:
            parentFolder = os.path.dirname(src)
            dst = makeAbsDirs(parentFolder + os.path.sep + evaluateName)
            random.shuffle(imgList)
            cnt = 0
            for i in imgList:
                if evaluateNum <= cnt:
                    break
                shutil.move(i, dst + os.path.basename(i))
                cnt += 1
        return


if "__main__" == __name__:
    globalT0 = globalStart()
### Parameters region
    src = "/home/devin/MyGit/TfLab/FrontalProfileFace/Data/p1"
### Job region
    divider(src, 500, "evaP")

    globalEnd(globalT0)
