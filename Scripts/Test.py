import tensorflow as tf
from tensorflow.keras import Model

import numpy as np
import onnxruntime as ort

import subprocess
import os
import sys
import time
import timeit
import re
import concurrent.futures
import cv2

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

tfModelFile = "tmp/2020060910" + os.path.sep
# dataDir = "/home/devin/MyGit/TfLab/FrontalProfileFace/Data/TestData"
# """Load tensorflow model"""
# subprocess.run(["python", "-m", "tf2onnx.convert", "--saved-model", \
#                 tfModelFile, "--output", tfModelFile + "model.onnx",  "--inputs", "input_1:0[1, 3, 48, 48]", "--inputs-as-nchw", "input_1:0"])

testList = traversFilesInDir("/home/devin/MyGit/TfLab/FrontalProfileFace/Data/TestData/1")
sess_ort = ort.InferenceSession(tfModelFile + "fp.onnx")
input_name = sess_ort.get_inputs()[0].name
output_name = sess_ort.get_outputs()[0].name

cnt = 0
for i in testList:
    mat = cv2.imread(i)
    diff = mat.shape[1] - mat.shape[0]
    if 0 < diff:
        half = diff // 2 
        mat = mat[:, half:(half - diff)]
    elif 0 > diff:
        diff = -diff
        half = diff // 2 
        mat = mat[half:(half - diff), :]
    else:
        pass
    mat = cv2.resize(mat, (48, 48), interpolation=cv2.INTER_CUBIC)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    mat = mat.reshape((1, 48, 48, 3))
    mat = mat.transpose((0, 3, 1, 2))
    img = mat / 255.0
    img = img.astype('float32')

    res = sess_ort.run(output_names=[output_name], input_feed={input_name: img})
    if res[0].flatten()[0] > 0.5:
        cnt += 1
    else:
        print(i)
        print(res[0].flatten()[0])
print("Acc: ", cnt / len(testList))