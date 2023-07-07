#!/usr/bin/env python3
import csv
import numpy as np
import accuBar as accuBar
import groupBar as groupBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pylab
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, LinearLocator
import os
import pandas as pd
import sys
from OoOCommon import *
import time

# OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 30
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

MARKERS = (['*', '|', 'v', "^", "", "h", "<", ">", "+", "d", "<", "|", "", "+", "_"])
# you may want to change the color map for different figures
COLOR_MAP = (
    '#B03A2E', '#2874A6', '#239B56', '#7D3C98', '#FFFFFF', '#F1C40F', '#F5CBA7', '#82E0AA', '#AEB6BF', '#AA4499')
# you may want to change the patterns for different figures
PATTERNS = (["////", "o", "", "||", "-", "//", "\\", "o", "O", "////", ".", "|||", "o", "---", "+", "\\\\", "*"])
LABEL_WEIGHT = 'bold'
LINE_COLORS = COLOR_MAP
LINE_WIDTH = 3.0
MARKER_SIZE = 15.0
MARKER_FREQUENCY = 1000

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42

scanTag = "sketchDimension"


def singleRun(exePath, singleValue, resultPath, configTemplate):
    # resultFolder="singleValueTests"
    configFname = "config_" + scanTag + str(singleValue) + ".csv"
    # configTemplate = "config.csv"
    # clear old files
    os.system("cd " + exePath + "&& rm *.csv")

    editConfig(configTemplate, exePath + configFname, scanTag, singleValue)
    # prepare new file
    # run
    os.system("cd " + exePath + "&& ./benchmarkPCA " + configFname)
    # copy result
    os.system("rm -rf " + resultPath + "/" + str(singleValue))
    os.system("mkdir " + resultPath + "/" + str(singleValue))
    os.system("cd " + exePath + "&& cp *.csv " + resultPath + "/" + str(singleValue))


def runScanVector(exePath, singleValueVec, resultPath, templateName="config.csv"):
    for i in singleValueVec:
        singleRun(exePath, i, resultPath, templateName)


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/PCA.csv"
    AMMElapsedTime = readConfig(resultFname, "AMMElapsedTime")
    AMMError = readConfig(resultFname, "AMMError")
    # cacheMiss = readConfig(resultFname, "cacheMiss")
    # cacheRefs = readConfig(resultFname, "cacheRefs")
    ElseElapsedTime = readConfig(resultFname, "ElseElapsedTime")
    PCAError = readConfig(resultFname, "PCAError")
    # errorBoundRatio = readConfig(resultFname, "errorBoundRatio")
    # return elapsedTime, cacheMiss, cacheRefs, PCAError, errorBoundRatio
    return AMMElapsedTime, AMMError, ElseElapsedTime, PCAError


def readResultVector(singleValueVec, resultPath):
    AMMElapsedTimeVec = []
    AMMErrorVec = []
    cacheMissVec = []
    cacheRefVec = []
    ElseElapsedTimeVec = []
    PCAErrorVec = []
    errorBoundRatioVec = []
    for i in singleValueVec:
        # elapsedTime, cacheMiss, cacheRefs, PCAError, errorBoundRatio = readResultSingle(i, resultPath)
        AMMElapsedTime, AMMError, ElseElapsedTime, PCAError = readResultSingle(i, resultPath)
        AMMElapsedTimeVec.append(float(AMMElapsedTime) / 1000.0)
        AMMErrorVec.append(float(AMMError))
        # cacheMissVec.append(float(cacheMiss))
        # cacheRefVec.append(float(cacheRefs))
        ElseElapsedTimeVec.append(float(ElseElapsedTime) / 1000.0)
        PCAErrorVec.append(float(PCAError))
        # errorBoundRatioVec.append(float(errorBoundRatio))
    # return np.array(elapseTimeVec), np.array(cacheMissVec), np.array(cacheRefVec), np.array(PCAErrorVec), np.array(errorBoundRatioVec)
    return np.array(AMMElapsedTimeVec), np.array(AMMErrorVec), np.array(ElseElapsedTimeVec), np.array(PCAErrorVec)


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplates, periodVec, reRun=1):
    
    AMMElapsedTimeAll = []
    AMMErrorAll = []
    cacheMissAll= []
    cacheRefAll = []
    ElseElapsedTimeAll = []
    PCAErrorAll = []
    errorBoundRatioAll = []
    periodAll = []

    for i in range(len(csvTemplates)):
        resultPath = commonPathBase + resultPaths[i]
        if (reRun == 1):
            Path(resultPath).mkdir(parents=True, exist_ok=True)
            runScanVector(exeSpace, periodVec, resultPath, csvTemplates[i])
        # elapsedTime, cacheMiss, cacheRef, error, eb = readResultVector(periodVec, resultPath)
        AMMElapsedTime, AMMerror, ElseElapsedTime, PCAError = readResultVector(periodVec, resultPath)
        AMMElapsedTimeAll.append(AMMElapsedTime)
        AMMErrorAll.append(AMMerror)
        ElseElapsedTimeAll.append(ElseElapsedTime)
        PCAErrorAll.append(PCAError)
        periodAll.append(periodVec)

    return np.array(AMMElapsedTimeAll), np.array(AMMErrorAll), np.array(ElseElapsedTimeAll), np.array(PCAErrorAll), periodAll


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/"
    resultPath = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/results/" + scanTag
    figPath = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/figures/" + scanTag
    configTemplate = exeSpace + "config.csv"
    commonBase = resultPath + "/"
    resultPaths = ["mm", "crs", "tugOfWar", "cooFD", "smp-pca"]
    csvTemplates = ["config_"+i+".csv" for i in resultPaths]
    evaTypes = resultPaths
    valueVec = [100, 500, 1000, 3000, 5000]
    print(configTemplate)
    reRun = 0
    # run
    if (len(sys.argv) < 2):
        if os.path.exists(resultPath): shutil.rmtree(resultPath)
        if os.path.exists(figPath): shutil.rmtree(figPath)
        Path(resultPath).mkdir(parents=True, exist_ok=True)
        Path(figPath).mkdir(parents=True, exist_ok=True)
        #
        reRun = 1
        tRows = len(resultPaths)
        tCols = len(valueVec)
        AMMElapsedTimeAllSum = np.zeros((tRows, tCols))
        AMMErrorAllSum = np.zeros((tRows, tCols))
        ElseElapsedTimeAllSum = np.zeros((tRows, tCols))
        PCAErrorAllSum = np.zeros((tRows, tCols))
        errorBoundRatioSum = np.zeros((tRows, tCols))
        cacheMissAll = np.zeros((tRows, tCols))
    rounds = 1
    for i in range(rounds):
        # elapseTimeAll, ch, periodAll, error, eb = compareMethod(exeSpace, commonBase, resultPaths, csvTemplates, valueVec, reRun)
        AMMElapsedTimeAll, AMMErrorAll, ElseElapsedTimeAll, PCAErrorAll, periodAll = compareMethod(exeSpace, commonBase, resultPaths, csvTemplates, valueVec, reRun)
        AMMElapsedTimeAllSum = AMMElapsedTimeAllSum + AMMElapsedTimeAll
        AMMErrorAllSum = AMMErrorAllSum + AMMErrorAll
        ElseElapsedTimeAllSum = ElseElapsedTimeAllSum + ElseElapsedTimeAll
        PCAErrorAllSum = PCAErrorAllSum + PCAErrorAll
        # errorBoundRatioSum = errorBoundRatioSum + eb
        # cacheMissAll = cacheMissAll + ch
    AMMElapsedTimeAllSum /= float(rounds)
    AMMErrorAllSum /= float(rounds)
    ElseElapsedTimeAllSum /= float(rounds)
    PCAErrorAllSum /= float(rounds)
    # errorBoundRatioSum = errorBoundRatioSum / float(rounds)
    # cacheMissAll = cacheMissAll / float(rounds)
    # evaTypes = ['FDAMM', 'MM', 'Co-FD', 'BCO-FD']

    # elapseTimeVecFD, cacheMissVecFD, cacheRefVecFD = readResultVector(valueVecRun, resultPathFDAMM)
    # elapseTimeVecCoFD, cacheMissVecCoFD, cacheRefVecCoFD = readResultVector(valueVecRun, resultPathCoFD)
    # elapseTimeVeCB, cacheMissVecB, cacheRefVecB = readResultVector(valueVecRun, resultPathBetaCoFD)

    # os.system("mkdir " + figPath)
    groupLine.DrawFigureXYnormal(periodAll,
                                 1 / AMMElapsedTimeAllSum,
                                 evaTypes,
                                 "sketch dimension", "1/AMM elapsed time (1/ms)", 0, 1, figPath + "/" + "AMMElapsedTime",
                                 True)
    groupLine.DrawFigureXYnormal(periodAll,
                                 AMMErrorAllSum * 100.0,
                                 evaTypes,
                                 "sketch dimension", "AMM error %", 0, 1, figPath + "/" + "AMMError",
                                 True)
    groupLine.DrawFigureXYnormal(periodAll,
                                 1 / ElseElapsedTimeAllSum,
                                 evaTypes,
                                 "sketch dimension", "1/Else elapsed time (1/ms)", 0, 1, figPath + "/" + "ElseElapsedTime",
                                 True)
    groupLine.DrawFigureXYnormal(periodAll,
                                 PCAErrorAllSum * 100.0,
                                 evaTypes,
                                 "sketch dimension", "PCA error %", 0, 1, figPath + "/" + "PCAError",
                                 True)
    # groupLine.DrawFigureXYnormal(periodAll,
    #                              errorBoundRatioSum * 100.0,
    #                              evaTypes,
    #                              "sketch dimension", "error bound ratio %", 0, 1, figPath + "/" + "_ebRatio",
    #                              True)
    # groupLine.DrawFigureXYnormal(periodAll,
    #                              cacheMissAll,
    #                              evaTypes,
    #                              "sketch dimension", "cache miss %", 0, 1, figPath + "/" + "_cachemiss",
    #                              True)
    # draw2yLine("watermark time (ms)",singleValueVecDisp,lat95Vec,errVec,"95% Latency (ms)","Error","ms","",figPath+"wm_lat")
    # draw2yLine("watermark time (ms)",singleValueVecDisp,thrVec,errVec,"Throughput (KTp/s)","Error","KTp/s","",figPath+"wm_thr")
    # draw2yLine("watermark time (ms)",singleValueVecDisp,lat95Vec,compVec,"95% Latency (ms)","Completeness","ms","",figPath+"wm_omp")
    # groupLine.DrawFigureYnormal([singleValueVec,singleValueVec],[errVec,aqpErrVec],['w/o aqp',"w/ MeanAqp"],"watermark time (ms)","Error",0,1,figPath+"wm_MeanAqp",True)
    # print(errVec)
    # print(aqpErrVec)
    # print(elapseTimeVecFD)
    # readResultsingleValue(50,resultPath)


if __name__ == "__main__":
    main()
