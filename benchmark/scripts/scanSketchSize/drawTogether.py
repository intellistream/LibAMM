#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'common')))

import csv
import numpy as np
import accuBar as accuBar
import groupBar as groupBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it

import matplotlib
import numpy as np
import pylab
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, LinearLocator
import pandas as pd
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


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBase = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/" + scanTag + "/"
    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/" + scanTag

    algos = [MM_CPP, COOFD_CPP, BCOOFD_CPP, TUG_OF_WAR_CPP]
    methodTags = list(map(lambda algo: algo.name, algos))

    csvTemplate = "config.csv"

    valueVec = [10, 25, 50, 100, 200, 300, 400, 500]
    valueVecDisp = np.array(valueVec)

    # run
    reRun = 0
    if (len(sys.argv) < 2):
        os.system("mkdir ../../results")
        os.system("mkdir ../../figures")
        os.system("sudo rm -rf " + commonBase)
        os.system("sudo mkdir " + commonBase)
        os.system("mkdir " + figPath)
        #
        reRun = 1
        tRows = len(resultPaths)
        tCols = len(valueVec)
        elapseTimeAllSum = np.zeros((tRows, tCols))
        froErroAllSum = np.zeros((tRows, tCols))
        errorBoundRatioSum = np.zeros((tRows, tCols))
        cacheMissAll = np.zeros((tRows, tCols))

    rounds = 10
    for i in range(rounds):
        elapseTimeAll, ch, periodAll, fro, eb = compareMethod(exeSpace, commonBase, scanTag,
                                                                         algos, csvTemplate, valueVec, reRun)
        elapseTimeAllSum = elapseTimeAllSum + elapseTimeAll
        froErroAllSum = froErroAllSum + fro
        errorBoundRatioSum = errorBoundRatioSum + eb
        cacheMissAll = cacheMissAll + ch
    elapseTimeAllSum = elapseTimeAllSum / float(rounds)
    froErroAllSum = froErroAllSum / float(rounds)
    errorBoundRatioSum = errorBoundRatioSum / float(rounds)
    cacheMissAll = cacheMissAll / float(rounds)
    # evaTypes = ['FDAMM', 'MM', 'Co-FD', 'BCO-FD']

    # elapseTimeVecFD, cacheMissVecFD, cacheRefVecFD = readResultVector(valueVecRun, resultPathFDAMM)
    # elapseTimeVecCoFD, cacheMissVecCoFD, cacheRefVecCoFD = readResultVector(valueVecRun, resultPathCoFD)
    # elapseTimeVeCB, cacheMissVecB, cacheRefVecB = readResultVector(valueVecRun, resultPathBetaCoFD)

    # os.system("mkdir " + figPath)
    groupLine.DrawFigureXYnormal(periodAll,
                                 1 / elapseTimeAllSum,
                                 evaTypes,
                                 "sketch dimension", "1/elapsed time (1/ms)", 0, 1, figPath + "/" + "_elapsedTime",
                                 True)
    groupLine.DrawFigureXYnormal(periodAll,
                                 froErroAllSum * 100.0,
                                 evaTypes,
                                 "sketch dimension", "normalized error %", 0, 1, figPath + "/" + "_froError",
                                 True)
    groupLine.DrawFigureXYnormal(periodAll,
                                 errorBoundRatioSum * 100.0,
                                 evaTypes,
                                 "sketch dimension", "error bound ratio %", 0, 1, figPath + "/" + "_ebRatio",
                                 True)
    groupLine.DrawFigureXYnormal(periodAll,
                                 cacheMissAll,
                                 evaTypes,
                                 "sketch dimension", "cache miss %", 0, 1, figPath + "/" + "_cachemiss",
                                 True)
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
