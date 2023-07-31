#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'common')))

# I will streaming A and B!!!
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

scanTag = "batchSize"


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBase = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/" + scanTag + "2S" + "/"
    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/" + scanTag + "2S" + "CPP"

    algos = [MM_CPP, CRSV2_CPP, SMP_PCA_CPP, COUNT_SKETCH_CPP]
    methodTags = list(map(lambda algo: algo.name, algos))

    csvTemplate = "config.csv"
    valueVec = [10, 20, 50, 100, 250, 500, 1000]
    valueVecDisp = np.array(valueVec)
    # run
    reRun = 0
    if (len(sys.argv) < 2):
        os.system("mkdir ../../results")
        os.system("mkdir ../../figures")
        os.system("mkdir " + figPath)
        os.system("sudo rm -rf " + commonBase)
        os.system("sudo mkdir " + commonBase)
        reRun = 1
    # skech
    thrAll, lat95All, periodAll, fro, eb = compareMethodStreaming(exeSpace, commonBase, scanTag,
                                                                  algos, csvTemplate, valueVec, reRun)
    groupLine.DrawFigureYnormal(periodAll, thrAll / 1000.0,
                                methodTags,
                                "batch size (#rows)", "throughput (K elements/s)", 0, 1,
                                figPath + "/" + scanTag + "stream_cpp_thr",
                                True)
    groupLine.DrawFigure(periodAll, lat95All,
                         methodTags,
                         "batch size (#rows)", "95% latency (ms)", 0, 1,
                         figPath + "/" + scanTag + "stream_cpp_lat95",
                         True)
    groupLine.DrawFigureYnormal(periodAll, fro * 100.0,
                                methodTags,
                                "batch size (#rows)", "fro error (%)", 0, 1,
                                figPath + "/" + scanTag + "stream_cpp_fro",
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
