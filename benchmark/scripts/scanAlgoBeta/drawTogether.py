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
from algorithms import *

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

scanTag = "algoBeta"


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBase = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/" + scanTag + "/"
    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/" + scanTag

    algos = [MM_CPP, BCOOFD_CPP]
    methodTags = list(map(lambda algo: algo.name, algos))

    csvTemplate = "config.csv"
    valueVec = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 88.0]
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

    elapsedTimeAll, cacheMissAll, periodAll, fro, eb = compareMethod(exeSpace, commonBase, scanTag,
                                                                     algos, csvTemplate, valueVec, reRun)
    groupLine.DrawFigure(periodAll, elapsedTimeAll,
                         methodTags,
                         "value of beta", "elapsed time (ms)", 0, 1,
                         figPath + "/" + scanTag + "_elapsedTime",
                         True)
    groupLine.DrawFigureYnormal(periodAll, cacheMissAll,
                                methodTags,
                                "value of beta", "cacheMiss (%)", 0, 1,
                                figPath + "/" + scanTag + "_cacheMiss",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                fro * 100.0,
                                methodTags,
                                "value of beta", "normalized error %", 0, 1, figPath + "/" + scanTag + "_froError",
                                True)
    groupLine.DrawFigureYnormal(periodAll,
                                eb * 100.0,
                                methodTags,
                                "value of beta", "error bound ratio %", 0, 1, figPath + "/" + scanTag + "_ebRatio",
                                True)


if __name__ == "__main__":
    main()
