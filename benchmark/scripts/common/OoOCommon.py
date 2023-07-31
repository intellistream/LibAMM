import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, LinearLocator
import os
import pandas as pd
import sys
import matplotlib.ticker as mtick

OPT_FONT_NAME = 'Helvetica'
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


def editConfig(src, dest, values):
    df = pd.read_csv(src)
    df = df.set_index('key')
    df = pd.concat([df, values])
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(dest, index=True, header=True)


def readConfig(src, key):
    df = pd.read_csv(src, header=None)
    rowIdx = 0
    idxt = 0
    for cell in df[0]:
        # print(cell)
        if (cell == key):
            rowIdx = idxt
            break
        idxt = idxt + 1
    return df[1][rowIdx]


def draw2yLine(NAME, Com, R1, R2, l1, l2, m1, m2, fname):
    fig, ax1 = plt.subplots(figsize=(10, 6.4))
    lines = [None] * 2
    # ax1.set_ylim(0, 1)
    print(Com)
    print(R1)
    lines[0], = ax1.plot(Com, R1, color=LINE_COLORS[0], \
                         linewidth=LINE_WIDTH, marker=MARKERS[0], \
                         markersize=MARKER_SIZE
                         #
                         )

    # plt.show()
    ax1.set_ylabel(m1, fontproperties=LABEL_FP)
    ax1.set_xlabel(NAME, fontproperties=LABEL_FP)
    # ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴
    plt.xticks(rotation=0, size=TICK_FONT_SIZE)
    plt.yticks(rotation=0, size=TICK_FONT_SIZE)
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    ax2 = ax1.twinx()

    # ax2.set_ylabel('latency/us')
    # ax2.set_ylim(0, 0.5)
    lines[1], = ax2.plot(Com, R2, color=LINE_COLORS[1], \
                         linewidth=LINE_WIDTH, marker=MARKERS[1], \
                         markersize=MARKER_SIZE)

    ax2.set_ylabel(m2, fontproperties=LABEL_FP)
    # ax2.vlines(192000, min(R2)-1, max(R2)+1, colors = "GREEN", linestyles = "dashed",label='total L1 size')
    # plt.grid(axis='y', color='gray')

    # style = dict(size=10, color='black')
    # ax2.hlines(tset, 0, x2_list[len(x2_list)-1]+width, colors = "r", linestyles = "dashed",label="tset")
    # ax2.text(4, tset, "$T_{set}$="+str(tset)+"us", ha='right', **style)

    # plt.xlabel('batch', fontproperties=LABEL_FP)

    # plt.xscale('log')
    # figure.xaxis.set_major_locator(LinearLocator(5))
    ax1.yaxis.set_major_locator(LinearLocator(5))
    ax2.yaxis.set_major_locator(LinearLocator(5))
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.legend(lines,
               [l1, l2],
               prop=LEGEND_FP,
               loc='upper center',
               ncol=1,
               bbox_to_anchor=(0.55, 1.3
                               ), shadow=False,
               columnspacing=0.1,
               frameon=True, borderaxespad=-1.5, handlelength=1.2,
               handletextpad=0.1,
               labelspacing=0.1)
    plt.yticks(rotation=0, size=TICK_FONT_SIZE)
    plt.tight_layout()

    plt.savefig(fname + ".pdf")


def cleanPath(path):
    os.system("sudo rm -rf " + path)
    os.system("sudo mkdir " + path)


def singleRun(exePath, scanTag, singleValue, resultPath, configTemplate, algo):
    # resultFolder="singleValueTests"
    configFname = "config_" + scanTag + str(singleValue) + ".csv"
    # configTemplate = "config.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")

    df = algo.config.copy()
    df.loc[scanTag] = [singleValue, 'U64']
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, exePath + configFname, df)
    # prepare new file
    # run
    os.system("cd " + exePath + "&& sudo env OMP_NUM_THREADS=1 ./benchmark " + configFname)
    # copy result
    cleanPath(resultPath + "/" + str(singleValue))
    cleanPath(resultPath + "/" + str(singleValue))
    os.system("cd " + exePath + "&& sudo cp *.csv " + resultPath + "/" + str(singleValue))


def runScanVector(exePath, scanTag, singleValueVec, resultPath, templateName, algo):
    for i in singleValueVec:
        singleRun(exePath, scanTag, i, resultPath, templateName, algo)


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/default.csv"
    elapsedTime = readConfig(resultFname, "perfElapsedTime")
    cacheMiss = readConfig(resultFname, "cacheMiss")
    cacheRefs = readConfig(resultFname, "cacheRefs")
    froError = readConfig(resultFname, "froError")
    errorBoundRatio = readConfig(resultFname, "errorBoundRatio")
    return elapsedTime, cacheMiss, cacheRefs, froError, errorBoundRatio


def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    cacheMissVec = []
    cacheRefVec = []
    froErrorVec = []
    errorBoundRatioVec = []
    for i in singleValueVec:
        elapsedTime, cacheMiss, cacheRefs, froError, errorBoundRatio = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        cacheMissVec.append(float(cacheMiss))
        cacheRefVec.append(float(cacheRefs))
        froErrorVec.append(float(froError))
        errorBoundRatioVec.append(float(errorBoundRatio))
    return np.array(elapseTimeVec), np.array(cacheMissVec), np.array(cacheRefVec), np.array(froErrorVec), np.array(
        errorBoundRatioVec)


def compareMethod(exeSpace, commonPathBase, scanTag, algos, csvTemplate, periodVec, reRun=1):
    elapsedTimeAll = []
    cacheMissAll = []
    cacheRefAll = []
    periodAll = []
    froAll = []
    errorBoundRatioAll = []
    for algo in algos:
        resultPath = commonPathBase + algo.resultPath
        if (reRun == 1):
            cleanPath(resultPath)
            runScanVector(exeSpace, scanTag, periodVec, resultPath, csvTemplate, algo)
        elapsedTime, cacheMiss, cacheRef, fro, eb = readResultVector(periodVec, resultPath)
        elapsedTimeAll.append(elapsedTime)
        cacheMissAll.append(cacheMiss)
        cacheRefAll.append(cacheRef)
        periodAll.append(periodVec)
        cacheMissRateAll = np.array(cacheMissAll) / np.array(cacheRefAll) * 100.0
        froAll.append(fro)
        errorBoundRatioAll.append(eb)
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), cacheMissRateAll, periodAll, np.array(froAll), np.array(errorBoundRatioAll)

def readResultSingleStreaming(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/result_streaming.csv"
    throughput = readConfig(resultFname, "throughputByElements")
    lat95 = readConfig(resultFname, "95%latency")
    froError = readConfig(resultFname, "froError")
    errorBoundRatio = readConfig(resultFname, "errorBoundRatio")
    return throughput, lat95, froError, errorBoundRatio


def readResultVectorStreaming(singleValueVec, resultPath):
    thrVec = []
    lat95Vec = []
    froErrorVec = []
    errorBoundRatioVec = []
    for i in singleValueVec:
        thr, lat95, froError, errorBoundRatio = readResultSingle(i, resultPath)
        thrVec.append(float(thr))
        lat95Vec.append(float(lat95) / 1000.0)
        froErrorVec.append(float(froError))
        errorBoundRatioVec.append(float(errorBoundRatio))
    return np.array(thrVec), np.array(lat95Vec), np.array(froErrorVec), np.array(
        errorBoundRatioVec)

def compareMethodStreaming(exeSpace, commonPathBase, scanTag, algos, csvTemplate, periodVec, reRun=1):
    thrAll = []
    lat95All = []
    periodAll = []
    froAll = []
    errorBoundRatioAll = []
    for algo in algos:
        resultPath = commonPathBase + algo.resultPath
        if (reRun == 1):
            cleanPath(resultPath)
            runScanVector(exeSpace, scanTag, periodVec, resultPath, csvTemplate, algo)
        thr, lat95, fro, eb = readResultVector(periodVec, resultPath)
        thrAll.append(thr)
        lat95All.append(lat95)

        periodAll.append(periodVec)

        froAll.append(fro)
        errorBoundRatioAll.append(eb)
        # periodAll.append(periodVec)
    return np.array(thrAll), np.array(lat95All), periodAll, np.array(froAll), np.array(errorBoundRatioAll)
