#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
import accuBar as accuBar
import groupBar2 as groupBar2
import groupLine as groupLine
from autoParase import *
import itertools as it
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib import ticker
from matplotlib.ticker import LogLocator, LinearLocator

import os
import pandas as pd
import sys
from OoOCommon import *

OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 22
LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
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

dataset_acols_mapping={
    'AST':765,
    'BUS':10595,
    'DWAVE':512,
    'ECO':260,
    'QCD':3072,
    'RDB':2048,
    'UTM':1700,
    'ZENIOS':2873,
}

def runPeriod(exePath, srcA,srcB, algoTag, resultPath, configTemplate="config.csv",prefixTag="null"):
    # resultFolder="periodTests"
    configFname = "config_period_"+prefixTag + ".csv"
    configTemplate = "config_e2e_static_lazy.csv"
    # clear old files
    os.system("cd " + exePath + "&& sudo rm *.csv")
    os.system("cp perfListEvaluation.csv " + exePath)
    # editConfig(configTemplate, exePath + configFname, "earlierEmitMs", 0)
    editConfig(configTemplate, exePath+"temp1.csv", "srcA", srcA)
    editConfig(exePath+"temp1.csv", exePath+"temp2.csv", "srcB", srcB)
    editConfig(exePath+"temp2.csv", exePath+"temp1.csv", "sketchDimension", int(dataset_acols_mapping[prefixTag]*0.1))
    editConfig(exePath+"temp1.csv",exePath+"temp2.csv", "cppAlgoTag", algoTag)

    # int8 or int8_fp32
    if algoTag=='int8_fp32':
        editConfig(exePath+"temp2.csv",exePath+"temp1.csv", "fpMode", "fp32")
    else:
        editConfig(exePath+"temp2.csv",exePath+"temp1.csv", "fpMode", "INT8")

    # load Codeword LookUpTable for vq or pq
    pqvqCodewordLookUpTableDir = f'{exePath}/torchscripts/VQ/AMME2E/CodewordLookUpTable'
    pqvqCodewordLookUpTablePath = "dummy"
    import glob
    if algoTag == 'vq':
        pqvqCodewordLookUpTablePath = glob.glob(f'{pqvqCodewordLookUpTableDir}/{prefixTag}_m1_*')[0]
    elif algoTag =='pq':
        pqvqCodewordLookUpTablePath = glob.glob(f'{pqvqCodewordLookUpTableDir}/{prefixTag}_m10_*')[0]
    editConfig(exePath+"temp1.csv",exePath+configFname, "pqvqCodewordLookUpTablePath", pqvqCodewordLookUpTablePath)

    # prepare new file
    # run
    os.system("export OMP_NUM_THREADS=1 &&" + "cd " + exePath + "&& sudo ./benchmark " + configFname)
    # copy result
    os.system("sudo rm -rf " + resultPath + "/" + str(prefixTag))
    os.system("sudo mkdir " + resultPath + "/" + str(prefixTag))
    os.system("cd " + exePath + "&& sudo cp *.csv " + resultPath + "/" + str(prefixTag))


def runPeriodVector (exePath,periodVec,pS,algoTag,resultPath,prefixTag, configTemplate="config.csv"):
    for i in  range(len(periodVec)):
        rf=periodVec[i]
        sf=pS[i]
        print(sf)
        runPeriod(exePath, rf,sf,algoTag, resultPath, configTemplate,prefixTag[i])


def readResultSingle(singleValue, resultPath):
    resultFname = resultPath + "/" + str(singleValue) + "/result_streaming.csv"
    elapsedTime = readConfig(resultFname, "perfElapsedTime")
    memLoad = readConfig(resultFname, "memLoad")
    memStore = readConfig(resultFname, "memStore")
    instructions = readConfig(resultFname, "instructions")
    fpVector = readConfig(resultFname, "fpVector")
    fpScalar = readConfig(resultFname, "fpScalar")
    branchIns = readConfig(resultFname, "branchIns")
    return elapsedTime, memLoad, memStore, instructions, fpVector, fpScalar, branchIns



def readResultVector(singleValueVec, resultPath):
    elapseTimeVec = []
    memLoadVec = []
    memStoreVec = []
    instructionsVec = []
    fpVectorVec = []
    fpScalarVec = []
    branchVec = []
    for i in singleValueVec:
        elapsedTime, memLoad, memStore, instructions, fpVector, fpScalar, branchIns = readResultSingle(i, resultPath)
        elapseTimeVec.append(float(elapsedTime) / 1000.0)
        memLoadVec.append(float(memLoad))
        memStoreVec.append(float(memStore))
        instructionsVec.append(float(instructions))
        fpVectorVec.append(float(fpVector) / 2)
        fpScalarVec.append(float(fpScalar) / 2)
        branchVec.append(float(branchIns))
    return np.array(elapseTimeVec), np.array(memLoadVec), np.array(memStoreVec), np.array(instructionsVec), np.array(
        fpVectorVec), np.array(fpScalarVec), np.array(branchVec)


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplate, srcAVec,srcBVec,algos,dataSetName,reRun=1):
    elapsedTimeAll = []
    memLoadAll = []
    memStoreAll = []
    periodAll = []
    instructionsAll = []
    fpVectorAll = []
    fpScalarAll = []
    branchAll = []
    for i in range(len(algos)):
        resultPath = commonPathBase + resultPaths[i]
        algoTag=algos[i]
        if (reRun == 1):
            os.system("sudo rm -rf " + resultPath)
            os.system("sudo mkdir " + resultPath)
            runPeriodVector(exeSpace, srcAVec,srcBVec,algoTag, resultPath, dataSetName,csvTemplate)
        #exit()
        elapsedTime, memLoad, memStore, instructions, fpVector, fpScalar, branchIns = readResultVector(dataSetName, resultPath)
        elapsedTimeAll.append(elapsedTime)
        memLoadAll.append(memLoad)
        memStoreAll.append(memStore)
        periodAll.append(elapsedTime)
        instructionsAll.append(instructions)
        fpVectorAll.append(fpVector)
        fpScalarAll.append(fpScalar)
        branchAll.append(branchIns)
        # periodAll.append(periodVec)
    return np.array(elapsedTimeAll), np.array(memLoadAll), np.array(periodAll), np.array(instructionsAll), np.array(
        memStoreAll), np.array(fpVectorAll), np.array(fpScalarAll), np.array(branchAll)
def getCyclesPerMethod(cyclesAll, valueChose):
    instructionsPerMethod = []
    for i in range(len(cyclesAll)):
        instructionsPerMethod.append(cyclesAll[int(i)][int(valueChose)])
    return np.array(instructionsPerMethod)       

def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/"
    commonBasePath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/results/instructions_breakdown_1/"

    figPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/figures/instructions_breakdown_1/"
    
    # add the datasets here
    # srcAVec=["datasets/AST/mcfe.mtx"] # 765*756
    # srcBVec=["datasets/AST/mcfe.mtx"] # 765*756
    # dataSetNames=['AST']
    # srcAVec=['datasets/UTM/utm1700a.mtx'] # 1700*1700
    # srcBVec=['datasets/UTM/utm1700b.mtx'] # 1700*1700
    # dataSetNames=['UTM']
    #srcAVec=['datasets/ECO/wm2.mtx',"datasets/DWAVE/dwa512.mtx","datasets/AST/mcfe.mtx",'datasets/UTM/utm1700a.mtx','datasets/RDB/rdb2048.mtx','datasets/ZENIOS/zenios.mtx','datasets/QCD/qcda_small.mtx',"datasets/BUS/gemat1.mtx",]
    #srcBVec=['datasets/ECO/wm3.mtx',"datasets/DWAVE/dwb512.mtx","datasets/AST/mcfe.mtx",'datasets/UTM/utm1700b.mtx','datasets/RDB/rdb2048l.mtx','datasets/ZENIOS/zenios.mtx','datasets/QCD/qcdb_small.mtx',"datasets/BUS/gemat1.mtx",]
    #dataSetNames=['ECO','DWAVE','AST','UTM','RDB','ZENIOS','QCD','BUS']
    srcAVec=['datasets/ECO/wm2.mtx','datasets/UTM/utm1700a.mtx','datasets/QCD/qcda_small.mtx']
    srcBVec=['datasets/ECO/wm3.mtx','datasets/UTM/utm1700b.mtx','datasets/QCD/qcdb_small.mtx']
    dataSetNames=['ECO','UTM','QCD']
    algosVec=[ 'crs','countSketch','mm']
    algoDisp=[ 'CRS','CS','LTMM']
    # add the algo tag here
    #algosVec=['mm', 'crs', 'countSketch', 'int8', 'weighted-cr', 'rip', 'smp-pca', 'tugOfWar', 'blockLRA', 'vq', 'pq', 'fastjlt', 'cooFD', 'int8_fp32']
    
    # this template configs all algos as lazy mode, all datasets are static and normalized
    csvTemplate = 'config_e2e_static_lazy.csv'
    # do not change the following
    resultPaths = algosVec

    # run
    reRun = 0
    if (len(sys.argv) < 2):
        os.system("mkdir ../../results")
        os.system("mkdir ../../figures")
        os.system("mkdir " + figPath)
        os.system("sudo rm -rf " + commonBasePath)
        os.system("sudo mkdir " + commonBasePath)
        reRun = 1
    methodTags =algoDisp
    elapsedTimeAll, memLoadAll, periodAll, instructions, memStoreAll, fpVectorAll, fpScalarAll, branchAll = compareMethod(exeSpace, commonBasePath, resultPaths, csvTemplate, srcAVec,srcBVec,algosVec,dataSetNames, reRun)
    # Add some pre-process logic for int8 here if it is used

    print(instructions, memLoadAll)
    otherIns = instructions - memLoadAll - memStoreAll - fpVectorAll - fpScalarAll - branchAll
    print(otherIns)
    print(otherIns[0], len(otherIns))
    allowLegend = 1
    valueVec=dataSetNames
    for valueChose in range(len(valueVec)):
        # instructionsPerMethod=getCyclesPerMethod(instructions,valueChose)
        memLoadPerMethod = getCyclesPerMethod(memLoadAll, valueChose)
        memStorePerMethod = getCyclesPerMethod(memStoreAll, valueChose)
        fpVectorPerMethod = getCyclesPerMethod(fpVectorAll, valueChose)
        fpScalarPerMethod = getCyclesPerMethod(fpScalarAll, valueChose)
        branchPerMethod = getCyclesPerMethod(branchAll, valueChose)
        otherPerMethod = getCyclesPerMethod(otherIns, valueChose)
        accuBar.DrawFigure(methodTags,
                           [memLoadPerMethod, memStorePerMethod, fpVectorPerMethod, fpScalarPerMethod, branchPerMethod,
                            otherPerMethod], ['load', 'store', 'fp-vector', 'fp-scalar', 'branch', 'others'], '',
                           'instructions', figPath + "/" + "insbreakDown"
                           + "_ins_accubar" + str(valueVec[valueChose]), allowLegend,
                           'dataset' + "=" + str(valueVec[valueChose]))
        allowLegend = 0
    #draw2yBar(methodTags,[lat95All[0][0],lat95All[1][0],lat95All[2][0],lat95All[3][0]],[errAll[0][0],errAll[1][0],errAll[2][0],errAll[3][0]],'95% latency (ms)','Error (%)',figPath + "sec6_5_stock_q1_normal")
    #groupBar2.DrawFigure(dataSetNames, errAll, methodTags, "Datasets", "Error (%)", 5, 15, figPath + "sec4_1_e2e_static_lazy_fro", True)
    #groupBar2.DrawFigure(dataSetNames, np.log(lat95All), methodTags, "Datasets", "95% latency (ms)", 5, 15, figPath + "sec4_1_e2e_static_lazy_latency_log", True)
    fpInsAll= fpVectorAll+fpScalarAll
    ratioFpIns=fpVectorAll/fpInsAll*100.0
    groupBar2.DrawFigureYLog(dataSetNames, instructions/instructions[-1], methodTags, "Datasets", "Ins (times of LTMM)", 5, 15, figPath + "/" + "instructions", True)
    groupBar2.DrawFigure(dataSetNames, ratioFpIns, methodTags, "Datasets", "SIMD Utilization (%)", 5, 15, figPath + "/" + "SIMD utilization", False)
    groupBar2.DrawFigure(dataSetNames, instructions/(memLoadAll+memStoreAll), methodTags, "Datasets", "IPM", 5, 15, figPath + "/" + "IPM", False)
    groupBar2.DrawFigure(dataSetNames, fpInsAll/(memLoadAll+memStoreAll), methodTags, "Datasets", "FP Ins per Unit Mem Access", 5, 15, figPath + "/" + "FPIPM", False)
    groupBar2.DrawFigure(dataSetNames, (memLoadAll+memStoreAll)/(instructions)*100.0, methodTags, "Datasets", "Ratio of Mem Ins (%)", 5, 15, figPath + "/" + "mem", False)
   
    groupBar2.DrawFigure(dataSetNames, branchAll/instructions*100.0, methodTags, "Datasets", "Ratio of Branch Ins (%)", 5, 15, figPath + "/" + "branches", False)
    groupBar2.DrawFigure(dataSetNames, otherIns/instructions*100.0, methodTags, "Datasets", "Ratio of Other Ins (%)", 5, 15, figPath + "/" + "others", False)
    print(instructions[-1],instructions[2])
    #groupBar2.DrawFigure(dataSetNames, np.log(thrAll), methodTags, "Datasets", "elements/ms", 5, 15, figPath + "sec4_1_e2e_static_lazy_throughput_log", True)
if __name__ == "__main__":
    main()
