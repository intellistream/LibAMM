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
from os.path import join
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


streaming=0

if streaming==0:
    scan_dictionary = {
        'scanPara': "threads",
        'paras':{
            'cppAlgoTag': ['mm', 'crs', 'weighted-cr', 'countSketch', 'tugOfWar', 'smp-pca', 'rip', 'fastjlt', 'cooFD', 'blockLRA', 'int8', 'pq-raw', 'vq'],
            'sketchDimension': 6000,
            'coreBind': 0,
            'threads': [1],
            'matrixLoaderTag': 'MNIST',
            'isStreaming': 0,
            'streamingTwoMatrices': 0,
        },
        'plot':{ # what needs to be plotted from results.csv
            'Sxx Fro Error': 'SxxFroError',
            'Sxy Fro Error': 'SxyFroError',
            'Syy Fro Error': 'SyyFroError',
            'AMMThroughput (rows per second)': 'AMMThroughput', 
            'Correlation Error': 'CorrelationError',
        },
        'rounds':1,
    }
    scanTag = "cca_nonstreaming"
# else:
#     scan_dictionary = {
#         'scanPara': "threads",
#         'paras':{
#             'cppAlgoTag': ['pq-hash', 'mm', 'crs', 'countSketch', 'tugOfWar', 'smp-pca', 'rip'], # streaming
#             'sketchDimension': 1000,
#             'coreBind': 0,
#             'threads': [1],
#             'matrixLoaderTag': 'MNIST',
#             'isStreaming': 1,
#             'streamingTwoMatrices': 1,
#         },
#         'plot':{
#             'AMM Fro Error': 'AMMFroError', # key shown in figure, value is from results.csv
#             'AMMThroughput (rows per second)': 'AMMThroughput', # throughput = aRows * 1e6 / tDone (us);
#             'Correlation Error': 'CorrelationError'
#             },
#         'rounds':1,
#     }
#     scanTag = "cca_streamingAB"


def singleRun(exePath, singleValue, resultPath, configTemplate):
    # where config.csv and result.csv is saved
    resultPath = join(resultPath, str(singleValue))
    Path(resultPath).mkdir(parents=True, exist_ok=True)

    # config_update_dict
    config_update_dict = scan_dictionary['paras'].copy()
    del config_update_dict['cppAlgoTag']
    config_update_dict[scan_dictionary['scanPara']] = singleValue
    assert config_update_dict['sketchDimension'] > 10

    # udpate config
    df = pd.read_csv(configTemplate)
    for key, value in config_update_dict.items(): df.loc[df['key'] == key, 'value'] = value
    # (Pdb) df
    #                 key                                              value    type
    # 3   matrixLoaderTag                                               MNIST  String
    # 4   sketchDimension                                                100     U64
    # 5          coreBind                                                  1     U64
    # 6           threads                                                  1     U64
    # 7            useCPP                                                  1     U64
    # 8           forceMP                                                  1     U64
    # 9        cppAlgoTag                                                 mm  String
    # 10       usingMeter                                                  1     U64
    df.to_csv(join(resultPath, 'config.csv'), index=False)

    # run benchmark to generate result.csv
    os.system(f"cd {resultPath} && sudo env OMP_NUM_THREADS=1 {exePath}/benchmarkCCA config.csv")
    

def runScanVector(exePath, singleValueVec, resultPath, templateName="config.csv"):
    # (Pdb) exePath, singleValueVec, resultPath, templateName
    # ('/home/heyuhao/AMMBench/build/benchmark/scripts/MM/', [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000], '/home/heyuhao/AMMBench/build/benchmark/scripts/MM/results/scanAaCol_aRow1000_bCol1000_sketch0.1_datasetrandommm', 'config_mm.csv')
    for i in singleValueVec:
        singleRun(exePath, i, resultPath, templateName)


def readResultSingle(singleValue, resultPath):
    resultFname = join(resultPath, str(singleValue), "CCA.csv")
    return {k:readConfig(resultFname, scan_dictionary['plot'][k]) for k in scan_dictionary['plot']}


def readResultVector(singleValueVec, resultPath):
    plot_results = {k:[] for k in scan_dictionary['plot']}
    for i in singleValueVec:
        results = readResultSingle(i, resultPath)
        for k in plot_results:
            plot_results[k].append(results[k])
    return {k: np.array(v, dtype=float) for k,v in plot_results.items()}


def compareMethod(exeSpace, commonPathBase, resultPaths, csvTemplates, periodVec, reRun=1):
    # (Pdb) exeSpace, commonPathBase, resultPaths, csvTemplates, periodVec
    # ('/home/heyuhao/AMMBench/build/benchmark/scripts/MM/', '/home/heyuhao/AMMBench/build/benchmark/scripts/MM/results/scanAaCol_aRow1000_bCol1000_sketch0.1_datasetrandom', ['mm', 'crs', 'tugOfWar', 'cooFD', 'smp-pca'], ['config_mm.csv', 'config_crs.csv', 'config_tugOfWar.csv', 'config_cooFD.csv', 'config_smp-pca.csv'], [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    
    plot_results = {k:[] for k in scan_dictionary['plot']}

    for i in range(len(csvTemplates)):
        resultPath = join(commonPathBase, resultPaths[i])
        # runScanVector(exeSpace, periodVec, resultPath, csvTemplates[i])
        results = readResultVector(periodVec, resultPath)
        for k in plot_results:
            plot_results[k].append(results[k])
    return {k: np.array(v) for k,v in plot_results.items()}
    # (Pdb) {k: np.array(v) for k,v in plot_results.items()}
    # {'AMM Error': array([['0.000000', '0.000000'],
    #     ['0.286175', '0.199001']], dtype='<U8'), 'AMM Elapsed Time': array([['79760', '70267'],
    #     ['75051', '65826']], dtype='<U5')}


def main():
    exeSpace = os.path.abspath(os.path.join(os.getcwd(), ".."))
    resultPath = join(exeSpace, 'results', scanTag)
    figPath = join(exeSpace, 'figures', scanTag)
    
    # clear
    # if os.path.exists(resultPath): shutil.rmtree(resultPath)
    # if os.path.exists(figPath): shutil.rmtree(figPath)
    Path(resultPath).mkdir(parents=True, exist_ok=True)
    Path(figPath).mkdir(parents=True, exist_ok=True)

    # define some variables for runing benchmark
    commonBase = resultPath
    resultPaths = scan_dictionary['paras']['cppAlgoTag']
    evaTypes = resultPaths
    csvTemplates = ["config_"+i+".csv" for i in resultPaths]
    valueVec = scan_dictionary['paras'][scan_dictionary['scanPara']]
    periodAll = [valueVec]*len(csvTemplates)

    # metrics need to be collected
    metrics_kwargs = {key: np.zeros((len(resultPaths), len(valueVec))) for key in scan_dictionary['plot'].keys()}
    for i in range(scan_dictionary['rounds']):
        metrics_update = compareMethod(exeSpace, commonBase, resultPaths, csvTemplates, valueVec)
        for key in metrics_kwargs:
            metrics_kwargs[key] += metrics_update[key]
    metrics_kwargs = {key: value/scan_dictionary['rounds'] for key, value in metrics_kwargs.items()}

    ## plot
    # draw seperate pdfs
    for key in metrics_kwargs.keys():
        groupBar.DrawBarPlot(
            x_values=[i for i in range(len(scan_dictionary['paras']['cppAlgoTag']))],
            y_values=metrics_kwargs[key].flatten(),
            legend_labels=[f'{i} {v}' for i,v in enumerate(scan_dictionary['paras']['cppAlgoTag'])],
            x_label=[i for i in range(len(scan_dictionary['paras']['cppAlgoTag']))], 
            y_label=key,
            filename= figPath + "/" + key)

    # manually add other metrics
    # standardize to make mm as 1

    mm_index = scan_dictionary['paras']['cppAlgoTag'].index('mm')
    
    groupBar.DrawBarPlot(
        x_values=[i for i in range(len(scan_dictionary['paras']['cppAlgoTag']))],
        y_values=1/metrics_kwargs['AMMThroughput (rows per second)'].flatten()*metrics_kwargs['AMMThroughput (rows per second)'][mm_index],
        legend_labels=[f'{i} {v}' for i,v in enumerate(scan_dictionary['paras']['cppAlgoTag'])],
        x_label=[i for i in range(len(scan_dictionary['paras']['cppAlgoTag']))], 
        y_label=f"AMM Latency (second/{int(metrics_kwargs['AMMThroughput (rows per second)'][mm_index])}rows)",
        filename= figPath + "/" + "AMM Latency")
    

if __name__ == "__main__":
    main()
