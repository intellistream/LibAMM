import itertools as it
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, LinearLocator

OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 24
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 15
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

MARKERS = (["", 'o', 's', 'v', "^", "", "h", "<", ">", "+", "d", "<", "|", "", "+", "_"])
# you may want to change the color map for different figures
COLOR_MAP = (
    '#7FFFFF', '#B03A2E', '#2874A6', '#FFFFFF', '#7FFFFF', '#B03A2E', '#2874A6', '#FFFFFF', '#F5CBA7', '#82E0AA',
    '#AEB6BF',
    '#AA4499')
# you may want to change the patterns for different figures
PATTERNS = (
    ["////", "\\\\", "//", "o", "*", "||", "-", "//", "\\", "o", "O", "////", ".", "|||", "o", "---", "+", "\\\\", "*"])
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

exp_dir = "/data1/xtra"

FIGURE_FOLDER = exp_dir + '/results/figure'


def DrawLegend(legend_labels, filename):
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    FIGURE_LABEL = legend_labels
    LEGEND_FP = FontProperties(style='normal', size=26)
    figlegend = pylab.figure(figsize=(16, 0.5))
    bars = [None] * (len(FIGURE_LABEL))
    data = [1]
    x_values = [1]

    width = 0.3
    for i in range(len(FIGURE_LABEL)):
        bars[i] = ax1.bar(x_values, data, width,
                          hatch=PATTERNS[i],
                          color=LINE_COLORS[i],
                          label=FIGURE_LABEL[i],
                          edgecolor='black', linewidth=3)

    # LEGEND

    figlegend.legend(bars, FIGURE_LABEL, prop=LEGEND_FP, \
                     loc=1, ncol=len(FIGURE_LABEL), mode="expand", shadow=True, \
                     frameon=True, handlelength=2, handletextpad=0.3, columnspacing=0.5,
                     borderaxespad=-0.2, fancybox=True
                     )
    figlegend.savefig(FIGURE_FOLDER + '/' + filename + '.pdf')

def DrawBarPlot(x_values, y_values, legend_labels, x_label, y_label, filename):
    # Example usage:
    # x_values = ['A', 'B', 'C', 'D']
    # y_values = [10, 15, 20, 25]
    # legend_labels = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
    # x_label = 'X-axis Label'
    # y_label = 'Y-axis Label'
    # filename = 'bar_plot'
    # DrawBarPlot(x_values, y_values, legend_labels, x_label, y_label, filename)

    import matplotlib.colors as mcolors

    cmap = plt.cm.viridis

    # Draw the bars with different colors from the color map
    fig, ax = plt.subplots(figsize=(12, 12))
    width = 0.3

    # Draw the bars with different colors from the color map
    for i, (x_val, y_val, legend_label) in enumerate(zip(x_values, y_values, legend_labels)):
        color = mcolors.to_hex(cmap(i / len(legend_labels)))  # Convert RGBA to valid color string
        ax.bar(x_val, y_val, width, color=color, label=legend_label, linewidth=3)

    # plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=20)

    for bar in ax.containers:  # Loop through all the bars
        for rect in bar:  # Loop through all the rectangles (individual bars) in the container
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),  # Display y-value with 2 decimal digits
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=17)

    plt.tight_layout()
    plt.savefig(filename + "_nolegend.pdf", bbox_inches='tight')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.4, 0.6), fontsize=20)
    plt.savefig(filename + "_legend.pdf", bbox_inches='tight')
    plt.show()


def DrawGroupBarPlot(x_values, y_values_list, legend_labels, x_label, y_label, filename):

    # Example usage:
    # x_values = ['A', 'B', 'C', 'D']
    # y_values_list = [[10, 15, 20, 25], [12, 18, 22, 28], [8, 12, 16, 20]]  # Each sublist represents a different category
    # legend_labels = ['Category 1', 'Category 2', 'Category 3']
    # x_label = 'X-axis Label'
    # y_label = 'Y-axis Label'
    # filename = 'group_bar_plot'
    # DrawGroupBarPlot(x_values, y_values_list, legend_labels, x_label, y_label, filename)

    COLOR_MAP = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(20, 8))

    # The bar width for each group
    bar_width = 0.2

    # The spacing between each group of bars
    bar_spacing = 0.1

    # Calculate the number of bars in each group
    num_bars = len(y_values_list)
    num_groups = len(x_values)
    total_width = num_bars * bar_width + (num_bars - 1) * bar_spacing

    # Calculate the x positions for each group of bars
    x_positions = np.arange(num_groups) * total_width

    # Draw the bars for each category
    for i, (y_values, legend_label) in enumerate(zip(y_values_list, legend_labels)):
        color = COLOR_MAP(i % COLOR_MAP.N)  # Select color from the COLOR_MAP based on the index
        bars = ax.bar(x_positions + i * bar_width, y_values, bar_width, color=color, label=legend_label)

        # Display the value on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.2f}'.format(height),  # Display y-value with 2 decimal digits
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    # Set the x-axis tick positions and labels
    ax.set_xticks(x_positions + (total_width - bar_width) / 2)
    ax.set_xticklabels(x_values, fontsize=12)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename + ".pdf", bbox_inches='tight')
    plt.show()



def DrawFigure(x_values, y_values, legend_labels, x_label, y_label, y_min, y_max, filename, allow_legend):
    # you may change the figure size on your own.
    fig = plt.figure(figsize=(30, 10))
    figure = fig.add_subplot(111)

    FIGURE_LABEL = legend_labels

    # values in the x_xis
    index = np.arange(len(x_values))
    # the bar width.
    # you may need to tune it to get the best figure.
    width = 0.2
    # draw the bars
    bars = []
    ts = 0
    pos = 0
    gl = len(y_values[0])
    for i in range(len(y_values)):
        pos = pos + 3 * width
        for j in range(len(y_values[i])):
            pos = pos + width
            bar = plt.bar(pos, y_values[i][j], width, hatch=PATTERNS[j], color=LINE_COLORS[j], label=FIGURE_LABEL[j],
                          edgecolor='black', linewidth=3)
            bars.append(bar)
            ts = ts + 1

    # sometimes you may not want to draw legends.
    if allow_legend == True:
        plt.legend(bars, FIGURE_LABEL,
                   prop=LEGEND_FP,
                   ncol=4,
                   loc='upper center',
                   #                     mode='expand',
                   shadow=False,
                   bbox_to_anchor=(0.45, 1.6),
                   columnspacing=0.1,
                   handletextpad=0.2,
                   #                     bbox_transform=ax.transAxes,
                   #                     frameon=True,
                   #                     columnspacing=5.5,
                   #                     handlelength=2,
                   )

    # you may need to tune the xticks position to get the best figure.
    plt.xticks(index, x_values)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.grid(axis='y', color='gray')
    # figure.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # you may need to tune the xticks position to get the best figure.
    # plt.yscale('log')
    #
    # plt.grid(axis='y', color='gray')
    figure.yaxis.set_major_locator(LinearLocator(5))
    # figure.xaxis.set_major_locator(LinearLocator(5))
    figure.get_xaxis().set_tick_params(direction='in', pad=10)
    figure.get_yaxis().set_tick_params(direction='in', pad=10)

    plt.xlabel(x_label, fontproperties=LABEL_FP)
    plt.ylabel(y_label, fontproperties=LABEL_FP)
    plt.ylim(y_min, y_max)
    plt.savefig(filename + ".pdf", bbox_inches='tight')

# example for reading csv file
def ReadFile():
    y = []
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []
    col9 = []

    for id in it.chain(range(38, 42)):
        col9.append(0)
    y.append(col9)  # this is a lz4_pipe2 empty line to separate eager and lazy.

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/NPJ_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get the 99th timestamp
        col1.append(x)
    y.append(col1)

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/PRJ_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get the 99th timestamp        
        col2.append(x)
    y.append(col2)

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/MWAY_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get the 99th timestamp       
        col3.append(x)
    y.append(col3)

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/MPASS_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get the 99th timestamp
        col4.append(x)
    y.append(col4)

    y.append(col9)  # this is a lz4_pipe2 empty line to separate eager and lazy.

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/SHJ_JM_NP_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get last timestamp
        col5.append(x)
    y.append(col5)

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/SHJ_JBCR_NP_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get last timestamp
        col6.append(x)
    y.append(col6)

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/PMJ_JM_NP_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get last timestamp
        col7.append(x)
    y.append(col7)

    for id in it.chain(range(38, 42)):
        file = exp_dir + '/results/latency/PMJ_JBCR_NP_{}.txt'.format(id)
        f = open(file, "r")
        read = f.readlines()
        x = float(read.pop(int(len(read) * 0.95)).strip("\n"))  # get last timestamp
        col8.append(x)
    y.append(col8)
    return y


if __name__ == "__main__":
    x_values = ["Stock", "Rovio", "YSB", "DEBS"]

    y_values = ReadFile()

    legend_labels = ['Lazy:', 'NPJ', 'PRJ', 'MWAY', 'MPASS',
                     'Eager:', 'SHJ$^{JM}$', 'SHJ$^{JB}$', 'PMJ$^{JM}$', 'PMJ$^{JB}$']
    print(y_values)
    DrawFigure(x_values, y_values, legend_labels,
               '', 'Latency (ms)', 0,
               400, 'latency_figure_app', False)

    # DrawLegend(legend_labels, 'latency_legend')
