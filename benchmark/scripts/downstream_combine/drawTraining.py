import sys
import os
import numpy as np


from draw import main as run



def main():
    time, accuracy = run()
    time, accuracy =  [220*element for element in time], [(100 - element)/100 for element in accuracy]
    #print(time,accuracy)
    time[0] = time[0]/time[-2]*time[-1]
    return np.array(time), np.array(accuracy)
main()