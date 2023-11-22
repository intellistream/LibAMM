import sys
import os
import numpy as np


from draw import main as run



def main():
    time, accuracy, time2, accuracy2= run()
    time, accuracy =  [50*(element-time[-1]*0.78) for element in time], [(100 - element)/100 for element in accuracy]
    time2, accuracy2 =  [50*(element-time2[-1]*0.78) for element in time2], [(100 - element)/100 for element in accuracy2]
    #print(time,accuracy)
    time[0] = time[0]/time[-2]*time[-1]
    time2[0] = time2[0]/time2[-2]*time2[-1]
    print(time, accuracy, time2, accuracy2)
    return np.array(time), np.array(accuracy), np.array(time2), np.array(accuracy2)
main()
