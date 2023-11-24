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
    ammE = [1.14356545566352e-08, 1.109919834136963, 0.8704940795898437, 0.06829112768173218, 0.018635477125644683, 0.9807984352111816, 0, 0, 1.0115696907043457, 0.8641192436218261, 2.496437644958496, 0.6275016307830811, 1.14356545566352e-08, 0.0]
    ammE2 = [2.0378884357796778e-08, 1.8020999908447266, 2.0562496185302734, 0.13076682090759278, 0.02833651900291443, 2.5514612197875977, 0, 0, 1.9445787429809571, 1.986338996887207, 5.2983863830566404, 1.6422035217285156, 2.0378884357796778e-08, 0]
    return np.array(time), np.array(accuracy), np.array(ammE), np.array(time2), np.array(accuracy2), np.array(ammE2)
main()
