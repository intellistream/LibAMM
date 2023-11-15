import sys
import os



from draw import main as run


def main():
    time, accuracy = run()
    time, accuracy =  [220*element for element in time], [(100 - element)/100 for element in accuracy]
    time[0] = time[0]/time[12]*time[13]
    print(time,accuracy)
    return time, accuracy

main()
