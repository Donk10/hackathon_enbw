#!/usr/bin/python3

import numpy as np

# load file with cloud data
file_in_npArray = np.loadtxt("./data/R1_MN008.txt", dtype=str, delimiter=";")

# store data and description in 2 arrays
description = file_in_npArray[0,1:4]
rainValue_mm = np.array(file_in_npArray[1:,2:4], dtype=float)

# get only relevant timeframe 01.12.2017 to 01.08.2018
rainValue_mm = rainValue_mm[:np.where(rainValue_mm[:,0] == 201808012300)[0][0]]


'''
Function parameters, intervalls, integer:
    time = 0; for hour 0 to 1
    time = 1; for hour 1 to 2
    time = 0...23

    day = 1...31

    month = 1...12

    year = 2017...2018

    -> as integers

    rainArray
    -> numpy array with timeStamps and corresponding values

Returns:
    If timestamp not found: None
    Else: precipitation ammount in mm (Niederschlag)
'''

def getRainValue_in_mm(year, month, day, time, rainArray):
    # write input into strings suitable for merging into timeStamp
    yearStr = str(year)
    monthStr = ("0"+str(month))[-2:]
    dayStr = ("0"+str(day))[-2:]
    timeStr = ("0"+str(time)+"00")[-4:]
    # merge into timeStamp
    timeStamp = int(yearStr + monthStr + dayStr + timeStr)
    # index of cloudValueForStation, where timeStamp is
    index = np.where(rainArray[:,0] == timeStamp)

    # check if element is in cloudValueForStation-Array
    if index[0].size == 0:
        return None
    else:
        index = index[0][0]
        # cloud value at requested time
        return rainArray[index,1]

# EXAMPLE: how to get values
print(getRainValue_in_mm(2017, 12, 4, 5, rainValue_mm))
