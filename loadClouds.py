#!/usr/bin/python3

import numpy as np

# load file with cloud data
file_in_npArray = np.loadtxt("./data/N_MN008.txt", dtype=str, delimiter=";")

# store data and description in 2 arrays
description = file_in_npArray[0,1:4]
data = np.array(file_in_npArray[1:,1:4], dtype=int)

# Numbers of weather stations
#weatherStation = 13674 # Waibstadt
weatherStation = 5906 # Mannheim

# Get weather for mentioned Station
cloudValueForStation = data[np.where(data == weatherStation)[0][0]:
                           np.where(data == weatherStation)[0][-1]]
# Only get data for relevant time frame from 01.12.2017 to 01.08.2018
cloudValueForStation = cloudValueForStation[
                        np.where(cloudValueForStation == 201712010000)[0][0]
                        :np.where(cloudValueForStation == 201808020000)[0][0]
                        ]

'''
Function parameters, intervalls, integer:
    time = 0; for hour 0 to 1
    time = 1; for hour 1 to 2
    time = 0...23

    day = 1...31

    month = 1...12

    year = 2017...2018

    -> as integers

    cloudArray
    -> numpy array with timeStamps and corresponding values

Returns:
    None, if timestamp not found
    -1, for sky not visible
    0...8, sunny to cloudy
'''
def getCloudValue(year, month, day, time, cloudArray):
    # write input into strings suitable for merging into timeStamp
    yearStr = str(year)
    monthStr = ("0"+str(month))[-2:]
    dayStr = ("0"+str(day))[-2:]
    timeStr = ("0"+str(time)+"00")[-4:]
    # merge into timeStamp
    timeStamp = int(yearStr + monthStr + dayStr + timeStr)
    # index of cloudArray, where timeStamp is
    index = np.where(cloudArray == timeStamp)

    # check if element is in cloudArray-Array
    if index[0].size == 0:
        return None
    else:
        index = index[0][0]
        # cloud value at requested time
        return cloudArray[index,2]


# EXAMPLE: how to get values
print(getCloudValue(2017, 12, 4, 5, cloudValueForStation))
