#!/usr/bin/python3

import numpy as np

def loadCloudData():
    # load file with cloud data
    file_in_npArray = np.loadtxt("./data/N_MN008.txt", dtype=str, delimiter=";")

    # store data and description in array
    data = np.array(file_in_npArray[1:,1:4], dtype=int)

    # Numbers of weather stations
    #weatherStation = 13674 # Waibstadt
    weatherStation = 5906 # Mannheim

    # Get weather for mentioned Station
    cloudValueForStation = data[np.where(data == weatherStation)[0][0]:
                               np.where(data == weatherStation)[0][-1]]

    daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    months = np.arange(1,13,1, dtype=int)
    hours = np.arange(0,24,1, dtype=int)

    timeStamps = np.array([])

    start2017 = 201712010000
    end2017 = 201712312300
    start2018 = 201801010000
    end2018 = 201808012200

    for year in [2017,2018]:
        for month in months:
            for day in range(1,1+daysInMonth[month-1]):
                for hour in hours:
                    temporaryTimeStamp = year*10**8 + month*10**6 + day*10**4 + hour*10**2
                    if temporaryTimeStamp >= start2017 and temporaryTimeStamp <= end2017:
                        timeStamps = np.append(timeStamps, temporaryTimeStamp)
                    if temporaryTimeStamp >= start2018 and temporaryTimeStamp <= end2018:
                        timeStamps = np.append(timeStamps, temporaryTimeStamp)

    newValueArray = np.zeros_like(timeStamps)
    for i in range(timeStamps.shape[0]):
        index = np.where(cloudValueForStation[:,1] == int(timeStamps[i]))[0]
        if index.size > 0:
            newValueArray[i] = cloudValueForStation[np.where(cloudValueForStation[:,1] == int(timeStamps[i]))[0][0],2]
            if newValueArray[i] == -1:
                newValueArray[i] = 9.
        if index.size == 0:
            newValueArray[i] = newValueArray[i-1]
    return newValueArray

print(loadCloudData())


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
# cloudArray = loadCloudData()
# print(getCloudValue(2017, 12, 4, 5, cloudArray))

# Save cloud data for specified weather station into .txt file
# np.savetxt("cloudValues.txt", cloudValueForStation)
