#!/usr/bin/python3

import numpy as np

def loadRainData():
    # load file with cloud data from
    # Neckargemuend
    file_in_npArray = np.loadtxt("./data/R2_MN008.txt", dtype=str, delimiter=";")

    # store data and description in 2 arrays
    description = file_in_npArray[0,1:4]
    rainValue_mm = np.array(file_in_npArray[1:,2:4], dtype=float)

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
        # newValueArray[i] = rainValue_mm[np.where(rainValue_mm[:,0] == timeStamps[i])[0][0],1]
        index = np.where(rainValue_mm[:,0] == int(timeStamps[i]))[0]
        if index.size > 0:
            newValueArray[i] = rainValue_mm[np.where(rainValue_mm[:,0] == int(timeStamps[i]))[0][0],1]
        if index.size == 0:
            newValueArray[i] = newValueArray[i-1]
    return newValueArray

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


def getRainValue_in_mm(year, month, day, time, rainArray):
    # create timeStamp like "yyyymmddtttt"
    timeStamp = year*10**8 + month*10**6 + day*10**4 + time*10**2
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
# rainValue_mm = loadRainData()
# print(getRainValue_in_mm(2017, 12, 4, 5, rainValue_mm))

# Save rain data into .txt file
# np.savetxt("rainValues.txt", rainValue_mm)
'''
