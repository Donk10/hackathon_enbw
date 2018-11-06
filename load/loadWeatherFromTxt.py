
import numpy    as  np

'''
    loadWeatherData - Function

    loads weather data (downloaded from 'http//:cdc.dwd.de') out of .txt file

    returns 1-dim numpy array for relevant timeframe of our provided traffic data
'''


def loadWeatherData(filePath):
    # load .txt file with weather data (humidity, temperature, ...) as numpy array
    file_in_npArray = np.loadtxt(filePath, dtype=str, delimiter=";")

    # store data in array
    data = np.array(file_in_npArray[1:,2:4], dtype=float)




    # create time stamp array for relevant time frame
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




    # get data for given time frame
    newValueArray = np.zeros_like(timeStamps)
    for i in range(timeStamps.shape[0]):
        # search index of timeStamp in data
        index = np.where(data[:,0] == int(timeStamps[i]))[0]
        if index.size > 0:
            newValueArray[i] = data[np.where(data[:,0] == int(timeStamps[i]))[0][0],1]
        # if measurement at given time point is missing, take last measurement
        if index.size == 0:
            newValueArray[i] = newValueArray[i-1]
    # Return waether data of relevant time frame in 1-dim array
    return newValueArray
