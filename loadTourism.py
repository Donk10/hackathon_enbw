import numpy as np
import pandas as pd
from IPython import display
from calendar import monthrange


#retruns array with length numberOfDays total number of tourism in "month"
#"month" [1-9]: 1 is 12.2017, 9 is 08.2018
#"numberOfDays": number of days in "month"
def loadToursim(month, numberOfDays):
    #load file
    csvTrouismFile            = "./data/tourism.csv"
    csvTrouismFile_DataFrame  = pd.read_csv(csvTrouismFile, sep=";")
    csvTrouismFile_NumpyArray = csvTrouismFile_DataFrame.values[3:]

    #get month
    data = csvTrouismFile_NumpyArray[month-1]
    #array with numberOfDays data
    monthValue = np.full(numberOfDays, csvTrouismFile_NumpyArray[month-1][1], dtype=float)

    return monthValue

#returns numpy array with hourly tourism data of size 5850
def arrayTourism():

    csvTrouismFile            = "./data/tourism.csv"
    csvTrouismFile_DataFrame  = pd.read_csv(csvTrouismFile, sep=";")
    monthlyArray              = csvTrouismFile_DataFrame.values[3:]

    tourismHourly = np.array([], dtype=float)
    for m in range(np.shape(monthlyArray)[0]):
        month, year, tourismValue = int(monthlyArray[m,0]), int(monthlyArray[m,1]), float(monthlyArray[m,2])
        if m == (np.shape(monthlyArray)[0]-1):
            tourismHourly = np.append(tourismHourly, np.full(23, tourismValue))
        else:
            days          = monthrange(year, month)[1]
            tourismHourly = np.append(tourismHourly, np.full(24*days, tourismValue))
    return tourismHourly
