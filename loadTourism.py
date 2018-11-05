import numpy as np
import pandas as pd
from IPython import display


#retruns array with length numberOfDays total number of tourism in "month"
#"month" [1-9]: 1 is 12.2017, 9 is 08.2018
#"numberOfDays": number of days in "month"
#
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
