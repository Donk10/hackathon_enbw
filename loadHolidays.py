import numpy as np
import pandas as pd
from IPython import display
from calendar import monthrange

def loadHolidays():
    # load holidays
    holidaysCsv  = "./data/holidays.csv"
    holidaysCsv_DataFrame  = pd.read_csv(holidaysCsv, sep=";")
    holidaysNumpy = holidaysCsv_DataFrame.values
    #load traffic for reference
    trafficCsv           = "./data/pandas_wide.csv"
    trafficCsv_DataFrame = pd.read_csv(trafficCsv, sep=",")
    trafficNumpy         = trafficCsv_DataFrame.values

    holidays = np.zeros(np.shape(trafficNumpy)[0])

    for h in range(np.shape(holidaysNumpy)[0]):
        for d in range(np.shape(trafficNumpy)[0]):
            yearH, monthH, dayH = holidaysNumpy[h,0], holidaysNumpy[h,1],  holidaysNumpy[h,2]
            yearT, monthT, dayT = trafficNumpy[d,1], trafficNumpy[d,2], trafficNumpy[d,3]
            if ((yearH == yearT) & (monthH == monthT) & (dayH == dayT)):
                holidays[d] = 1
            else: continue

    return holidays


loadHolidays()
