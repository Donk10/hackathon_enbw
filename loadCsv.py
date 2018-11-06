#!/usr/bin/python3

import numpy as np
import pandas as pd

'''
 returns traffic data in numpy array like:
 [[station index, time stamp 1, value]
  [station index, time stamp 2, value]
  [...] ]
'''
def loadTraffic():
    csvFileName = "./data/schoenau_2017-2018.csv"

    # read data from csv file
    csvFile_In_DataFrame = pd.read_csv(csvFileName, sep=",")
    # put data into numpy array
    csvFile_In_NumpyArray = csvFile_In_DataFrame.values
    # return numpy array with data
    return csvFile_In_NumpyArray
