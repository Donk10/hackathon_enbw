#!/usr/bin/python3

import numpy as np
import pandas as pd
from IPython import display

# import sys to get arguments from command line
import sys

csvFileName = "../data/schoenau_2017-2018.csv"

# read data from csv file
csvFile_In_DataFrame = pd.read_csv(csvFileName, sep=",")
# put data into numpy array
csvFile_In_NumpyArray = csvFile_In_DataFrame.values

# number of measurements
numberOfMeasurements = csvFile_In_NumpyArray[:,0].shape[0]

# value classification moto = 1., pkw = 2., bus = 3., lkw = 4.
typeOfVehicle = 2.

# count number of vehicles
typeCount = np.zeros(4)
for i in range(4):
    # i+1 gives the type of vehicle
    typeCount[i] = np.count_nonzero(
        (csvFile_In_NumpyArray[:,2]-(i+1)*np.ones(numberOfMeasurements))==0
        )

#print(typeCount, np.sum(typeCount), numberOfMeasurements)

#print(csvFile_In_DataFrame.head())
print(csvFile_In_DataFrame["value.timeStamp"].unique())
