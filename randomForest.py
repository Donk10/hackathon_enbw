#!/usr/bin/python3

import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
from loadData import loadWeatherData
from loadTourism import arrayTourism
from loadHolidays import loadHolidays

import matplotlib.pyplot as plt
'''
def loadTrafficData(weekday, hour):
    data_df = pd.read_csv("./data/pandas_wide_extended.csv")
    dataForWeekday_df = data_df.loc[data_df['weekday'] == weekday]
    dataForWeekdayAndHour_df = dataForWeekday_df.loc[dataForWeekday_df['h'] == hour]
    return dataForWeekdayAndHour_df

def getMeanAndStd(weekday, hour, indexStr):
    data_df = pd.read_csv("./data/pandas_wide_extended.csv")
    dataForWeekday_df = data_df.loc[data_df['weekday'] == weekday]
    dataForWeekdayAndHour_df = dataForWeekday_df.loc[dataForWeekday_df['h'] == hour]
    return np.mean(dataForWeekdayAndHour_df[indexStr].values), \
            np.std(dataForWeekdayAndHour_df[indexStr].values)

def getValues(weekday, hour, indexStr, data_df):
    dataForWeekday_df = data_df.loc[data_df['weekday'] == weekday]
    dataForWeekdayAndHour_df = dataForWeekday_df.loc[dataForWeekday_df['h'] == hour]
    return dataForWeekdayAndHour_df

print(getMeanAndStd(2,10,'2'))

def predictAt(datumStr, hour):
    data_df = pd.read_csv("./data/pandas_wide_extended.csv")
    dataAtDate = data_df.loc[data_df['datum'] == datumStr]
    dataPoint = dataAtDate.loc[dataAtDate['h'] == hour]
    print(dataPoint)

predictAt('2017-12-01', 12)
'''





'''
def loadTraffic():

    data_df = pd.read_csv("./data/pandas_wide.csv", sep=",")

    print(data_df.head())
    data_df = data_df.sort_values(['weekday'])
    data = data_df.values
    #train, test, _, _ = train_test_split(np.linspace(0,len(data)-1,len(data),dtype=int),
    #                    np.zeros((len(data))), test_size=0.20, random_state=42)

    return data_df.values


print(loadTraffic()[:48,-1])


data = loadTraffic()
plt.figure()
plt.plot(data[-720:,0],data[-720:,5])
plt.plot(data[-720:,0],data[-720:,-1])
plt.show()
'''

def appendCsv():
    data_df = pd.read_csv("./data/pandas_wide_extended.csv", sep=",")
    #print(data_df.head())

    data_df['Unnamed: 0'] = data_df['Unnamed: 0.1.1']
    #data_df = data_df.drop(['Unnamed: 0'], axis=1)
    data_df = data_df.drop(['Unnamed: 0.1'], axis=1)
    data_df = data_df.drop(['Unnamed: 0.1.1'], axis=1)

    count_df = data_df['count']
    humidity_df = data_df['humidity']
    temperature_df = data_df['temperature']
    tourism_df = data_df['tourism']
    weekday_df = data_df['weekday']
    prozent1_df = data_df['prozent1']
    prozent2_df = data_df['prozent2']
    prozent3_df = data_df['prozent3']
    prozent4_df = data_df['prozent4']
    holidays_df = data_df['holidays']

    data_df = data_df.drop(['y', 'd', 'h', '1', '2', '3', '4', 'prozent1'
                            ,'prozent2', 'prozent3', 'prozent4','count'
                            , 'temperature', 'humidity', 'tourism', 'holidays'
                            , 'datum', 'weekday'], axis=1)

    data_df['weekday'] = weekday_df
    data_df['humidity'] = humidity_df
    data_df['temperature'] = temperature_df
    data_df['holidays'] = holidays_df
    data_df['tourism'] = tourism_df
    data_df['count'] = count_df
    data_df['prozent1'] = prozent1_df
    data_df['prozent2'] = prozent2_df
    data_df['prozent3'] = prozent3_df
    data_df['prozent4'] = prozent4_df

    data_df['count'] = count_df

    data_np = data_df.values
    np.save('./data/pandas_wide_extended.npy', data_np)
'''
    # append data_df by humidity values
    data_df['humidity'] = loadWeatherData("./data/humidityData.txt")

    # append with temperature
    data_df['temperature'] = loadWeatherData("./data/temperatureData.txt")

    # append with tourism
    data_df['tourism'] = arrayTourism()

    # append with holidays
    data_df['holidays'] = loadHolidays()

    #data_df.to_csv('./data/pandas_wide_extended.csv')
'''
appendCsv()
