#!/usr/bin/python3

from loadData import loadWeatherData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from IPython import display

humidity = loadWeatherData("./data/humidityData.txt")
temperature = loadWeatherData("./data/temperatureData.txt")
rain = loadWeatherData("./data/R2_MN008.txt")
cloud = loadWeatherData("./data/N_MN008.txt")


humidityNormed = (humidity-np.mean(humidity))/np.std(humidity)
temperatureNormed = (temperature-np.mean(temperature))/np.std(temperature)
rainNormed = (rain-np.mean(rain))/np.std(rain)
cloudNormed = (cloud-np.mean(cloud))/np.std(cloud)

'''
weatherArray = np.concatenate((np.array([humidityNormed]),
                                np.array([temperatureNormed]),
                                np.array([rainNormed]),
                                np.array([cloudNormed])), 0)
'''
df = pd.DataFrame()

df['humidity'] = humidityNormed
df['temperature'] = temperatureNormed
df['rain'] = rainNormed
df['cloud'] = cloudNormed
'''
print(df.corr())
f,ax = plt.subplots()
sb.heatmap(df.corr())
plt.show()
'''

plt.scatter(humidity, temperature, c=rain, cmap=plt.cm.autumn)
plt.show()
