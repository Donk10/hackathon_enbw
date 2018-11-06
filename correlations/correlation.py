import numpy as np
import pandas as pd
import seaborn           as sns
import matplotlib.pyplot as plt

from loadData import loadWeatherData
from loadHolidays import loadHolidays

def normalize(array): return (array-np.mean(array))/np.std(array)


#load
humidity = loadWeatherData("./data/humidityData.txt")
features = "./data/features_normalized.csv"
f        = pd.read_csv(features, sep=",")

f["Humidity"] = pd.Series(normalize(humidity))
f["Holiday"]  = pd.Series(normalize(loadHolidays()))

"""
csvTrouismFile            = "./data/tourism.csv"
csvTrouismFile_DataFrame  = pd.read_csv(csvTrouismFile, sep=";")
tourism                   = normalize(csvTrouismFile_DataFrame.iloc[:,2])

print(tourism)
"""

#get weekends
weekend  = f

#[(f["weekday"]>6)&(f["m"]>3)]

#print(weekend)
#corr motorrad with weather
motorBike = weekend["1"]
rain      = weekend["Rain"]
cloud     = weekend["Cloud"]

bike     = f["prozent1"]
pkwKlein = f["prozent2"]
pkwGroß  = f["prozent3"]
LKW      = f["prozent4"]

# feature map
bikeWeather = weekend[["1", "2", "3", "4", "count", "prozent1", "prozent2", "prozent3", "prozent4", "Tourism", "Holiday"]]

print(bikeWeather)

#print(bikeWeather.corr())
print(bikeWeather.corr())
f, ax = plt.subplots()
sns.heatmap(bikeWeather.corr())
sns.pairplot(bikeWeather.corr())
plt.show()
