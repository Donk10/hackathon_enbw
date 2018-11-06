import numpy as np
import pandas as pd
import seaborn           as sns
import matplotlib.pyplot as plt

from loadData import loadWeatherData

def normalize(array): return (array-np.mean(array))/np.std(array)


#load
humidity = loadWeatherData("./data/humidityData.txt")
features = "./data/features_normalized.csv"
f        = pd.read_csv(features, sep=",")

f["Humidity"] = pd.Series(normalize(humidity))

#get weekends
weekend  = f[(f["weekday"]>6)&(f["m"]>3)]

#print(weekend)

#corr motorrad with weather
motorBike = weekend["1"]
rain      = weekend["Rain"]
cloud     = weekend["Cloud"]

# feature map
bikeWeather = weekend[["1", "2", "3", "4", "Rain", "Cloud", "Humidity", "Tourism"]]

print(bikeWeather.corr())

#f, ax = plt.subplots()
#sns.heatmap(bikeWeather.corr())
#sns.pairplot(bikeWeather)
#plt.show()
