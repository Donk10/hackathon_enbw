import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt

from IPython             import display
from loadCsv             import loadTraffic
from loadTourism         import arrayTourism
from loadRain            import loadRainData
from loadClouds          import loadCloudData

def normalize(array): return (array-np.mean(array))/np.std(array)

#load traffic data
csvTraffic                = "./data/pandas_wide.csv"
csvTrafficFile_DataFrame  = pd.read_csv(csvTraffic, sep=",")

#get features
trafficeFeatures = normalize(csvTrafficFile_DataFrame.iloc[:,5:10])
tourismFeatures  = normalize(arrayTourism())
cloudFeatures    = normalize(loadCloudData())
rainFeatures     = normalize(loadRainData())

#append features
features            = trafficeFeatures
features["Tourism"] = pd.Series(tourismFeatures)
features["Cloud"]   = pd.Series(cloudFeatures)
features["Rain"]    = pd.Series(rainFeatures)

features.to_csv("./data/features_normalized")

#do correlation map of features
f, ax = plt.subplots()
sns.heatmap(features.corr())
plt.show()
