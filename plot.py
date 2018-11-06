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
trafficFeatures  = normalize(csvTrafficFile_DataFrame.iloc[:,5:14])
tourismFeatures  = normalize(arrayTourism())
cloudFeatures    = normalize(loadCloudData())
rainFeatures     = normalize(loadRainData())

#append features
csvTrafficFile_DataFrame["1"]        = trafficFeatures.iloc[:,0]
csvTrafficFile_DataFrame["2"]        = trafficFeatures.iloc[:,1]
csvTrafficFile_DataFrame["3"]        = trafficFeatures.iloc[:,2]
csvTrafficFile_DataFrame["4"]        = trafficFeatures.iloc[:,3]
csvTrafficFile_DataFrame["count"]    = trafficFeatures.iloc[:,4]
csvTrafficFile_DataFrame["prozent1"] = trafficFeatures.iloc[:,5]
csvTrafficFile_DataFrame["prozent2"] = trafficFeatures.iloc[:,6]
csvTrafficFile_DataFrame["prozent3"] = trafficFeatures.iloc[:,7]
csvTrafficFile_DataFrame["prozent4"] = trafficFeatures.iloc[:,8]
csvTrafficFile_DataFrame.insert(14, "Tourism", tourismFeatures)
csvTrafficFile_DataFrame.insert(15, "Cloud", cloudFeatures)
csvTrafficFile_DataFrame.insert(16, "Rain", rainFeatures)

print(csvTrafficFile_DataFrame)

csvTrafficFile_DataFrame.to_csv("./data/features_normalized.csv")

#do correlation map of features
#f, ax = plt.subplots()
#sns.heatmap(features.corr())
#plt.show()
