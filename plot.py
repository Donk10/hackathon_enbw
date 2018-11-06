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
csvTraffic = "./data/pandas_wide.csv"
features   = pd.read_csv(csvTraffic, sep=",")

#get features
trafficFeatures  = features.iloc[:,5:14]
tourismFeatures  = arrayTourism()
cloudFeatures    = loadCloudData()
rainFeatures     = loadRainData()

#append features
features["1"]        = trafficFeatures.iloc[:,0]
features["2"]        = trafficFeatures.iloc[:,1]
features["3"]        = trafficFeatures.iloc[:,2]
features["4"]        = trafficFeatures.iloc[:,3]
features["count"]    = trafficFeatures.iloc[:,4]
features["prozent1"] = trafficFeatures.iloc[:,5]
features["prozent2"] = trafficFeatures.iloc[:,6]
features["prozent3"] = trafficFeatures.iloc[:,7]
features["prozent4"] = trafficFeatures.iloc[:,8]
features.insert(14, "Tourism", tourismFeatures)
features.insert(15, "Cloud", cloudFeatures)
features.insert(16, "Rain", rainFeatures)

#get pfingsten
pfingsten = features[(features["y"]==2018)&(features["m"]==6)&(features["d"]>5)&(features["d"]<19)]
getBeforePfingsten = features[(features["y"]==2018)&((features["m"]==6)&(features["d"]>0)&(features["d"]<6))|((features["m"]==5)&(features["d"]>1))]
getAfterPfingsten = features[(features["y"]==2018)&((features["m"]==6)&(features["d"]>18))|((features["m"]==7)&(features["d"]<25))]


#plot pfingsten
pfingstenC1 = pfingsten[["prozent1"]]
pfingstenC2 = pfingsten[["prozent2"]]
pfingstenC3 = pfingsten[["prozent3"]]
pfingstenC4 = pfingsten[["prozent4"]]

pfingstenC1Before = getBeforePfingsten[["prozent1"]]
pfingstenC2Before = getBeforePfingsten[["prozent2"]]
pfingstenC3Before = getBeforePfingsten[["prozent3"]]
pfingstenC4Before = getBeforePfingsten[["prozent4"]]

pfingstenC1After = getAfterPfingsten[["prozent1"]]
pfingstenC2After = getAfterPfingsten[["prozent2"]]
pfingstenC3After = getAfterPfingsten[["prozent3"]]
pfingstenC4After = getAfterPfingsten[["prozent4"]]

#get mean in pfingsten
pfingstenC4         = pfingstenC4
mean                = np.mean(pfingstenC4)
pfingstenC4Before2  = pfingstenC4Before-mean

#


plt.figure(1)
plt.subplot(211)
plt.plot(pfingstenC4)
plt.plot(pfingstenC4Before)
plt.plot(pfingstenC4After)
plt.subplot(212)
plt.plot(pfingstenC4)
plt.plot(pfingstenC4Before2)
plt.plot(pfingstenC4After)
plt.show()

#csvTrafficFile_DataFrame.to_csv("./data/features_normalized.csv")

#do correlation map of features
#f, ax = plt.subplots()
#sns.heatmap(pfingsten.corr())
#plt.show()
