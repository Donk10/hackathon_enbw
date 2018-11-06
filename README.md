![alt text](./visualization/deepTrap.png)

# ENBW Hackathon – All About Traffic

Traffic prediction using LSTMs for personalized Out-of-Home advertisement.


## Prerequisites

Packages needed to run the code, installed using ```pip install```:

```
pytorch
seaborn
pandas
numpy
matplotlib
```

Data used:
 - weather  https://cdc.dwd.de/portal/201810240858/searchview
 - tourism  http://www.statistik-bw.de/TourismGastgew/Tourismus/
 - traffic  provided by EnBW


## Running the prediction using LSTM.py

The code can't be used directly. Traffic data provided by EnBW was used and stored for specific timeframe together with weather, tourism and other data.


## Visualization


Comparison of LSTM vs. statistically predicted traffic count (Statistical prediction by calculating mean over same times of same weekdays).

Time range 2.5 days (1 bin = 1 hour):
![alt text](./visualization/plot.png)

Time range 21 days (1 bin = 1 hour):
![alt text](./visualization/plot_long.png)

Predicted values using LSTM prediction are at most times more precise than the naive guess using averages. More training data and a broader feature space would probably increase the accuracy even more.


## Authors

* **Arnold, Elias**
* **Blessing, Luca**
* **Dorkenwald, Michael**
* **Lüth, Carsten**
* **Ziegler, John**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
