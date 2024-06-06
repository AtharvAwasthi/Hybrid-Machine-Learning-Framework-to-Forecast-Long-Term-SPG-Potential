# Hybrid-Machine-Learning-Framework-to-Forecast-Long-Term-SPG-Potential
This project highlights my independent work with the Lifelong Machine Learning Group of the Modular Robotics (GRASP) Lab at the University of Pennsylvania.
My research was funded by the Vagelos Integrated Program in Energy Research (VIPER).
I developed a climate-change-informed algorithm based on three learning models (Prophet, RFR, CNN) to forecast long-term solar power generation.

Datasets used to train constituent models were exported from Kaggle.

**Expected Input**

The input dataset is a weather dataset in CSV format and should contain the following columns:

  • Temperature (C)

  • Humidity

  • Wind Speed (km/h)

  • Pressure (millibars)

  • Date (format = yyyy-mm-dd hh:mm:00)


**Implementation Details**

• Facebook Prophet Model:

Given an input spanning the past _x_ years, the algorithm uses the Prophet library to perform time series forecasting (hourly freq.) for Temperature, Humidity, Wind Speed, and Pressure _x_ years into the future.

• Random Forest Regression:

The forecasted output from the Prophet model is then inputted into a seperately trained Random Forest Regression model to generate radiation values for each row of future weather predictions in the predictions DataFrame.

• Convolutional Neural Network:

The algorithm utilizes a seperately-trained L2-regularizated Convolutional Neural Network to predict solar power generation potential for a 10-acre solar farm based on the combined forecasted weather features.

**Potential Impact**
We selected a random sample of current US solar farms, and for each of these facilities, we randomly selected five regions in a 100-mile radius, and applied their 5-year trailing weather data to our algorithm, scaling the projected solar power generation relative to the size of . We then randomly selected five regions in a 100-mile radius, and applied our algorithm to these locations. For each US solar farm, 
