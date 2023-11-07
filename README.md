# Hybrid-Machine-Learning-Framework-to-Forecast-Long-Term-SPG-Potential
This project highlights my independent work with the Modular Robotics (GRASP) Lab at the University of Pennsylvania.
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

The algorithm utilizes a seperately-trained L2-regularizated Convolutional Neural Network to predict solar power generation potential based on the combined forecasted weather features.
