import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from typing import List
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Layer

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
import shap
from keras import regularizers
import keras.backend as K
from sklearn.svm import SVR



def weather_prophet(train: pd.DataFrame) -> pd.DataFrame:
    model = Prophet() 
    model.fit(train)
    future = model.make_future_dataframe(periods=(len(train)))
    predictions = model.predict(future)
    return predictions

def features_pred(data: pd.DataFrame, column) -> pd.DataFrame:
    train = data.copy()
    train = train.rename(columns = {"Date": "ds", column: "y"})
    predictions = weather_prophet(train.iloc[:((int) (0.5 * len(data)))])
    predictions = predictions.rename(columns = {"yhat_upper": column, "ds": "Date"})
    return predictions[[column]]


def radiation_rf(file_path: str) -> RandomForestRegressor:

    # Read and format data
    data = pd.read_csv(file_path)

    data['Hour'] = data['Time'].apply(time_to_hour)
    data['Date'] = pd.to_datetime(data['Data'], format='%m/%d/%Y %I:%M:%S %p')
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Year'] = data['Date'].dt.year
    data['WindSpeed'] = data['Speed']

    data = data[['Temperature', 'Pressure', 'Humidity', 'WindSpeed', 'Year',
                 'Day', 'Month', 'Hour', 'Radiation']]

    # Select the target column you want to predict
    target_column = 'Radiation'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Create the Random Forest Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=None)

    # Train the model using the scaled training data
    model.fit(X_train, y_train)
    
    return model



def spg_cnn(data:pd.DataFrame) -> Sequential:
    # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)
  
    # Select the target column you want to predict
    target_column = 'generated_power_kw'

    # Split data into features (X) and target column (y)
    X = data[['temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd', 'mean_sea_level_pressure_MSL',
              'shortwave_radiation_backwards_sfc', 'wind_gust_10_m_above_gnd', 'angle_of_incidence', 'zenith', 'azimuth']]
    # included columns = humidity, windspeed (10 m), angle of incidence
    #                    pressure, temperature, radiation, azimuth, zenith
    y = data[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Normalize/Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape input data to match CNN's input shape
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # Create the CNN model with L2 regularization
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1), kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization added here
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization added here
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))   # L2 regularization added here
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))   # L2 regularization added here
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=40, batch_size=32, validation_split=0.2)

    return model


def final_algo(data:pd.DataFrame) -> data:pd.DataFrame:

    # Order data
    data = data.groupby('Date').mean().reset_index()
    data = data.sort_values(by='Date')
    data = data.reset_index(drop=True)

    # Use Meta's Prophet software to create a dataframe forecasting hourly Temperature (C),
    # Humidity, Wind Speed (km/h),  Wind Bearing (degrees), Cloud Cover,  Pressure (millibars)
    predictions = pd.DataFrame()
    predictions['Date'] = data['Date']

    predictions['Year'] = predictions['Date'].dt.year
    predictions['Month'] = predictions['Date'].dt.month
    predictions['Day'] = predictions['Date'].dt.day
    predictions['Hour'] = predictions['Date'].dt.hour 

    predictions['Temperature (C)'] = features_pred(data, "Temperature (C)")
    predictions['Humidity'] = features_pred(data, "Humidity")
    predictions['Wind Speed (km/h)'] = features_pred(data, "Wind Speed (km/h)")
    predictions['Pressure (millibars)'] = features_pred(data, "Pressure (millibars)")


    # Use Random Forest Regressor to generate radiation values for each future day
    rf_model = radiation_rf('SolarPrediction.csv')
    predictions['Radiation'] = model.predict(predictions[['Temperature (C)', 'Pressure (millibars)', 'Humidity',
                                                          'Wind Speed (km/h)', 'Year', 'Day', 'Month', 'Hour']])


    # Use Convolutional Neural Network to generate SPG predictions for each future day
    cnn_model = cpg_cnn('spg.csv')
    predictions['SPG'] = model.predict(predictions[['Temperature (C)', 'Humidity', 'Pressure (millibars)', 
                                                    'Radiation', 'Wind Speed (km/h)']])
    
    # Returns predictions dataframe
    return predictions

parser = argparse.ArgumentParser(description='Weather Dataset')
    parser.add_argument('file_path', type=str, help='Path to the data file')
    args = parser.parse_args()
    print("")
    
    # We assume inputted csv has the following columns:
    ###### Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees),
    ###### Cloud Cover, Pressure (millibars), Date (format = yyyy-mm-dd hh:mm:ss)
    data = pd.read_csv(args.file_path)

    # Run algorithm
    predictions = final_algo(data)
    
    # Create a scatter plot for SPG values
    plt.scatter(predictions['Date'], predictions['SPG'], label='Predicted Values', color='blue', marker='o')

    # Label the axes
    plt.xlabel('Date')
    plt.ylabel('Projected SPG')

    # Add a legend
    plt.legend()

    # Add a title
    plt.title('Scatter Plot of Date vs. Projected SPG')
    plt.savefig('Predicted_SPG.png')
    

if __name__ == '__main__':
    main()   
