import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_squared_error, r2_score

from prophet import Prophet


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


def main():
    parser = argparse.ArgumentParser(description='NNs')
    parser.add_argument('file_path', type=str, help='Path to the data file')
    args = parser.parse_args()
    print("")
    data = pd.read_csv(args.file_path)

 
    data['Date'] = pd.to_datetime(data['Formatted Date'].str.slice(0, 18))

    data = data.drop(columns=['Visibility (km)', 'Summary', 'Precip Type', 'Daily Summary', 'Apparent Temperature (C)', 'Formatted Date'])
    # included columns = Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees),
    #################### Cloud Cover, Pressure (millibars), Date
    
    data = data.groupby('Date').mean().reset_index()
    data = data.sort_values(by='Date')
    data = data.reset_index(drop=True)



    # Use Meta's Prophet software to create a dataframe forecasting Temperature (C),
    # Humidity, Wind Speed (km/h),  Wind Bearing (degrees), Cloud Cover,  Pressure (millibars)
    predictions = pd.DataFrame()
    predictions['Date'] = data['Date']    
    predictions['Temperature (C)'] = features_pred(data, "Temperature (C)")
    predictions['Humidity'] = features_pred(data, "Humidity")
    predictions['Wind Speed (km/h)'] = features_pred(data, "Wind Speed (km/h)")
    predictions['Wind Bearing (degrees)'] = features_pred(data, "Wind Bearing (degrees)")
    predictions['Pressure (millibars)'] = features_pred(data, "Pressure (millibars)")



    # Create a scatter plot comparing actual and predicted temperatures (replicable for humidity, wind speed, etc.)
    plt.scatter(predictions['Date'], predictions['Temperature (C)'], label='Predicted Values', color='blue', marker='o')
    plt.scatter(data['Date'], data['Temperature (C)'], label='Actual Values', color='red', marker='x')

    # Label the axes
    plt.xlabel('Date')
    plt.ylabel('Temperature')

    # Add a legend
    plt.legend()

    # Add a title
    plt.title('Scatter Plot of Date vs. Predicted Temperature')
    plt.savefig('Predicted_Temp.png')
    

if __name__ == '__main__':
    main()   
