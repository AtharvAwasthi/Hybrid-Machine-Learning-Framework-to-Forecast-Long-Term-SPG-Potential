import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from typing import List
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten

from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet



def load_data(file_path: str) -> pd.DataFrame:
    """Loads data into a readable format using pandas."""
    return pd.read_csv(file_path)


def time_to_hour(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return hours

def time_to_seconds_since_2016(time_str):
    # Define the start date (January 1, 2016) as a datetime object
    start_date = datetime(2016, 1, 1)

    # Convert the input time string to a datetime object
    time_obj = datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p')

    # Calculate the time difference in seconds between the input time and the start of 2016
    time_difference = time_obj - start_date

    # Convert the time difference to total seconds
    total_seconds = time_difference.total_seconds()

    return total_seconds


def radiation_lr_random(data: pd.DataFrame) -> None: 
    # Split the data into features (X) and the target variable (y)
    X = data.drop(columns=['Radiation'])
    y = data['Radiation']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the coefficients (b1, b2, ..., bn) for the features (x1, x2, ..., xn)
    coefficients = model.coef_

    # Get the intercept (b0)
    intercept = model.intercept_

    # Print the equation of the linear regression model
    print("Equation of the Non-Time-Based Radiation Linear Regression Model:")
    equation = f"y = {intercept:.2f} + "
    for feature, coef in zip(X_train.columns, coefficients):
        equation += f"{coef:.2f} * {feature} + "
    print(equation[:-3])  # Remove the extra ' + ' at the end

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error and R-squared score to evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("")
    print(f"    Mean Squared Error: {mse:.5f}")
    print(f"    R-squared: {r2:.5f}")

def radiation_lr_timebased(data: pd.DataFrame) -> None: 
   
    train_size = int(0.8 * len(data))

    # The top 80% of the sorted DataFrame will be used for training
    df_train = data.iloc[:train_size]

    # The bottom 20% of the sorted DataFrame will be used for testing
    df_test = data.iloc[train_size:]

    # Split the data into features (X) and the target variable (y)
    X_train = df_train.drop(columns=['Radiation'])
    y_train = df_train['Radiation']

    X_test = df_test.drop(columns=['Radiation'])
    y_test = df_test['Radiation']

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the coefficients (b1, b2, ..., bn) for the features (x1, x2, ..., xn)
    coefficients = model.coef_

    # Get the intercept (b0)
    intercept = model.intercept_

    # Print the equation of the linear regression model
    print("Equation of the Time-Based Radiation Linear Regression Model:")
    equation = f"y = {intercept:.2f} + "
    for feature, coef in zip(X_train.columns, coefficients):
        equation += f"{coef:.2f} * {feature} + "
    print(equation[:-3])  # Remove the extra ' + ' at the end

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error and R-squared score to evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("")
    print(f"    Mean Squared Error: {mse:.5f}")
    print(f"    R-squared: {r2:.5f}")

def radiation_lstm(data: pd.DataFrame) -> None:

    # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)
  
    # Select the target column you want to predict
    target_column = 'Radiation'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column]) #, 'Time', 'Data', 'TimeSunRise', 'TimeSunRise'])
    y = data[target_column]

    # Prepare input sequences and labels
    def prepare_sequences(X, y, sequence_length):
        assert len(X) == len(y), "X and y should have same length"
        
        sequences = []
        labels = []
        for i in range(len(X) - sequence_length):
            seq = X.iloc[i : i + sequence_length]
            label = y.iloc[i + sequence_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    sequence_length = 10  # Adjusted as needed
    X_train_seq, y_train_seq = prepare_sequences(X, y, sequence_length)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_seq, y_train_seq, test_size=0.2, shuffle=False)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, X_train.shape[2])))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mse', optimizer='adam')

    # Print the model summary
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)

    # Make predictions
    y_pred = model.predict(X_test)

    print("")

    print("Radiation LSTM Model:")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("     Mean Squared Error:", mse)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print("     R-squared:", r2)

def radiation_ann(data:pd.DataFrame) -> None:
    # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)

    # Select the target column you want to predict
    target_column = 'Radiation'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Build the ANN model
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print("Test loss:", loss)

    # Make predictions
    y_pred = model.predict(X_test)

    print("")

    print("Radiation ANN Model:")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("     Mean Squared Error:", mse)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print("     R-squared:", r2)

def radiation_rf(data:pd.DataFrame) -> None:
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

    # Make predictions on the scaled testing data
    y_pred = model.predict(X_test)

    # Calculate mean squared error and R-squared score to evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Radiation Random Forest Regression Model")
    print(f"    Mean Squared Error: {mse:.5f}")
    print(f"    R-squared: {r2:.5f}")




    # Create a scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, c='blue', label='Predicted vs. Actual')

    # Add a diagonal line through the middle
    min_val = min(y_test.min(), y_test.min())
    max_val = max(y_test.max(), y_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Diagonal Line')
    
    # Add labels
    title_font = {
        'family': 'serif',    # Font family (e.g., 'serif', 'sans-serif', 'monospace')
        'color':  'black',     # Text color
        'weight': 'bold',     # Font weight (e.g., 'normal', 'bold', 'heavy', 'light')
        'size': 16            # Font size
    }
    label_font = {
        'family': 'sans-serif',    # Font family (e.g., 'serif', 'sans-serif', 'monospace')
        'color':  'black',     # Text color
        'weight': 'normal',     # Font weight (e.g., 'normal', 'bold', 'heavy', 'light')
        'size': 14            # Font size
    }
    plt.xlabel('Actual Radiation', label_font)
    plt.ylabel('Predicted Radiation', label_font)
    plt.title(f'Random Forest Regressor Targeting Radiation', title_font)
    plt.savefig('Radiation.png')

def radiation_cnn(data:pd.DataFrame) -> None:
    # Select the target column you want to predict
    target_column = 'Radiation'

    # Split data into features (X) and target column (y)
    X = data.drop(columns=[target_column])
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

    # Create the CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(X_test_reshaped, y_test)

    # Make predictions
    y_pred = model.predict(X_test_reshaped)

    print("")

    print("Radiation CNN Model:")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("     Mean Squared Error:", mse)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print("     R-squared:", r2)


def main():
    parser = argparse.ArgumentParser(description='NNs')
    parser.add_argument('file_path', type=str, help='Path to the data file')

    args = parser.parse_args()

    
    print("")
    data = load_data(args.file_path)
    
    data['Hour'] = data['Time'].apply(time_to_hour)
    data['Date'] = pd.to_datetime(data['Data'], format='%m/%d/%Y %I:%M:%S %p')
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Year'] = data['Date'].dt.year

    data['WindSpeed'] = data['Speed']

    data = data[['Temperature', 'Pressure', 'Humidity', 'WindSpeed', 'Year',
                 'Day', 'Month', 'Hour', 'Radiation']]

    #radiation_lr_random(data) #R^2 = 0.5 - 0.6
    print("")
    #radiation_lr_timebased(data) #R^2 = 0.25 - 0.4
    print("")
    #radiation_lstm(data) #R^2 = 0.4 - 0.55
    print("")
    #radiation_ann(data) #R^2 = 0.4 - 0.55
    print("")
    #radiation_cnn(data) #R^2 = ~0.87
    print("")
    radiation_rf(data) #R^2 = ~0.95
    print("")


if __name__ == '__main__':
    main()   


