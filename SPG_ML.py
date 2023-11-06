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





def load_data(file_path: str) -> pd.DataFrame:
    """Loads data into a readable format using pandas."""
    return pd.read_csv(file_path)

def time_to_seconds(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

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


def spg_lr_random(data: pd.DataFrame) -> None: 
    # Split the data into features (X) and the target variable (y)
    X = data.drop(columns=['generated_power_kw'])
    y = data['generated_power_kw']

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
    print("Equation of the Non-Time-Based Generated Solar Power Linear Regression Model:")
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

def spg_lr_timebased(data: pd.DataFrame) -> None: 
   
    train_size = int(0.8 * len(data))

    # The top 80% of the sorted DataFrame will be used for training
    df_train = data.iloc[:train_size]

    # The bottom 20% of the sorted DataFrame will be used for testing
    df_test = data.iloc[train_size:]

    # Split the data into features (X) and the target variable (y)
    X_train = df_train.drop(columns=['generated_power_kw'])
    y_train = df_train['generated_power_kw']

    X_test = df_test.drop(columns=['generated_power_kw'])
    y_test = df_test['generated_power_kw']

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the coefficients (b1, b2, ..., bn) for the features (x1, x2, ..., xn)
    coefficients = model.coef_

    # Get the intercept (b0)
    intercept = model.intercept_

    # Print the equation of the linear regression model
    print("Equation of the Time-Based Generated Solar Power Linear Regression Model:")
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

def spg_lstm(data: pd.DataFrame) -> None:

    # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)
  
    # Select the target column you want to predict
    target_column = 'generated_power_kw'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column])
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

    print("SPG LSTM Model:")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("     Mean Squared Error:", mse)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print("     R-squared:", r2)

    # Calculate baseline performance
    baseline_preds = model.predict(X_test)
    baseline_mse = mean_squared_error(y_test, baseline_preds)

    # Initialize an array to store feature importance scores
    feature_importance = np.zeros(X_train.shape[2])

    # Iterate over each feature
    for feature_index in range(X_train.shape[2]):
        # Make a copy of the original test data
        perturbed_X_test = X_test.copy()
        
        # Permute the values of the selected feature
        perturbed_X_test[:, :, feature_index] = np.random.permutation(perturbed_X_test[:, :, feature_index])
        
        # Make predictions on the perturbed data
        perturbed_preds = model.predict(perturbed_X_test)
        
        # Calculate mean squared error on the perturbed data
        perturbed_mse = mean_squared_error(y_test, perturbed_preds)
        
        # Calculate feature importance score as the decrease in performance
        feature_importance[feature_index] = baseline_mse - perturbed_mse

    # Normalize the feature importance scores
    feature_importance /= feature_importance.sum()

    # Get the names of the features
    feature_names = X.columns

    # Sort feature importance and feature names together
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Print the feature importance scores
    print()
    for feature_name, importance in zip(sorted_feature_names, sorted_feature_importance):
        print(f"Feature: {feature_name}, Importance: {importance}")

    # Create a bar plot of feature importances
    plt.figure(figsize=(20, 6))
    plt.barh(range(len(sorted_feature_names)), sorted_feature_importance)
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance (LSTM)')
    plt.savefig('LSTM_PFI.png')


def spg_ann(data:pd.DataFrame) -> None:
    # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)

    # Select the target column you want to predict
    target_column = 'generated_power_kw'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Specify the indices of the features we want to emphasize
    feat1_index = X.columns.get_loc("zenith")
    feat2_index = X.columns.get_loc("azimuth")
    feat3_index = X.columns.get_loc("angle_of_incidence")
    feat4_index = X.columns.get_loc("shortwave_radiation_backwards_sfc")

    # Build the ANN model
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_dim=X_train.shape[1]))
    model.add(CustomAttentionLayer(feat_indices=[feat1_index, feat2_index, feat3_index, feat4_index]))  # Adding custom attention layer
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

    print("SPG ANN Model:")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("     Mean Squared Error:", mse)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print("     R-squared:", r2)

    # Calculate baseline performance
    baseline_mse = mse

    # Initialize an array to store feature importance scores
    feature_importance = np.zeros(X_test.shape[1])

    # Iterate over each feature
    for feature_index in range(X_test.shape[1]):
        # Make a copy of the original test data
        perturbed_X_test = X_test.copy()
        
        # Permute the values of the selected feature
        perturbed_X_test.iloc[:, feature_index] = np.random.permutation(perturbed_X_test.iloc[:, feature_index])
        
        # Make predictions on the perturbed data
        perturbed_preds = model.predict(perturbed_X_test)
        
        # Calculate mean squared error on the perturbed data
        perturbed_mse = mean_squared_error(y_test, perturbed_preds)
        
        # Calculate feature importance score as the decrease in performance
        feature_importance[feature_index] = baseline_mse - perturbed_mse

    # Normalize the feature importance scores
    feature_importance /= feature_importance.sum()

    # Get the names of the features
    feature_names = X.columns

    # Sort feature importance and feature names together
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Print the feature importance scores
    print()
    for feature_name, importance in zip(sorted_feature_names, sorted_feature_importance):
        print(f"Feature: {feature_name}, Importance: {importance}")

    # Create a bar plot of feature importances
    plt.figure(figsize=(20, 6))
    plt.barh(range(len(sorted_feature_names)), sorted_feature_importance)
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance (ANN)')
    plt.savefig('ANN_PFI.png')

def spg_rf(data:pd.DataFrame) -> None:
    # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)
    
    # Select the target column you want to predict
    target_column = 'generated_power_kw'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column, 'high_cloud_cover_high_cld_lay', 'wind_direction_10_m_above_gnd', 'wind_speed_900_mb', 'low_cloud_cover_low_cld_lay', 'wind_speed_80_m_above_gnd', 'wind_direction_900_mb', 'medium_cloud_cover_mid_cld_lay', 'total_precipitation_sfc', 'wind_direction_80_m_above_gnd', 'wind_speed_10_m_above_gnd', 'snowfall_amount_sfc'])
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

    print("SPG Random Forest Regression Model")
    print(f"    Mean Squared Error: {mse:.5f}")
    print(f"    R-squared: {r2:.5f}")











    # Calculate baseline performance
    baseline_mse = mse

    # Initialize an array to store feature importance scores
    feature_importance = np.zeros(X_test.shape[1])

    # Iterate over each feature
    for feature_index in range(X_test.shape[1]):
        # Make a copy of the original test data
        perturbed_X_test = X_test.copy()
        
        # Permute the values of the selected feature
        perturbed_X_test.iloc[:, feature_index] = np.random.permutation(perturbed_X_test.iloc[:, feature_index])
        
        # Make predictions on the perturbed data
        perturbed_preds = model.predict(perturbed_X_test)
        
        # Calculate mean squared error on the perturbed data
        perturbed_mse = mean_squared_error(y_test, perturbed_preds)
        
        # Calculate feature importance score as the decrease in performance
        feature_importance[feature_index] = baseline_mse - perturbed_mse

    # Normalize the feature importance scores
    feature_importance /= feature_importance.sum()

    # Get the names of the features
    feature_names = X.columns

    # Sort feature importance and feature names together
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Print the feature importance scores
    print()
    for feature_name, importance in zip(sorted_feature_names, sorted_feature_importance):
        print(f"Feature: {feature_name}, Importance: {importance}")

    # Create a bar plot of feature importances
    plt.figure(figsize=(20, 6))
    plt.barh(range(len(sorted_feature_names)), sorted_feature_importance)
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.savefig('RF_PFI.png')

def spg_svm(data:pd.DataFrame) -> None:
     # To Preprocess Data, Initialize the StandardScaler or MinMaxScaler
    scaler = StandardScaler()

    # Apply StandardScaler or MinMaxScaler to the DataFrame
    normalized_data = scaler.fit_transform(data)

    # Create a new DataFrame with normalized data
    data = pd.DataFrame(normalized_data, columns=data.columns)
    
    # Select the target column you want to predict
    target_column = 'generated_power_kw'

    # Split your data into input features and target column
    X = data.drop(columns=[target_column, 'high_cloud_cover_high_cld_lay', 'wind_direction_10_m_above_gnd', 'wind_speed_900_mb', 'low_cloud_cover_low_cld_lay', 'wind_speed_80_m_above_gnd', 'wind_direction_900_mb', 'medium_cloud_cover_mid_cld_lay', 'total_precipitation_sfc', 'wind_direction_80_m_above_gnd', 'wind_speed_10_m_above_gnd', 'snowfall_amount_sfc'])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Define hyperparameters and their possible values
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'epsilon': [0.1, 0.01, 0.001]
    }

    # Create the SVM model
    base_model = SVR()

    # Create GridSearchCV with cross-validation
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Train the final model using the best hyperparameters
    model = SVR(**best_params)

    # Train the model using the scaled training data
    model.fit(X_train, y_train)

    # Make predictions on the scaled testing data
    y_pred = model.predict(X_test)

    # Calculate mean squared error and R-squared score to evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("SPG SVM Model")
    print(f"    Mean Squared Error: {mse:.5f}")
    print(f"    R-squared: {r2:.5f}")











    # Calculate baseline performance
    baseline_mse = mse

    # Initialize an array to store feature importance scores
    feature_importance = np.zeros(X_test.shape[1])

    # Iterate over each feature
    for feature_index in range(X_test.shape[1]):
        # Make a copy of the original test data
        perturbed_X_test = X_test.copy()
        
        # Permute the values of the selected feature
        perturbed_X_test.iloc[:, feature_index] = np.random.permutation(perturbed_X_test.iloc[:, feature_index])
        
        # Make predictions on the perturbed data
        perturbed_preds = model.predict(perturbed_X_test)
        
        # Calculate mean squared error on the perturbed data
        perturbed_mse = mean_squared_error(y_test, perturbed_preds)
        
        # Calculate feature importance score as the decrease in performance
        feature_importance[feature_index] = baseline_mse - perturbed_mse

    # Normalize the feature importance scores
    feature_importance /= feature_importance.sum()

    # Get the names of the features
    feature_names = X.columns

    # Sort feature importance and feature names together
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]

    # Print the feature importance scores
    print()
    for feature_name, importance in zip(sorted_feature_names, sorted_feature_importance):
        print(f"Feature: {feature_name}, Importance: {importance}")

    # Create a bar plot of feature importances
    plt.figure(figsize=(20, 6))
    plt.barh(range(len(sorted_feature_names)), sorted_feature_importance)
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance (SVM)')
    plt.savefig('SVM_PFI.png')




def spg_cnn(data:pd.DataFrame) -> None:
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

    # To implement Permutation Feature Importance, we create a perturbed version of the
    # test data by permuting the values of the selected feature, and calculate the mean
    # squared error on the perturbed data. The decrease in performance (increase in MSE)
    # after permuting a feature is used as an estimate of its importance. The importance
    # scores are normalized to sum up to 1.


    # Calculate the baseline performance (mean squared error) on the original test data
    baseline_preds = model.predict(X_test_reshaped)
    baseline_mse = mean_squared_error(y_test, baseline_preds)

    # Initialize an array to store feature importance scores
    feature_importance = np.zeros(X_test_reshaped.shape[1])
    
    # Iterate over each feature
    for feature_index in range(X_test_reshaped.shape[1]):
        # Make a copy of the original test data
        perturbed_X_test = X_test_reshaped.copy()
        
        # Permute the values of the selected feature
        perturbed_X_test[:, feature_index, :] = np.random.permutation(perturbed_X_test[:, feature_index, :])
        
        # Calculate predictions on the perturbed data
        perturbed_preds = model.predict(perturbed_X_test)
        
        # Calculate mean squared error on the perturbed data
        perturbed_mse = mean_squared_error(y_test, perturbed_preds)
        
        # Calculate feature importance score as the decrease in performance
        feature_importance[feature_index] = baseline_mse - perturbed_mse

    # Normalize the feature importance scores
    feature_importance /= feature_importance.sum()

    # Print the feature importance scores
    for feature_name, importance in zip(X.columns, feature_importance):
        print(f"Feature: {feature_name}, Importance: {importance}")

    # Sort feature importance and feature names together
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [X.columns[i] for i in sorted_indices]

    # Evaluate the model
    loss = model.evaluate(X_test_reshaped, y_test)

    # Make predictions
    y_pred = model.predict(X_test_reshaped)

    print("")

    print("SPG CNN Model:")

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("     Mean Squared Error:", mse)

    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    print("     R-squared:", r2)

    # Reshape y_pred and y_test to 1-dimensional arrays
    y_pred_1d = y_pred.flatten()
    y_test_1d = y_test.values.flatten()

    # Calculate errors for each data point
    errors = np.square(y_pred_1d - y_test_1d)

    # Sort errors and predictions based on original order of test indices
    original_order_indices = X_test.index.argsort()
    sorted_errors = errors[original_order_indices]
    sorted_predictions = y_pred_1d[original_order_indices]






    # Create a scatterplot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, c='red', label='Predicted vs. Actual')

    # Add a diagonal line through the middle
    min_val = min(y_test.min(), y_test.min())
    max_val = max(y_test.max(), y_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='blue', label='Diagonal Line')
    
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
    plt.xlabel('Actual Solar Power Generation (SPG)', label_font)
    plt.ylabel('Predicted Solar Power Generation (SPG)', label_font)
    plt.title(f'Convolutional Neural Network Predicting SPG', title_font)
    plt.savefig('SPG.png')
        










def main():
    parser = argparse.ArgumentParser(description='NNs')
    parser.add_argument('file_path', type=str, help='Path to the data file')

    args = parser.parse_args()

    
    print("")
    data = load_data(args.file_path)
    
    #spg_lr_random(data) #R^2 = ~0.71
    #spg_lr_timebased(data) #R^2 = ~0.53
    #spg_lstm(data) #R^2 = 0.35 - 0.55 
    #spg_ann(data) #R^2 = 0.55 - 0.62     
    #spg_rf(data) #R^2 = 0.75 - 0.85 
    #spg_svm(data) #R^2 = 0.75 - 0.8
    print("")

    spg_cnn(data) #R^2 = 0.80 - 0.85 
    


if __name__ == '__main__':
    main()   
