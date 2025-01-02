# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_and_preprocess_data(csv_file_path):
    # Load the data from CSV
    data = pd.read_csv(csv_file_path)
    
    # Check for missing values
    if data.isnull().sum().any():
        print("Data contains missing values. Handling missing data...")
        data = data.dropna()  # Or fill missing values depending on your preference

    # Calculate the 7-day rolling volatility using the Crude Oil Price (or OVX)
    # Assuming volatility is the rolling standard deviation over 7 days of Crude_Oil_Price
    data['7_day_volatility'] = data['Crude_Oil_Price'].rolling(window=7).std()

    # Check for missing values after rolling calculation (first 6 rows will have NaN values for volatility)
    data = data.dropna()

    # Create the target variable for next week's volatility (shift by 7 days)
    data['next_week_volatility'] = data['7_day_volatility'].shift(-7)  # Shift by 7 days for next week

    # Drop the last 7 rows, as they will have NaN for 'next_week_volatility'
    data = data.dropna()

    # Features: current day's crude oil price and OVX
    X = data[['Crude_Oil_Price', 'OVX']]  # Features for the current day

    # Target: volatility for the next week (shifted 7 days)
    y = data['next_week_volatility']  # Target variable (next week's volatility)

    return X, y

def train_random_forest(X, y):
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

def save_model(model, model_path):
    # Save the trained model to a file using joblib
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Define file paths
    csv_file_path = 'public/merged_crude_oil_ovx_data.csv'
    model_path = 'models/crude_oil_volatility_model.pkl'  # Save model in 'models' folder

    # Load and preprocess the data
    X, y = load_and_preprocess_data(csv_file_path)

    # Train the Random Forest model
    model = train_random_forest(X, y)

    # Save the trained model
    save_model(model, model_path)

if __name__ == '__main__':
    main()
