import yfinance as yf
import pandas as pd
import os

def fetch_data():
    # Fetch crude oil and OVX data
    crude_oil = yf.Ticker("BZ=F")
    ovx = yf.Ticker("^OVX")

    # Retrieve historical data (daily, last 5 years)
    crude_oil_data = crude_oil.history(period="5y")
    ovx_data = ovx.history(period="5y")

    # Keep only the 'Close' columns for both datasets
    crude_oil_data = crude_oil_data[['Close']].rename(columns={'Close': 'Crude_Oil_Price'})
    ovx_data = ovx_data[['Close']].rename(columns={'Close': 'OVX'})

    # Merge both datasets on the Date index
    merged_data = pd.merge(crude_oil_data, ovx_data, left_index=True, right_index=True, how='inner')

    # Define the path to the static folder
    static_folder = 'static'

    # Create the static folder if it doesn't exist
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    # Save the merged data to a CSV file inside the static folder
    csv_path = os.path.join(static_folder, 'merged_crude_oil_ovx_data.csv')
    merged_data.to_csv(csv_path)

    print(f"Data fetched and saved to '{csv_path}'.")

# Run the fetch_data function if this file is executed directly
if __name__ == "__main__":
    fetch_data()
