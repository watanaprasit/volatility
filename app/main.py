from fastapi import FastAPI
import pandas as pd
from fastapi.responses import JSONResponse
import os

app = FastAPI()

# Define the path to your CSV file
csv_file_path = "public/merged_crude_oil_ovx_data.csv"

@app.get("/")
async def read_csv():
    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_file_path)

        # Convert the DataFrame to a dictionary and return it as JSON
        data_dict = data.to_dict(orient="records")
        return JSONResponse(content=data_dict)
    else:
        return {"message": "CSV file not found."}

