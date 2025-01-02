from fastapi import FastAPI
import pandas as pd
import os
import joblib
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# Path to the CSV file
csv_file_path = "public/merged_crude_oil_ovx_data.csv"

# Path to the model file in the 'models' folder
model_file_path = "models/crude_oil_volatility_model.pkl"

# Load the model when the app starts
try:
    model = joblib.load(model_file_path)  # Load the model directly from the models folder
    print(f"Model loaded successfully from {model_file_path}")
except Exception as e:
    print(f"Error loading model from file: {e}")
    model = None

# Define a Pydantic model to handle input data
class PredictionRequest(BaseModel):
    crude_oil_price: float  # Current crude oil price
    ovx_value: float  # OVX (Oil Volatility Index) value

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

@app.post("/predict")
async def predict_volatility(request: PredictionRequest):
    # Check if the model is loaded
    if model is None:
        return {"error": "Model is not loaded. Please check the model file."}

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Crude_Oil_Price': [request.crude_oil_price],
        'OVX': [request.ovx_value],
    })

    try:
        # Check if the model has the 'predict' method
        if hasattr(model, 'predict'):
            # Predict the next week's volatility using the model
            prediction = model.predict(input_data)
            
            # Return the prediction in a simple JSON format
            return {"next_week_volatility_prediction": prediction[0]}
        else:
            return {"error": "Model does not have a 'predict' method."}
    except Exception as e:
        return {"error": str(e)}
