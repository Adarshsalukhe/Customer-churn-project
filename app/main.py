from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Get the directory this file lives in, regardless of where it's run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved artifacts using absolute paths
model = joblib.load(os.path.join(BASE_DIR, 'churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

app = FastAPI(title="Customer Churn Prediction API")

# Define the expected input structure
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert incoming data to a dataframe
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Apply same preprocessing as training
    binary_map = {'Yes': 1, 'No': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map(binary_map)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols)

    # Align columns with training set (add missing cols as 0, drop extras, fix order)
    df = df.reindex(columns=model_columns, fill_value=0)

    # Scale and predict
    df_scaled = scaler.transform(df)
    probability = model.predict_proba(df_scaled)[0][1]
    prediction = "Yes" if probability >= 0.5 else "No"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(probability), 4)
    }