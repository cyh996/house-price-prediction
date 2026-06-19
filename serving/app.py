import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

APP_NAME = os.getenv("APP_NAME", "House Price API")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

app = FastAPI(title=APP_NAME)

model = joblib.load("models/xgb_house_price_model.pkl")

class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FirstFlrSF: int
    FullBath: int
    YearBuilt: int
    YearRemodAdd: int
    ExterQual: str
    Neighborhood: str
    KitchenQual: str
    BsmtQual: str
    FireplaceQu: str
    BsmtExposure: str
    MSZoning: str

@app.post("/predict")
def predict(data: HouseInput):
    input_df = pd.DataFrame([{
        "OverallQual": data.OverallQual,
        "GrLivArea": data.GrLivArea,
        "GarageCars": data.GarageCars,
        "TotalBsmtSF": data.TotalBsmtSF,
        "1stFlrSF": data.FirstFlrSF,
        "FullBath": data.FullBath,
        "YearBuilt": data.YearBuilt,
        "YearRemodAdd": data.YearRemodAdd,
        "ExterQual": data.ExterQual,
        "Neighborhood": data.Neighborhood,
        "KitchenQual": data.KitchenQual,
        "BsmtQual": data.BsmtQual,
        "FireplaceQu": data.FireplaceQu,
        "BsmtExposure": data.BsmtExposure,
        "MSZoning": data.MSZoning
    }])

    log_pred = model.predict(input_df)[0]
    predicted_price = np.expm1(log_pred)

    return {
        "predicted_price": round(float(predicted_price), 2)
    }

@app.get("/info")
def info():
    return {
        "app_name": APP_NAME,
        "model_version": MODEL_VERSION
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }
