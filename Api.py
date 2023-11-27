from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

pipeline = joblib.load('./dev/pipeline2.joblib')
encoder = joblib.load('./dev/encoder.joblib')

app = FastAPI()

class SmartFeatures(BaseModel):
    PRG: int 
    PL: int
    PR: int
    SK: int
    TS: int
    M11: float 
    BD2: float
    Age: int
    Insurance: int
    Sepssis: object

@app.post('/predict_infection')
def predict_sepsis_infection(sepsis_features: SmartFeatures):
    try:
        df = pd.DataFrame([sepsis_features.dict()])
        # Use encoder if needed
        # encoded_data = encoder.transform(df)
        predict = pipeline.predict(df)[0]
        return {"Prediction": predict}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Additional routes and functionality can be added as needed
