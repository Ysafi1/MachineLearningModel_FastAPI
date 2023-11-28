from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

pipeline = joblib.load('./dev/pipeline2.joblib')
encoder = joblib.load('./dev/encoder.joblib')

app = FastAPI()

class smartfeatures(BaseModel):
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

@app.post('/predict_infec')

def predict_sespis_infec(sepsiss_features:smartfeatures):
    
    df = pd.DataFrame([sepsiss_features.dict()])
    predict = pipeline.predict(df)[0]

    return {"Prediction": predict}




#############
@app.post('/predict_infection')
def predict_sepsis_infection(sepsis_features: SmartFeatures):
    try:
        df = pd.DataFrame([sepsis_features.dict()])
        
        # Flatten the data
        flattened_data = df.values.flatten()

        # Create a new DataFrame with the flattened data and use numeric column names
        df_for_prediction = pd.DataFrame([flattened_data], columns=[f"col_{i}" for i in range(len(flattened_data))])

        # Log dimensions before prediction
        logging.debug(f"Shape before prediction: {df_for_prediction.shape}")

        # Prediction
        predict = pipeline.predict(df_for_prediction)
        # Log dimensions after prediction
        logging.debug(f"Shape after prediction: {predict.shape}")

        return {"Prediction": predict[0]}
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
