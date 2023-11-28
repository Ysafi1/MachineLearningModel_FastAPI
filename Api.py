from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

pipeline = joblib.load('./dev/pipeline2.joblib')
encoder = joblib.load('./dev/encoder.joblib')

app = FastAPI(
    title= "Sepsis Analysis",
    description= "This App Is For Sepsis Analysis"
)

# Input Features
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
    #Sepssis: object
    
@app.get('/')
def read_root():
    return 'predicting the sepsis infection'

@app.post('/predict_infection')
def predict_sepsis_infection(sepsis_features: SmartFeatures):
    try:
        df = pd.DataFrame([sepsis_features.dict()])
        
        # encoded_data = encoder.transform(df)
        
        flattened_data = df.values.flatten()
        
        df_for_prediction = pd.DataFrame([flattened_data], columns=df.columns)
        
        # Log dimensions before prediction
        logging.debug(f"Shape before prediction: {df_for_prediction.shape}")

        #Prediction
        predict = pipeline.predict(df_for_prediction.values.reshape(1, -1))
        
        # Log dimensions after prediction
        logging.debug(f"Shape after prediction: {predict.shape}")
        
        return {"Prediction": predict[0]}
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Additional routes and functionality can be added as needed

