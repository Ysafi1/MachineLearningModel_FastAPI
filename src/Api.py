from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

pipeline = joblib.load('./dev/pipeline2.joblib')
encoder = joblib.load('./dev/encoder.joblib')

print(pipeline)
print(encoder)

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
