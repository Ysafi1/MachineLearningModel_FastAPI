

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Load the entire pipeline
#pipeline_filepath = "pipeline1.joblib"
pipeline = joblib.load('./dev/pipeline1.joblib')

class PatientData(BaseModel):
    Plasma_glucose : float
    Blood_Work_Result_1: float
    Blood_Pressure : float
    Blood_Work_Result_2 : float
    Blood_Work_Result_3 : float
    Body_mass_index  : float
    Blood_Work_Result_4: float
    Age: float
    Insurance: int

@app.get("/")
def read_root():
    explanation = {
        'message': "Welcome to the Sepsis Prediction App",
        'description': "This API allows you to predict sepsis based on patient data.",
        'usage': "Submit a POST request to /predict with patient data to make predictions.",
        
    }
    return explanation

@app.post("/predict")
def get_data_from_user(data: PatientData):
    user_input = data.dict()

    input_df = pd.DataFrame([user_input])

    # Make predictions using the loaded pipeline
    prediction = pipeline.predict(input_df)
    probabilities = pipeline.predict_proba(input_df)

    
    probability_of_positive_class = probabilities[0][1]

    # Calculate the prediction
    sepsis_status = "Positive" if prediction[0] == 1 else "Negative"
    sepsis_explanation = "A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention." if prediction[0] == 1 else "A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

    result = {
        'predicted_sepsis': sepsis_status,
        'probability': probability_of_positive_class,
        'sepsis_explanation': sepsis_explanation
    }
    return result