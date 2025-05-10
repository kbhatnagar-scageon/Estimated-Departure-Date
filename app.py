from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import pickle
import uvicorn
from datetime import datetime
import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model classes from the module
from hospital_los_module import FeatureBridgePredictor

# Create FastAPI app
app = FastAPI(
    title="Hospital Length of Stay Predictor",
    description="API to predict patient's expected discharge date",
    version="1.0.0",
)

# Define the list of supported comorbidities
SUPPORTED_COMORBIDITIES = [
    "Diabetes",
    "Hypertension",
    "COPD",
    "Coronary Artery Disease",
    "Chronic Kidney Disease",
    "Cancer",
    "Obesity",
    "Dementia",
    "Alcohol Use Disorder",
    "None",
    "Tuberculosis",
    "Anemia",
    "Malnutrition",
]


# Define simplified input schema
class PatientInput(BaseModel):
    mrn: str
    visit_date: str
    primary_diagnosis: str
    age: int
    gender: str
    comorbidities: str = ""


# Define output schema
class DischargeOutput(BaseModel):
    expected_discharge_date: str
    predicted_los: int
    confidence_interval: Dict[str, str]


# Helper function to validate input data
def validate_input(patient: PatientInput):
    # Validate comorbidities
    if patient.comorbidities:
        comorbidities_list = [c.strip() for c in patient.comorbidities.split(",")]
        for c in comorbidities_list:
            if c and c not in SUPPORTED_COMORBIDITIES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid comorbidity: '{c}'. Must be one of: {', '.join(SUPPORTED_COMORBIDITIES)}",
                )

    # Validate date format
    try:
        datetime.strptime(patient.visit_date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid visit_date format. Expected format: YYYY-MM-DD HH:MM:SS",
        )

    # Validate age
    if patient.age < 0 or patient.age > 120:
        raise HTTPException(status_code=400, detail="Age must be between 0 and 120")

    # Validate gender
    if patient.gender not in ["M", "F"]:
        raise HTTPException(status_code=400, detail="Gender must be either 'M' or 'F'")


# Load the model
try:
    # Load the pickle file
    model_path = "feature_bridge_model.pkl"
    with open(model_path, "rb") as f:
        predictor = pickle.load(f)
    print(f"Model loaded successfully from {model_path}!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    predictor = None


@app.get("/")
def read_root():
    model_status = "loaded" if predictor is not None else "not loaded"
    return {
        "message": f"Hospital Length of Stay Prediction API is running. Model is {model_status}."
    }


@app.post("/predict", response_model=DischargeOutput)
def predict_discharge_date(patient: PatientInput):
    """Predict the expected discharge date for a patient"""
    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check server logs for details.",
        )

    # Validate input data
    validate_input(patient)

    # Convert Pydantic model to dict
    patient_dict = patient.dict()

    # Make prediction
    try:
        # Using predict_from_json method
        result = predictor.predict_from_json(patient_dict)

        # Return output
        return {
            "expected_discharge_date": result["estimated_discharge_date"],
            "predicted_los": result["predicted_los"],
            "confidence_interval": {
                "earliest": result["earliest_discharge"],
                "latest": result["latest_discharge"],
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
