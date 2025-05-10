import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

class FeatureBridgePredictor:
    """
    Predictor that bridges between limited JSON input and full feature set
    """
    def __init__(self):
        self.full_model = None
        self.feature_stats = {}
        self.diagnosis_stats = {}
        self.features_by_diagnosis = {}
        self.input_columns = []
        self.expected_columns = []
        self.model_type = None
        
    def fit(self, df):
        """Train the model and compute feature statistics"""
        # Implementation details (not required for loading the model)
        pass
    
    def predict_from_json(self, patient_json):
        """
        Predict LOS using only the fields in the JSON input
        by building a bridge to the full feature set
        """
        # Convert JSON to DataFrame
        patient_data = pd.DataFrame([patient_json])
        
        # Extract basic information
        visit_date = pd.to_datetime(patient_json.get('visit_date'))
        diagnosis = patient_json.get('primary_diagnosis', '')
        age = patient_json.get('age', 50)  # Default age if not provided
        gender = patient_json.get('gender', 'M')  # Default gender if not provided
        comorbidities = patient_json.get('comorbidities', '')
        
        # Get diagnosis-specific feature values or fallback to overall medians
        # This is the key part of our feature bridge
        full_features = {}
        
        # Start with features we directly have
        full_features['age'] = age
        full_features['gender'] = gender
        
        # Add date-related features
        full_features['visit_year'] = visit_date.year
        full_features['visit_month'] = visit_date.month
        full_features['visit_day'] = visit_date.day
        full_features['visit_hour'] = visit_date.hour
        full_features['visit_weekday'] = visit_date.weekday()
        full_features['visit_is_weekend'] = 1 if visit_date.weekday() >= 5 else 0
        
        # Add comorbidity features
        full_features['comorbidities'] = comorbidities
        full_features['num_comorbidities'] = len(comorbidities.split(',')) if comorbidities else 0
        
        # Add diagnosis
        full_features['primary_diagnosis'] = diagnosis
        
        # Add estimated values for all other numeric features based on diagnosis
        if diagnosis in self.features_by_diagnosis:
            diagnosis_features = self.features_by_diagnosis[diagnosis]
            
            for feature, stats in diagnosis_features.items():
                if feature not in full_features:
                    # Use median as it's more robust
                    full_features[feature] = stats['median']
        else:
            # If diagnosis not found, use overall feature statistics
            for feature, stats in self.feature_stats.items():
                if feature not in full_features:
                    full_features[feature] = stats['median']
        
        # Create a DataFrame with all expected columns
        bridged_df = pd.DataFrame([full_features])
        
        # Make sure all expected columns are present
        for col in self.expected_columns:
            if col not in bridged_df.columns:
                # Add with a default value
                if col in self.feature_stats:
                    bridged_df[col] = self.feature_stats[col]['median']
                else:
                    # For categorical features, use empty string
                    bridged_df[col] = ''
        
        # Remove any extra columns
        bridged_df = bridged_df[self.expected_columns]
        
        # Make prediction
        los_prediction = self.full_model.predict(bridged_df)[0]
        los_prediction_rounded = round(los_prediction)
        
        # Calculate estimated discharge date
        discharge_date = visit_date + pd.Timedelta(days=los_prediction_rounded)
        
        # Calculate confidence interval
        std_dev = self.diagnosis_stats['std_los'].get(diagnosis, 2.0) if diagnosis in self.diagnosis_stats.get('std_los', {}) else 2.0
        
        lower_bound = max(1, round(los_prediction - 1.96 * std_dev))
        upper_bound = round(los_prediction + 1.96 * std_dev)
        
        # Create result dictionary
        prediction = {
            'predicted_los': los_prediction_rounded,
            'estimated_discharge_date': discharge_date.strftime('%Y-%m-%d'),
            'earliest_discharge': (visit_date + pd.Timedelta(days=lower_bound)).strftime('%Y-%m-%d'),
            'latest_discharge': (visit_date + pd.Timedelta(days=upper_bound)).strftime('%Y-%m-%d')
        }
        
        return prediction
