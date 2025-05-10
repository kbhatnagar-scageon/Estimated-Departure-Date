"""
Hospital Length of Stay Prediction Module

This module contains all functionality for training and using the hospital LOS prediction model.
It's extracted from the original Jupyter notebook and organized for use in production applications.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the class for hospital LOS prediction
class HospitalLOSPredictor:
    """
    End-to-end ML pipeline for predicting hospital length of stay
    """
    def __init__(self):
        self.preprocessor = None
        self.feature_selector = None
        self.model = None
        self.selected_features = None
        self.top_comorbidities = None
        self.diagnosis_stats = None
        self.model_type = None
        self.expected_columns = None
        
    def fit(self, db_connection_params=None, data=None):
        """
        Train the entire pipeline
        
        Parameters:
        -----------
        db_connection_params : dict
            Parameters for connecting to the database
        data : pandas DataFrame
            Data for training the model (if not loading from database)
        """
        # Load data
        print("Loading data...")
        if data is not None:
            df = data
        elif db_connection_params is not None:
            # Connect to database (implementation details omitted for brevity)
            # ...
            pass
        else:
            raise ValueError("Either db_connection_params or data must be provided")
        
        print(f"Loaded {len(df)} records")
        
        # Preprocess data
        print("Preprocessing data...")
        # Extract comorbidities
        self.top_comorbidities = self._extract_top_comorbidities(df, n=10)
        
        # Extract diagnosis stats
        self.diagnosis_stats = self._create_diagnosis_stats(df)
        
        # Process time fields if they exist
        time_cols = ['laboratory_report_time', 'pharmacy_billing_time', 
                    'insurance_claim_settlement_time', 'discharge_summary_time']
        
        for col in time_cols:
            if col in df.columns:
                df[f'{col}_hours'] = df[col].apply(self._time_to_hours)
                df = df.drop(columns=[col])
        
        # Create binary flags for comorbidities
        if 'comorbidities' in df.columns:
            for comorbidity in self.top_comorbidities:
                df[f'has_{comorbidity.lower().replace(" ", "_")}'] = df['comorbidities'].apply(
                    lambda x: 1 if comorbidity in str(x) else 0
                )
        
        # Encode categorical variables
        if 'gender' in df.columns:
            df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0})
            df = df.drop(columns=['gender'])
        
        if 'is_surgical' in df.columns:
            df['is_surgical_encoded'] = df['is_surgical'].map({'Yes': 1, 'No': 0})
            df = df.drop(columns=['is_surgical'])
        
        # Add diagnosis features
        if 'primary_diagnosis' in df.columns:
            df['diagnosis_avg_los'] = df['primary_diagnosis'].map(self.diagnosis_stats['avg_los'])
            df['diagnosis_std_los'] = df['primary_diagnosis'].map(self.diagnosis_stats['std_los'])
            df['diagnosis_frequency'] = df['primary_diagnosis'].map(self.diagnosis_stats['frequency'])
        
        # Create interaction features
        if 'age' in df.columns and 'severity_score' in df.columns:
            df['age_severity'] = df['age'] * df['severity_score']
        
        if 'age' in df.columns and 'num_comorbidities' in df.columns:
            df['age_comorbidities'] = df['age'] * df['num_comorbidities']
        
        # Define features and target
        print("Defining features and target...")
        # Make sure actual_los and mrn exist in the dataframe
        if 'actual_los' not in df.columns or 'mrn' not in df.columns:
            raise ValueError("DataFrame must contain 'actual_los' and 'mrn' columns")
        
        X = df.drop(columns=['actual_los', 'mrn'])
        y = df['actual_los']
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        # Separate numerical and categorical features
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numerical_features = [col for col in X.columns if X[col].dtype != 'object' and col != 'gender_encoded']
        binary_features = ['gender_encoded'] if 'gender_encoded' in X.columns else []
        binary_features += [col for col in X.columns if col.startswith('has_') or col == 'is_surgical_encoded']
        
        # Create preprocessing pipeline
        print("Creating preprocessing pipeline...")
        transformers = []
        
        if numerical_features:
            transformers.append(('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features))
        
        if categorical_features:
            transformers.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features))
        
        if binary_features:
            transformers.append(('bin', 'passthrough', binary_features))
        
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        # Fit and transform the data
        print("Fitting preprocessor...")
        X_train_preprocessed = self.preprocessor.fit_transform(X_train)
        X_val_preprocessed = self.preprocessor.transform(X_val)
        
        # Feature selection
        print("Performing feature selection...")
        self.feature_selector = SelectKBest(mutual_info_regression, k=min(15, X_train_preprocessed.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_preprocessed, y_train)
        X_val_selected = self.feature_selector.transform(X_val_preprocessed)
        
        # Get approximate feature names (this is simplified)
        self.selected_features = [f"feature_{i}" for i in range(X_train_selected.shape[1])]
        
        # Transform test set
        X_test_preprocessed = self.preprocessor.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_preprocessed)
        
        # Train and evaluate models
        print("Training and evaluating models...")
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train_selected, y_train)
            trained_models[name] = model
            
            # Make predictions on validation set
            y_pred = model.predict(X_val_selected)
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            
            # Store results
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Sort models by MAE (lower is better)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['MAE'])
        
        print("\nModels ranked by MAE:")
        for i, (name, metrics) in enumerate(sorted_results, 1):
            print(f"{i}. {name} - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.4f}")
        
        # Select the best model
        best_model_name, best_metrics = sorted_results[0]
        self.model_type = best_model_name
        self.model = trained_models[best_model_name]
        
        print(f"\n=== SELECTED MODEL: {best_model_name} ===")
        print(f"Best model metrics - MAE: {best_metrics['MAE']:.2f}, RMSE: {best_metrics['RMSE']:.2f}, R2: {best_metrics['R2']:.4f}")

        # Final evaluation on test set
        print("Evaluating final model on test set...")
        y_pred = self.model.predict(X_test_selected)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Final model performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Calculate discharge date accuracy
        predictions_within_1_day = sum(abs(y_test - y_pred) <= 1) / len(y_test) * 100
        predictions_within_2_days = sum(abs(y_test - y_pred) <= 2) / len(y_test) * 100
        predictions_within_3_days = sum(abs(y_test - y_pred) <= 3) / len(y_test) * 100
        
        print(f"Predictions within 1 day: {predictions_within_1_day:.2f}%")
        print(f"Predictions within 2 days: {predictions_within_2_days:.2f}%")
        print(f"Predictions within 3 days: {predictions_within_3_days:.2f}%")
        
        # Store the expected column list
        self.expected_columns = X.columns.tolist()
        
        return self
    
    def predict_from_json(self, patient_json):
        """
        Predict the length of stay for a new patient using JSON input
        
        Parameters:
        -----------
        patient_json : dict or str
            Patient data as a dictionary or JSON string
            
        Returns:
        --------
        prediction : dict
            Dictionary containing the predicted length of stay and estimated discharge date
        """
        # Convert JSON string to dictionary if needed
        if isinstance(patient_json, str):
            patient_json = json.loads(patient_json)
        
        # Create DataFrame from patient data
        patient_df = pd.DataFrame([patient_json])
        
        # Preprocess the data
        processed_data = self._preprocess_patient_data(patient_df)
        
        # Feature engineering
        processed_data_array = self.preprocessor.transform(processed_data)
        processed_data_selected = self.feature_selector.transform(processed_data_array)
        
        # Make prediction
        los_prediction = self.model.predict(processed_data_selected)[0]
        
        # Round to nearest integer
        los_prediction_rounded = round(los_prediction)
        
        # Calculate estimated discharge date
        visit_date = pd.to_datetime(patient_json.get('visit_date'))
        estimated_discharge_date = visit_date + pd.Timedelta(days=los_prediction_rounded)
        
        # Calculate confidence interval based on historical data for the diagnosis
        diagnosis = patient_json.get('primary_diagnosis')
        std_dev = 2.0  # Default standard deviation
        
        if diagnosis in self.diagnosis_stats['std_los']:
            std_dev = self.diagnosis_stats['std_los'][diagnosis]
        
        lower_bound = max(1, round(los_prediction - 1.96 * std_dev))
        upper_bound = round(los_prediction + 1.96 * std_dev)
        
        # Create prediction result
        result = {
            'predicted_los': los_prediction_rounded,
            'estimated_discharge_date': estimated_discharge_date.strftime('%Y-%m-%d'),
            'confidence_interval': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'earliest_discharge': (visit_date + pd.Timedelta(days=lower_bound)).strftime('%Y-%m-%d'),
            'latest_discharge': (visit_date + pd.Timedelta(days=upper_bound)).strftime('%Y-%m-%d')
        }
        
        return result
    
    def _preprocess_patient_data(self, patient_df):
        """
        Preprocess patient data for prediction
        
        Parameters:
        -----------
        patient_df : pandas DataFrame
            DataFrame containing patient data
        
        Returns:
        --------
        processed_df : pandas DataFrame
            Processed DataFrame ready for prediction
        """
        # Create a copy
        df = patient_df.copy()
        
        # Convert date columns to datetime
        if 'visit_date' in df.columns:
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            
            # Extract datetime features
            df['visit_year'] = df['visit_date'].dt.year
            df['visit_month'] = df['visit_date'].dt.month
            df['visit_day'] = df['visit_date'].dt.day
            df['visit_hour'] = df['visit_date'].dt.hour
            df['visit_weekday'] = df['visit_date'].dt.weekday
            df['visit_is_weekend'] = df['visit_weekday'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Remove the datetime column to avoid issues
            df = df.drop(columns=['visit_date'])
        
        # Handle missing fields with appropriate defaults
        available_fields = set(df.columns)
        essential_fields = {
            'age': 55,                # Median age as fallback
            'gender_encoded': 1,      # Default to male
            'severity_score': 3,      # Default to medium severity
            'is_surgical_encoded': 0, # Default to non-surgical
            'num_comorbidities': 0,   # Default to no comorbidities
            'ward_occupancy_pct': 80  # Default ward occupancy
        }
        
        # Add default values for missing essential fields
        for field, default_value in essential_fields.items():
            base_field = field.replace('_encoded', '')
            if base_field in available_fields and field not in available_fields:
                # Handle encoding for gender and is_surgical
                if field == 'gender_encoded':
                    df[field] = df[base_field].map({'M': 1, 'F': 0})
                    df = df.drop(columns=[base_field])
                elif field == 'is_surgical_encoded':
                    df[field] = df[base_field].map({'Yes': 1, 'No': 0})
                    df = df.drop(columns=[base_field])
            elif field not in available_fields:
                df[field] = default_value
        
        # Handle comorbidities
        if 'comorbidities' in available_fields:
            if 'num_comorbidities' not in available_fields:
                df['num_comorbidities'] = df['comorbidities'].apply(
                    lambda x: 0 if x == 'None' or pd.isna(x) else len(str(x).split(','))
                )
            
            # Create binary flags for top comorbidities
            for comorbidity in self.top_comorbidities:
                col_name = f'has_{comorbidity.lower().replace(" ", "_")}'
                df[col_name] = df['comorbidities'].apply(
                    lambda x: 1 if comorbidity in str(x) else 0
                )
            
            # We've extracted what we need, so drop the original column
            df = df.drop(columns=['comorbidities'])
        else:
            df['num_comorbidities'] = 0
            for comorbidity in self.top_comorbidities:
                col_name = f'has_{comorbidity.lower().replace(" ", "_")}'
                df[col_name] = 0
        
        # Handle diagnosis-specific features
        if 'primary_diagnosis' in available_fields:
            diagnosis = df['primary_diagnosis'].iloc[0]
            
            # Map diagnosis to its statistics
            df['diagnosis_avg_los'] = self.diagnosis_stats['avg_los'].get(diagnosis, 9.57)  # Default to average LOS
            df['diagnosis_std_los'] = self.diagnosis_stats['std_los'].get(diagnosis, 5.0)   # Default std deviation
            df['diagnosis_frequency'] = self.diagnosis_stats['frequency'].get(diagnosis, 1) # Default frequency
        else:
            # Use overall average if diagnosis not provided
            df['diagnosis_avg_los'] = 9.57
            df['diagnosis_std_los'] = 5.0
            df['diagnosis_frequency'] = 1
        
        # Handle vital signs with default values if missing
        vital_signs = {
            'heart_rate': 85,
            'systolic_bp': 120,
            'temperature': 37.5,
            'oxygen_saturation': 95
        }
        
        for field, default_value in vital_signs.items():
            if field not in available_fields:
                df[field] = default_value
        
        # Handle time metrics with default values if missing
        time_fields = {
            'laboratory_report_time_hours': 24,
            'pharmacy_billing_time_hours': 12,
            'insurance_claim_settlement_time_hours': 72,
            'discharge_summary_time_hours': 7
        }
        
        for field, default_value in time_fields.items():
            base_field = field.replace('_hours', '')
            
            if base_field in available_fields:
                df[field] = df[base_field].apply(self._time_to_hours)
                df = df.drop(columns=[base_field])
            elif field not in available_fields:
                df[field] = default_value
        
        # Create interaction features
        if 'age' in df.columns and 'severity_score' in df.columns:
            df['age_severity'] = df['age'] * df['severity_score']
        if 'age' in df.columns and 'num_comorbidities' in df.columns:
            df['age_comorbidities'] = df['age'] * df['num_comorbidities']
        
        return df
    
    def _time_to_hours(self, time_str):
        """Convert time string in format HH:MM:SS to decimal hours"""
        try:
            parts = time_str.split(':')
            if len(parts) != 3:
                return 0
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours + minutes/60.0 + seconds/3600.0
        except Exception:
            return 0  # Default to 0 if there's an error
    
    def _extract_top_comorbidities(self, df, n=10):
        """Extract the top n most common comorbidities from the dataset"""
        if 'comorbidities' not in df.columns:
            return []
            
        # Initialize counter for comorbidities
        comorbidity_counts = {}
        
        # Count each comorbidity
        for comorbidities in df['comorbidities']:
            if comorbidities == 'None' or pd.isna(comorbidities):
                continue
                
            for comorbidity in str(comorbidities).split(','):
                comorbidity = comorbidity.strip()
                if comorbidity not in comorbidity_counts:
                    comorbidity_counts[comorbidity] = 0
                comorbidity_counts[comorbidity] += 1
        
        # Sort comorbidities by frequency
        sorted_comorbidities = sorted(comorbidity_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top n comorbidities
        return [item[0] for item in sorted_comorbidities[:n]]
    
    def _create_diagnosis_stats(self, df):
        """Create detailed statistics for each diagnosis"""
        if 'primary_diagnosis' not in df.columns or 'actual_los' not in df.columns:
            # Return default empty stats if required columns are missing
            return {
                'avg_los': {},
                'median_los': {},
                'std_los': {},
                'min_los': {},
                'max_los': {},
                'frequency': {},
                'avg_severity': {}
            }
            
        # Group by diagnosis and calculate statistics
        diagnosis_groups = df.groupby('primary_diagnosis')['actual_los']
        
        # Calculate various statistics
        avg_los = diagnosis_groups.mean().to_dict()
        median_los = diagnosis_groups.median().to_dict()
        std_los = diagnosis_groups.std().fillna(5.0).to_dict()  # Fill NaN with default
        min_los = diagnosis_groups.min().to_dict()
        max_los = diagnosis_groups.max().to_dict()
        frequency = diagnosis_groups.count().to_dict()
        
        # Calculate average severity by diagnosis if available
        avg_severity = {}
        if 'severity_score' in df.columns:
            avg_severity = df.groupby('primary_diagnosis')['severity_score'].mean().to_dict()
        
        # Create a comprehensive mapping
        diagnosis_stats = {
            'avg_los': avg_los,
            'median_los': median_los,
            'std_los': std_los,
            'min_los': min_los,
            'max_los': max_los,
            'frequency': frequency,
            'avg_severity': avg_severity
        }
        
        return diagnosis_stats
    
    def save_model(self, output_path):
        """
        Save the trained model and preprocessing components
        
        Parameters:
        -----------
        output_path : str
            Path to save the model
        """
        model_data = {
            'preprocessor': self.preprocessor,
            'feature_selector': self.feature_selector,
            'model': self.model,
            'selected_features': self.selected_features,
            'top_comorbidities': self.top_comorbidities,
            'diagnosis_stats': self.diagnosis_stats,
            'model_type': self.model_type,
            'expected_columns': self.expected_columns
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {output_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            # If the saved data is a dictionary
            self.preprocessor = model_data['preprocessor']
            self.feature_selector = model_data['feature_selector']
            self.model = model_data['model']
            self.selected_features = model_data['selected_features']
            self.top_comorbidities = model_data['top_comorbidities']
            self.diagnosis_stats = model_data['diagnosis_stats']
            self.model_type = model_data.get('model_type', 'Unknown')
            self.expected_columns = model_data.get('expected_columns', [])
        elif isinstance(model_data, HospitalLOSPredictor):
            # If the saved data is an instance of this class
            self.preprocessor = model_data.preprocessor
            self.feature_selector = model_data.feature_selector
            self.model = model_data.model
            self.selected_features = model_data.selected_features
            self.top_comorbidities = model_data.top_comorbidities
            self.diagnosis_stats = model_data.diagnosis_stats
            self.model_type = model_data.model_type
            self.expected_columns = model_data.expected_columns
        
        print("Model loaded successfully")
        
        return self


# Feature Bridge Class for API usage
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
        # Compute statistics for each numeric feature grouped by diagnosis
        # This will help us predict missing features based on diagnosis
        features_by_diagnosis = {}
        
        if 'primary_diagnosis' in df.columns:
            # Get a list of all numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            # Exclude certain columns from feature estimation
            exclude_cols = ['mrn', 'visit_id', 'actual_los']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Compute statistics for each feature by diagnosis
            diagnoses = df['primary_diagnosis'].unique()
            for diagnosis in diagnoses:
                diagnosis_df = df[df['primary_diagnosis'] == diagnosis]
                features_by_diagnosis[diagnosis] = {}
                
                for col in numeric_cols:
                    if col in diagnosis_df.columns:
                        features_by_diagnosis[diagnosis][col] = {
                            'mean': diagnosis_df[col].mean(),
                            'median': diagnosis_df[col].median(),
                            'std': diagnosis_df[col].std()
                        }
        
        # Also compute general statistics for each feature
        feature_stats = {}
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            if col in df.columns:
                feature_stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                }
        
        # Compute diagnosis-specific statistics
        diagnosis_stats = {}
        if 'primary_diagnosis' in df.columns:
            diagnosis_groups = df.groupby('primary_diagnosis')['actual_los']
            diagnosis_stats = {
                'avg_los': diagnosis_groups.mean().to_dict(),
                'median_los': diagnosis_groups.median().to_dict(),
                'std_los': diagnosis_groups.std().fillna(2.0).to_dict(),
                'count': diagnosis_groups.count().to_dict()
            }
        
        # Store the statistics
        self.feature_stats = feature_stats
        self.features_by_diagnosis = features_by_diagnosis
        self.diagnosis_stats = diagnosis_stats
        
        # Train a model on the full data
        # Define features and target
        X = df.drop(columns=['actual_los', 'mrn'] if 'mrn' in df.columns else ['actual_los'])
        y = df['actual_los']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Identify categorical and numerical columns
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numerical_features = [col for col in X.columns if col not in categorical_features]
        
        print(f"Training with {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ],
            remainder='drop'
        )
        
        # Define models to evaluate
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Dictionary to store model results
        results = {}
        trained_models = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline
            
            # Make predictions on test set
            y_pred = pipeline.predict(X_test)
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
        # Find the best model
        sorted_models = sorted(results.items(), key=lambda x: x[1]['MAE'])
        best_model_name, best_metrics = sorted_models[0]
        
        print("\\nModels ranked by MAE:")
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"{i}. {name} - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.4f}")
        
        print(f"\\n=== SELECTED MODEL: {best_model_name} ===")
        print(f"Best model metrics - MAE: {best_metrics['MAE']:.2f}, RMSE: {best_metrics['RMSE']:.2f}, R2: {best_metrics['R2']:.4f}")
        
        # Store the best model
        self.full_model = trained_models[best_model_name]
        self.model_type = best_model_name
        
        # Store the expected column list
        self.expected_columns = X.columns.tolist()
        
        return self
    
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
        full_features['num_comorbidities'] = len(comorbidities.split(',')) if comorbidities and comorbidities != 'None' else 0
        
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

# Helper functions for training and using the model
def train_model_from_data(data_path, output_path=None):
    """
    Train a model from CSV/Excel data
    
    Parameters:
    -----------
    data_path : str
        Path to the data file (CSV or Excel)
    output_path : str
        Path to save the model pickle (default: 'hospital_los_model.pkl')
    
    Returns:
    --------
    predictor : FeatureBridgePredictor
        Trained predictor
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = 'hospital_los_model.pkl'
        
    print(f"Loading data from {data_path}...")
    
    # Load data based on file extension
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    print(f"Loaded {len(df)} records")
    
    # Train the model
    predictor = FeatureBridgePredictor()
    predictor.fit(data=df)
    
    # Save the model
    predictor.save_model(output_path)
    
    return predictor

def make_prediction(model_path, patient_json):
    """
    Make a prediction using a saved model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model pickle
    patient_json : dict
        Patient data as a dictionary
    
    Returns:
    --------
    prediction : dict
        Prediction results
    """
    # Load the model
    predictor = FeatureBridgePredictor()
    
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        if isinstance(loaded_model, FeatureBridgePredictor):
            predictor = loaded_model
        else:
            print("Warning: Loaded object is not a FeatureBridgePredictor instance")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Make prediction
    result = predictor.predict_from_json(patient_json)
    
    return result

# Script execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hospital Length of Stay Prediction')
    parser.add_argument('--train', help='Train model with provided data file path')
    parser.add_argument('--predict', help='Make prediction with JSON file path')
    parser.add_argument('--model', default='hospital_los_model.pkl', help='Model file path (default: hospital_los_model.pkl)')
    
    args = parser.parse_args()
    
    if args.train:
        # Train model with provided data
        train_model_from_data(args.train, args.model)
    
    elif args.predict:
        # Load prediction JSON
        try:
            with open(args.predict, 'r') as f:
                patient_data = json.load(f)
            
            # Make prediction
            result = make_prediction(args.model, patient_data)
            
            if result:
                print("\nPrediction Result:")
                print(f"Predicted Length of Stay: {result['predicted_los']} days")
                print(f"Expected Discharge Date: {result['estimated_discharge_date']}")
                print(f"Earliest Possible Discharge: {result['earliest_discharge']}")
                print(f"Latest Possible Discharge: {result['latest_discharge']}")
        
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
    
    else:
        print("Please provide either --train or --predict argument")
        print("Example usage:")
        print("  Train: python hospital_los_module.py --train data.csv --model model.pkl")
        print("  Predict: python hospital_los_module.py --predict patient.json --model model.pkl")