import pandas as pd
import numpy as np
import joblib
import os

def preprocess_input(input_df):
    """Preprocess user input for prediction"""
    # Create a copy to avoid modifying the original
    processed = input_df.copy()
    
    # Encode categorical columns
    for column in ['Hospital_type_code', 'Hospital_region_code', 'Department', 
                   'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 
                   'Severity of Illness', 'Age']:
        if column in processed.columns:
            encoder_path = os.path.join('models', f'{column}_encoder.pkl')
            if os.path.exists(encoder_path):
                le = joblib.load(encoder_path)
                # Handle unseen categories
                if processed[column].iloc[0] in le.classes_:
                    processed[column] = le.transform(processed[column].astype('str'))
                else:
                    # Use the most common category as a fallback
                    most_common = le.transform([le.classes_[0]])[0]
                    processed[column] = most_common
    
    # Fill missing values
    for column in processed.columns:
        if processed[column].isnull().any():
            if column in ['Bed Grade', 'City_Code_Patient']:
                # These were filled with mode in training
                processed[column].fillna(3, inplace=True)  # Assuming 3 is a common value
            else:
                # For other columns, use 0 as a default
                processed[column].fillna(0, inplace=True)
    
    # Add the feature engineering columns
    # Since we can't do the group by on a single record, use default values
    processed['count_id_patient'] = 1
    processed['count_id_patient_hospitalCode'] = 1
    processed['count_id_patient_wardfacilityCode'] = 1
    
    # Drop columns that were removed in training
    cols_to_drop = ['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code']
    for col in cols_to_drop:
        if col in processed.columns:
            processed = processed.drop(col, axis=1)
    
    return processed