# Save as backend/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(data):
    """Preprocess input data to match the format expected by the model"""
    # Create a dataframe from the input data
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    categorical_cols = ['hospital_type_code', 'department', 'ward_type', 
                         'type_of_admission', 'severity_of_illness', 'age']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype('str'))
    
    # Return preprocessed data
    return df

if __name__ == "__main__":
    # Test the preprocessing function
    sample_data = {
        'hospital_type_code': 'a',
        'department': 'TB & Chest disease',
        'ward_type': 'R',
        'type_of_admission': 'Emergency',
        'severity_of_illness': 'Moderate',
        'age': 'Adult',
        'bed_grade': 2,
        'city_code_patient': 7,
        'visitors_with_patient': 2,
        'admission_deposit': 4500
    }
    
    preprocessed = preprocess_input(sample_data)
    print("Preprocessed data:")
    print(preprocessed)