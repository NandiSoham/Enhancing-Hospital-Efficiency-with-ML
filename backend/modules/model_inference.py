import joblib
import pandas as pd
from modules.data_preprocessing import handle_missing_values, encode_categorical_columns

MODEL_PATH = 'models/model.json'

def make_predictions(df):
    model = joblib.load(MODEL_PATH)
    df = handle_missing_values(df)
    categorical_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 
                           'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 
                           'Severity of Illness', 'Age']
    df = encode_categorical_columns(df, categorical_columns)
    predictions = model.predict(df)
    return predictions
