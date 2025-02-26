import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the train and test data"""
    # Load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # Check for null values
    train['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace=True)
    train['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace=True)
    
    test['Bed Grade'].fillna(test['Bed Grade'].mode()[0], inplace=True)
    test['City_Code_Patient'].fillna(test['City_Code_Patient'].mode()[0], inplace=True)
    
    # Encode 'Stay' column
    le = LabelEncoder()
    train['Stay'] = le.fit_transform(train['Stay'].astype('str'))
    
    # Save the label encoder for later use
    joblib.dump(le, os.path.join('models', 'stay_encoder.pkl'))
    
    # Set test 'Stay' to -1 (placeholder)
    test['Stay'] = -1
    
    # Combine datasets
    df = pd.concat([train, test], ignore_index=True)
    
    # Encode categorical columns
    for column in ['Hospital_type_code', 'Hospital_region_code', 'Department', 
                   'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 
                   'Severity of Illness', 'Age']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype('str'))
        # Save each encoder
        joblib.dump(le, os.path.join('models', f'{column}_encoder.pkl'))
    
    # Feature engineering
    df = create_features(df)
    
    # Split back into train and test
    train = df[df['Stay'] != -1]
    test = df[df['Stay'] == -1]
    
    # Prepare final training data
    train1 = train.drop(['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis=1)
    
    # Prepare features and target
    X = train1.drop('Stay', axis=1)
    y = train1['Stay']
    
    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
    
    return X_train, X_test, y_train, y_test, test

def create_features(df):
    """Create additional features for the model"""
    # Helper function for count encoding
    def get_countid_encode(data, cols, name):
        temp = data.groupby(cols)['case_id'].count().reset_index().rename(columns={'case_id': name})
        data = pd.merge(data, temp, how='left', on=cols)
        data[name] = data[name].astype('float')
        data[name].fillna(np.median(temp[name]), inplace=True)
        return data
    
    # Apply count encoding
    df = get_countid_encode(df, ['patientid'], name='count_id_patient')
    df = get_countid_encode(df, ['patientid', 'Hospital_region_code'], name='count_id_patient_hospitalCode')
    df = get_countid_encode(df, ['patientid', 'Ward_Facility_Code'], name='count_id_patient_wardfacilityCode')
    
    return df

def train_model():
    """Train an XGBoost model and return it"""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    # Initialize model
    xgb_classifier = xgb.XGBClassifier(
        max_depth=4, 
        learning_rate=0.1, 
        n_estimators=800,
        objective='multi:softmax', 
        reg_alpha=0.5, 
        reg_lambda=1.5,
        booster='gbtree', 
        n_jobs=4, 
        min_child_weight=2, 
        base_score=0.75
    )
    
    # Train model
    model = xgb_classifier.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save accuracy for reference
    with open(os.path.join('models', 'model_accuracy.txt'), 'w') as f:
        f.write(str(accuracy))
    
    return model

def predict_stay(model, input_data):
    """Make predictions with the trained model"""
    # Get predictions
    prediction_idx = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Map back to original labels
    label_encoder = joblib.load(os.path.join('models', 'stay_encoder.pkl'))
    mapped_label = label_encoder.inverse_transform([prediction_idx])[0]
    
    # Get the probability for the predicted class
    probability = probabilities[prediction_idx]
    
    return mapped_label, float(probability)

if __name__ == "__main__":
    # If this file is run directly, train and save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model = train_model()
    joblib.dump(model, os.path.join('models', 'xgb_model.pkl'))
    print("Model trained and saved successfully!")