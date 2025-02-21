import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def handle_missing_values(df):
    df['Bed Grade'].fillna(df['Bed Grade'].mode()[0], inplace=True)
    df['City_Code_Patient'].fillna(df['City_Code_Patient'].mode()[0], inplace=True)
    return df

def encode_categorical_columns(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def prepare_data(train, test):
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    le = LabelEncoder()
    train['Stay'] = le.fit_transform(train['Stay'].astype(str))
    test['Stay'] = -1
    
    categorical_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 
                           'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 
                           'Severity of Illness', 'Age']
    
    train = encode_categorical_columns(train, categorical_columns)
    test = encode_categorical_columns(test, categorical_columns)
    
    X = train.drop('Stay', axis=1)
    y = train['Stay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
    
    return X_train, X_test, y_train, y_test, test
