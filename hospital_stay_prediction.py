
import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Training and Test Data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

# Handle Missing Values
def handle_missing_values(df):
    df['Bed Grade'].fillna(df['Bed Grade'].mode()[0], inplace=True)
    df['City_Code_Patient'].fillna(df['City_Code_Patient'].mode()[0], inplace=True)
    return df

# Encode Categorical Columns
def encode_categorical_columns(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Prepare Data for Modeling
def prepare_data(train, test):
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    le = LabelEncoder()
    train['Stay'] = le.fit_transform(train['Stay'].astype(str))
    test['Stay'] = -1
    
    categorical_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 
                          'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']
    
    train = encode_categorical_columns(train, categorical_columns)
    test = encode_categorical_columns(test, categorical_columns)
    
    return train, test

# Split Data into Training and Testing Sets
def split_data(train):
    X = train.drop('Stay', axis=1)
    y = train['Stay']
    return train_test_split(X, y, test_size=0.20, random_state=100)

# Train XGBoost Model
def train_xgboost(X_train, y_train):
    model = xgboost.XGBClassifier(
        max_depth=4, learning_rate=0.1, n_estimators=800,
        objective='multi:softmax', reg_alpha=0.5, reg_lambda=1.5,
        booster='gbtree', n_jobs=4, min_child_weight=2, base_score=0.75
    )
    model.fit(X_train, y_train)
    return model

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    return round(accuracy, 2)

# Main Execution Flow
if __name__ == "__main__":
    train, test = load_data()
    train, test = prepare_data(train, test)
    X_train, X_test, y_train, y_test = split_data(train)
    model = train_xgboost(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

