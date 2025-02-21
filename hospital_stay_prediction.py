import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Step 1: Load Training and Test Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Step 2: Check Missing Values
print("Missing values in train dataset:")
print(train.isnull().sum())
print("\nMissing values in test dataset:")
print(test.isnull().sum())

# Step 3: Handle Missing Values by Filling with Mode
def fill_missing_values(df, column):
    df[column].fillna(df[column].mode()[0], inplace=True)
    
fill_missing_values(train, 'Bed Grade')
fill_missing_values(train, 'City_Code_Patient')
fill_missing_values(test, 'Bed Grade')
fill_missing_values(test, 'City_Code_Patient')

# Step 4: Encode 'Stay' Column in Train Data
le = LabelEncoder()
train['Stay'] = le.fit_transform(train['Stay'].astype(str))

# Step 5: Assign -1 to 'Stay' in Test Data
test['Stay'] = -1

# Step 6: Merge Train and Test Data
df = pd.concat([train, test], ignore_index=True)

# Step 7: Label Encode Categorical Columns
categorical_cols = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 
                    'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Step 8: Separate Train and Test Data Again
train = df[df['Stay'] != -1]
test = df[df['Stay'] == -1]

# Step 9: Feature Engineering - Counting Unique Identifiers
def add_count_feature(train, test, group_cols, new_col_name):
    temp_train = train.groupby(group_cols)['case_id'].count().reset_index().rename(columns={'case_id': new_col_name})
    temp_test = test.groupby(group_cols)['case_id'].count().reset_index().rename(columns={'case_id': new_col_name})
    
    train = train.merge(temp_train, how='left', on=group_cols)
    test = test.merge(temp_test, how='left', on=group_cols)
    
    median_value = np.median(temp_train[new_col_name])
    train[new_col_name].fillna(median_value, inplace=True)
    test[new_col_name].fillna(median_value, inplace=True)
    
    return train, test

train, test = add_count_feature(train, test, ['patientid'], 'count_id_patient')
train, test = add_count_feature(train, test, ['patientid', 'Hospital_region_code'], 'count_id_patient_hospitalCode')
train, test = add_count_feature(train, test, ['patientid', 'Ward_Facility_Code'], 'count_id_patient_wardfacilityCode')

# Step 10: Drop Unnecessary Columns
test1 = test.drop(['Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis=1)
train1 = train.drop(['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis=1)

# Step 11: Split Data for Training and Testing
X1 = train1.drop('Stay', axis=1)
y1 = train1['Stay']

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=100)

# Step 12: Train XGBoost Classifier
classifier_xgb = xgb.XGBClassifier(
    max_depth=4, learning_rate=0.1, n_estimators=800,
    objective='multi:softmax', reg_alpha=0.5, reg_lambda=1.5,
    booster='gbtree', n_jobs=4, min_child_weight=2, base_score=0.75
)

model_xgb = classifier_xgb.fit(X_train, y_train)
prediction_xgb = model_xgb.predict(X_test)
acc_score_xgb = round(accuracy_score(prediction_xgb, y_test), 2)
print(f"XGBoost Model Accuracy: {acc_score_xgb}")

# Step 13: Make Predictions on Test Data
label_mapping = {
    0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50',
    5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100',
    10: 'More than 100 Days'
}

pred_xgb = classifier_xgb.predict(test1.iloc[:, 1:])
result_xgb = pd.DataFrame({'case_id': test1['case_id'], 'Stay': pred_xgb})
result_xgb['Stay'] = result_xgb['Stay'].replace(label_mapping)

# Step 14: Group and Count Unique 'Stay' Values
result = result_xgb.groupby('Stay')['case_id'].nunique()
print("\nCount of unique case_ids per Stay category:")
print(result)
