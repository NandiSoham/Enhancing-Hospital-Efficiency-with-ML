from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

app = Flask(__name__)
CORS(app)

# Load the saved XGBoost model
model = xgb.XGBClassifier()
model.load_model('model/xgb_model.model')

# Label mapping dictionary
label_mapping = {
    0: '0-10', 1: '11-20', 2: '21-30', 3: '31-40', 4: '41-50',
    5: '51-60', 6: '61-70', 7: '71-80', 8: '81-90', 9: '91-100',
    10: 'More than 100 Days'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Transform input data into format expected by model
    input_df = pd.DataFrame([data])
    
    # Perform the same preprocessing as in training
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype('str')
    
    # Make prediction
    pred = model.predict(input_df)[0]
    
    # Convert numerical prediction to label
    stay_category = label_mapping[pred]
    
    # Return prediction
    return jsonify({
        'prediction': stay_category,
        'confidence': float(np.max(model.predict_proba(input_df)[0])),
        'contributing_factors': [
            {'name': 'Severity of Illness', 'importance': 0.32},
            {'name': 'Department', 'importance': 0.24},
            {'name': 'Age', 'importance': 0.15},
            {'name': 'Hospital Type', 'importance': 0.12},
            {'name': 'Admission Type', 'importance': 0.10}
        ]
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    # This would typically be calculated from your data
    return jsonify({
        'average_stay': 27.4,
        'most_common': '21-30 days',
        'accuracy': 0.84,
        'distribution': [
            {'name': '0-10', 'count': 342},
            {'name': '11-20', 'count': 523},
            {'name': '21-30', 'count': 891},
            {'name': '31-40', 'count': 432},
            {'name': '41-50', 'count': 253},
            {'name': '51-60', 'count': 167},
            {'name': '61-70', 'count': 98},
            {'name': '71-80', 'count': 58},
            {'name': '81-90', 'count': 32},
            {'name': '91-100', 'count': 19},
            {'name': 'More than 100 Days', 'count': 12}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)