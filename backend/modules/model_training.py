import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.1, n_estimators=800,
        objective='multi:softmax', reg_alpha=0.5, reg_lambda=1.5,
        booster='gbtree', n_jobs=4, min_child_weight=2, base_score=0.75
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    return round(accuracy, 2)
