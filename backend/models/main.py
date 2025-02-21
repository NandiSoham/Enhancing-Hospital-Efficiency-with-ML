from fastapi import FastAPI, UploadFile, File
from modules import model_inference
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hospital Stay Prediction API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    predictions = model_inference.make_predictions(df)
    return {"predictions": predictions.tolist()}
