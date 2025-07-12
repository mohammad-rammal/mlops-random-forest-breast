import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder

# Load dataset for encoding class labels (only once)
data = pd.read_csv("../data-storage/breast_dataset.csv")
label_class = LabelEncoder()
data["Class"] = label_class.fit_transform(data["Class"])

app = FastAPI()

MODEL_NAME = "BreastCancer_RF"
MODEL_VERSION = "1"

model = None  # will be set at startup

EXPECTED_COLUMNS = [
    'Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 'Marginal_Adhesion',
    'Single_Epi_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'
]

mlflow.set_tracking_uri("http://mlflow:5000")

@app.on_event("startup")
async def load_model_on_startup():
    global model
    if os.getenv("TESTING") == "1":
        print("⚠️ TESTING mode enabled — skipping model load")
        model = None
        return
    for i in range(5):
        try:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
            print(f"✅ Model loaded successfully on attempt {i+1}")
            break
        except MlflowException as e:
            print(f"⏳ [Retry {i+1}/5] Model not found yet: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("❌ Model failed to load after 5 retries.")

@app.get("/")
async def root():
    return {"message": "Welcome to our API page!"}

@app.post("/predict/")
async def predict_cancer(data: dict):
    if model is None:
        return {"error": "Model is not loaded"}
    features = pd.DataFrame([data["features"]], columns=EXPECTED_COLUMNS).astype(np.float64)
    prediction = model.predict(features)
    class_name = label_class.inverse_transform(prediction)[0]
    return {"class": class_name}




# uvicorn backend:app --reload --host 192.168.0.103 --port 8000