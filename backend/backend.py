import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import time
from mlflow.exceptions import MlflowException

from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder

# Load dataset for encoding class labels
data = pd.read_csv("../data-storage/breast_dataset.csv")

# Encode class labels
label_class = LabelEncoder()
data["Class"] = label_class.fit_transform(data["Class"])

# Initialize FastAPI app
app = FastAPI()

# Set MLflow Tracking URI (internal Docker DNS)
mlflow.set_tracking_uri("http://mlflow:5000")

# Model Registry details
MODEL_NAME = "BreastCancer_RF"
MODEL_VERSION = "1"

# Retry logic to wait for model registration
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

# Health check
@app.get("/")
async def root():
    return {"message": "Welcome to our API page!"}

EXPECTED_COLUMNS = [
    'Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 'Marginal_Adhesion',
    'Single_Epi_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses'
]

# Prediction endpoint
@app.post("/predict/")
async def predict_cancer(data: dict):
    features = pd.DataFrame([data["features"]], columns=EXPECTED_COLUMNS).astype(np.float64)
    prediction = model.predict(features)
    class_name = label_class.inverse_transform(prediction)[0]
    return {"class": class_name}



# uvicorn backend:app --reload --host 192.168.0.103 --port 8000