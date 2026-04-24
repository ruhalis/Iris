import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI(title="Iris Classification API")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def _load_info():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return {"model_name": "unknown", "accuracy": None, "f1_macro": None}


@app.get("/")
def root():
    return {"message": "API is ready"}


@app.get("/info")
def info():
    return _load_info()


@app.post("/predict")
def predict_iris(data: IrisInput):
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]
    return predict(features)
