from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict

app = FastAPI(title="Iris Classification API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "API is ready"}

@app.post("/predict")
def predict_iris(data: IrisInput):
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]
    result = predict(features)
    return result
