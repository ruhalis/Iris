import os
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
import mlflow
import mlflow.pytorch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train():
    params = {
        "lr": 0.01,
        "epochs": 100,
        "hidden1": 16,
        "hidden2": 16,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "test_size": 0.2,
        "random_state": 42,
    }

    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    for epoch in range(params["epochs"]):
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)
        _, predicted = torch.max(predictions, 1)
        y_pred = predicted.numpy()
        y_true = y_test_t.numpy()
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
        print(f"Test F1 (macro): {f1:.4f}")

    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, "model.pth")
    scaler_path = os.path.join(weights_dir, "scaler.joblib")
    metrics_path = os.path.join(weights_dir, "metrics.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    model_name = "IrisMLP (4-16-16-3)"
    with open(metrics_path, "w") as f:
        json.dump(
            {"model_name": model_name, "accuracy": accuracy, "f1_macro": f1}, f, indent=2
        )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-classification")

    try:
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="iris-classifier",
            )
            print(f"Logged run {run.info.run_id} to {tracking_uri}")
    except Exception as e:
        print(f"[warn] MLflow logging skipped: {e}")


if __name__ == "__main__":
    train()
