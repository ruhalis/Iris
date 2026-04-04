import torch
import joblib
import numpy as np
from train import Model

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

def predict(features: list[float]) -> dict:
    scaler = joblib.load("scaler.joblib")
    model = Model()
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    with torch.no_grad():
        output = model(X_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    class_idx = predicted.item()
    return {
        "predicted_class": class_idx,
        "predicted_label": CLASS_NAMES[class_idx],
        "probabilities": {
            name: round(prob, 4)
            for name, prob in zip(CLASS_NAMES, probabilities[0].tolist())
        },
    }


if __name__ == "__main__":
    # Example: predict for a sample iris flower
    sample = [5.1, 3.5, 1.4, 0.2]
    result = predict(sample)
    print(f"Input features: {sample}")
    print(f"Prediction: {result['predicted_label']} (class {result['predicted_class']})")
    print(f"Probabilities: {result['probabilities']}")
