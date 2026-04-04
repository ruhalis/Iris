import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    iris = load_iris()
    X, y = iris.data, iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
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
        accuracy = (predicted == y_test_t).sum().item() / len(y_test_t)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    joblib.dump(scaler, "scaler.joblib")

if __name__ == "__main__":
    train()
