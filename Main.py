import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

print("\n----------------------------")
print("Code Execution Started..!")
print("----------------------------")

class GPARegressor(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(33, 66)
        self.fc2 = nn.Linear(66, 66)
        self.fc3 = nn.Linear(66, 33)
        self.fc4 = nn.Linear(33, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.output(x)
        return x

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = pd.read_csv("Impact of AI on Students/preprocessed_data.csv")
X, y = df.drop("Post_Semester_GPA", axis=1), df["Post_Semester_GPA"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

train_df = TensorDataset(X_train, y_train)
test_df = TensorDataset(X_test, y_test)
train_loader = DataLoader(
    train_df,
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    test_df,
    batch_size=32,
    shuffle=False
)

model = GPARegressor().to(device)
criterion = nn.MSELoss()
optimizer = Adam(
    model.parameters(),
    lr = 0.001
)

print("\n----------------------------")
print("Null Value Checkup")
print("----------------------------")
print(f"X_Train : {X_train.isnan().sum()}")
print(f"X_Test  : {X_test.isnan().sum()}")
print(f"Y_Train : {torch.tensor(y_train).isnan().sum()}")
print(f"Y_Test  : {torch.tensor(y_test).isnan().sum()}")
print("----------------------------")

print("\n----------------------------")
print("Model Training ...")
print("----------------------------")
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch : {epoch+1}/{epochs} | Loss : {running_loss/len(train_loader):.4f}")
print("----------------------------")
print("Model Training Completed..!")
print("----------------------------")

print("\n----------------------------")
print("Model Testing & Evaluation ...")
print("----------------------------")
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        predictions.extend(preds.cpu().numpy())
        actuals.extend(y_batch.numpy())
predictions = np.array(predictions)
actuals = np.array(actuals)

mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

print(f"Mean Squared Error     : {mse*100:.4f}")
print(f"Mean Absolute Error    : {mae*100:.4f}")
print(f"Root Mean Square Error : {rmse*100:.4f}")
print(f"R2 Score               : {r2*100:.4f}")
print("----------------------------")
print("Model Testing & Evaluation Completed..!")
print("----------------------------")


print("\n----------------------------")
print("Code Execution Completed..!")
print("----------------------------")