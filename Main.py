import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")


class GPARegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.output(x)


df = pd.read_csv("Impact of AI on Students/preprocessed_data.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

X = df.drop("Post_Semester_GPA", axis=1)
y = df["Post_Semester_GPA"]

FEATURE_COLUMNS = list(X.columns)
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("Scaler saved -> scaler.pkl")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values.reshape(-1, 1),  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=32, shuffle=False)

model     = GPARegressor().to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

print("\n--- Training (100 epochs) ---")
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

print("\n--- Evaluation ---")
model.eval()
preds_list, actuals_list = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds_list.extend(model(X_batch.to(device)).cpu().numpy())
        actuals_list.extend(y_batch.numpy())

predictions = np.array(preds_list)
actuals     = np.array(actuals_list)

mse  = mean_squared_error(actuals, predictions)
mae  = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mse)
r2   = r2_score(actuals, predictions)

print(f"  MSE  : {mse*100:.4f}")
print(f"  MAE  : {mae*100:.4f}")
print(f"  RMSE : {rmse*100:.4f}")
print(f"  R2   : {r2*100:.4f}%")

torch.save(model.state_dict(), "model.pt")
print("\nModel saved -> model.pt")
print("Done! Start the server:  uvicorn app:app --reload")
