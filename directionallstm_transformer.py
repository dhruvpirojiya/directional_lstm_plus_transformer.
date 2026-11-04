import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


TICKER = "TSLA"
df = yf.download(
    TICKER,
    period="7d",
    interval="1m",
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
print(f"Downloaded rows: {len(df)}")
print(df.tail())

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
CLOSE_INDEX = FEATURES.index("Close")
data = df[FEATURES].values.astype(np.float32)


split = int(len(data) * 0.8)
scaler = MinMaxScaler()
scaler.fit(data[:split])
scaled_data = scaler.transform(data)

close_min = scaler.data_min_[CLOSE_INDEX]
close_range = scaler.data_range_[CLOSE_INDEX]

# ------------------------------------------------------------------
# 3. Sequence builders
# ------------------------------------------------------------------
def create_diff_sequences(data, seq_len, horizon, target_start, target_end):
    """
    X  : past seq_len minutes of OHLCV
    y  : future horizon closing-price differences relative to the
         last close inside the input window
    """
    X, y = [], []
    for i in range(max(seq_len, target_start), target_end - horizon + 1):
        X.append(data[i - seq_len : i])
        last_close = data[i - 1, CLOSE_INDEX]
        y.append(data[i : i + horizon, CLOSE_INDEX] - last_close)
    return np.array(X), np.array(y)


def create_absolute_sequences(data, seq_len, horizon, target_start, target_end):
    """Used only to recover ground-truth absolute prices for metrics."""
    X, y = [], []
    for i in range(max(seq_len, target_start), target_end - horizon + 1):
        X.append(data[i - seq_len : i])
        y.append(data[i : i + horizon, CLOSE_INDEX])
    return np.array(X), np.array(y)


class BiLSTMTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, horizon=5):
        super().__init__()

        # Bidirectional LSTM: 2 layers, 64 units each direction
        # Output at every time step → 128 features
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Project 128 → 64 before the Transformer
        self.projection = nn.Linear(hidden_dim * 2, 64)

        # Two-layer Transformer encoder
        # d_model=64, 4 heads (16 dim each), FFN=128, GELU
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Final head: take last time step → horizon predictions
        self.output = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, horizon)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x, _ = self.bilstm(x)          # → (batch, seq_len, 128)
        x = self.projection(x)         # → (batch, seq_len, 64)
        x = self.transformer(x)        # → (batch, seq_len, 64)
        return self.output(x[:, -1, :])  # → (batch, horizon)

def train_model(model, X_train, y_train, epochs, lr, weight_decay=1e-4,
                batch_size=64, patience=None, val_ratio=0.1):
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    if patience is not None:
        # chronological validation split from the end of training windows
        n = len(dataset)
        val_size = int(n * val_ratio)
        train_size = n - val_size
        train_ds = torch.utils.data.Subset(dataset, range(train_size))
        val_ds   = torch.utils.data.Subset(dataset, range(train_size, n))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)

        avg_train = total / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_total += criterion(model(xb), yb).item() * len(xb)
            avg_val = val_total / len(val_loader.dataset)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d} | train {avg_train:.6f} | val {avg_val:.6f}")

            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_state)
                    break
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{epochs} | loss {avg_train:.6f}")

    return model

def evaluate(model, X_test, y_test_abs, horizon_name=""):
    model.eval()
    with torch.no_grad():
        pred_diff = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    last_close = X_test[:, -1, CLOSE_INDEX][:, np.newaxis]
    pred_scaled = pred_diff + last_close
    pred_usd = pred_scaled * close_range + close_min
    true_usd = y_test_abs * close_range + close_min

    mae  = mean_absolute_error(true_usd.ravel(), pred_usd.ravel())
    rmse = np.sqrt(mean_squared_error(true_usd.ravel(), pred_usd.ravel()))
    mape = np.mean(np.abs((true_usd.ravel() - pred_usd.ravel()) / true_usd.ravel())) * 100
    r2   = r2_score(true_usd.ravel(), pred_usd.ravel())

    print(f"\n=== {horizon_name} ===")
    print(f"MAE  : ${mae:.4f}")
    print(f"RMSE : ${rmse:.4f}")
    print(f"MAPE : {mape:.4f}%")
    print(f"R²   : {r2:.4f}")
    return pred_usd, true_usd


SEQ_LEN = 120


HORIZON = 5
X_tr, y_tr = create_diff_sequences(scaled_data, SEQ_LEN, HORIZON, SEQ_LEN, split)
X_te, _    = create_diff_sequences(scaled_data, SEQ_LEN, HORIZON, split, len(data))
_, y_te_abs = create_absolute_sequences(scaled_data, SEQ_LEN, HORIZON, split, len(data))

model5 = BiLSTMTransformer(input_dim=5, horizon=HORIZON).to(DEVICE)
print("\nTraining 5-step model ...")
model5 = train_model(model5, X_tr, y_tr, epochs=30, lr=1e-3)
evaluate(model5, X_te, y_te_abs, "5-minute horizon")


HORIZON = 30
X_tr, y_tr = create_diff_sequences(scaled_data, SEQ_LEN, HORIZON, SEQ_LEN, split)
X_te, _    = create_diff_sequences(scaled_data, SEQ_LEN, HORIZON, split, len(data))
_, y_te_abs = create_absolute_sequences(scaled_data, SEQ_LEN, HORIZON, split, len(data))

model30 = BiLSTMTransformer(input_dim=5, horizon=HORIZON).to(DEVICE)
print("\nTraining 30-step model ...")
model30 = train_model(model30, X_tr, y_tr, epochs=15, lr=5e-4, weight_decay=1e-4)
pred30, true30 = evaluate(model30, X_te, y_te_abs, "30-minute horizon")


HORIZON = 390
X_tr, y_tr = create_diff_sequences(scaled_data, SEQ_LEN, HORIZON, SEQ_LEN, split)
X_te, _    = create_diff_sequences(scaled_data, SEQ_LEN, HORIZON, split, len(data))
_, y_te_abs = create_absolute_sequences(scaled_data, SEQ_LEN, HORIZON, split, len(data))

model390 = BiLSTMTransformer(input_dim=5, horizon=HORIZON).to(DEVICE)
print("\nTraining 390-step model (with early stopping) ...")
model390 = train_model(
    model390, X_tr, y_tr,
    epochs=100, lr=1e-4, weight_decay=1e-4,
    patience=15
)
evaluate(model390, X_te, y_te_abs, "390-minute (full-day) horizon")


plt.figure(figsize=(10, 5))
plt.plot(true30[0], marker="o", label="Actual")
plt.plot(pred30[0], marker="x", linestyle="--", label="Predicted")
plt.title("TSLA – 30-minute forecast (first test window)")
plt.xlabel("Minute ahead")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsla_30min_forecast.png", dpi=150)
print("\nPlot saved: tsla_30min_forecast.png")