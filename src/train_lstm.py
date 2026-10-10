import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE     = get_device()
INPUT_DIR  = Path("saida_full")
OUTPUT_DIR = Path("output/lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED          = 42
BATCH_SIZE    = 64
EPOCHS        = 250
LR            = 5e-4
WEIGHT_DECAY  = 1e-5
HIDDEN        = 256
LAYERS        = 2
DROPOUT_RNN   = 0.3
DROPOUT_REG   = 0.2
PATIENCE      = 30
HUBER_DELTA   = 3.0
LR_FACTOR     = 0.5
LR_PATIENCE   = 10
LR_MIN        = 1e-5
GRAD_CLIP     = 1.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def load_data() -> tuple[DataLoader, DataLoader, np.ndarray]:
    X_train = np.load(INPUT_DIR / "X_train.npz")["X"].astype(np.float32)
    X_val   = np.load(INPUT_DIR / "X_val.npz")["X"].astype(np.float32)
    y_train = np.load(INPUT_DIR / "y_train.npy").astype(np.float32)
    y_val   = np.load(INPUT_DIR / "y_val.npy").astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, val_loader, y_val


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, layers,
            batch_first=True,
            dropout=DROPOUT_RNN if layers > 1 else 0,
            bidirectional=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_REG),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :]).squeeze(-1)


def train() -> None:
    set_seed()
    train_loader, val_loader, y_val_true = load_data()
    input_dim = next(iter(train_loader))[0].shape[2]

    model = LSTM(input_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE} | Parameters: {n_params:,}")

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    history          = {"train_loss": [], "val_mae": []}
    best_val_mae     = float("inf")
    patience_counter = 0
    ckpt_path        = OUTPUT_DIR / "best_model_lstm.pt"
    t_start          = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_mae += nn.L1Loss()(model(bx), by).item()

        train_loss /= len(train_loader)
        val_mae    /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Huber: {train_loss:.4f}"
                f" | Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} | Best val MAE: {best_val_mae:.4f}")
            break

    elapsed = time.time() - t_start

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in val_loader:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    y_pred = np.concatenate(preds)

    mae  = float(mean_absolute_error(y_val_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_val_true - y_pred) ** 2)))
    r2   = float(r2_score(y_val_true, y_pred))
    print(f"\n── LSTM Results ─────────────────────────")
    print(f"MAE:  {mae:.4f} BPM")
    print(f"RMSE: {rmse:.4f} BPM")
    print(f"R²:   {r2:.4f}")
    print(f"Time: {elapsed:.0f}s")

    history_path = OUTPUT_DIR / "history_lstm.json"
    with open(history_path, "w") as f:
        json.dump({**history, "input_dim": input_dim}, f)
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    train()
