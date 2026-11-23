"""
MLP Temporal: aplica um MLP por timestep (sem flatten) e agrega com
atenção temporal antes da regressão final.

Entrada: (batch, T, n_subcarriers)  — sequência preservada
Saída:   (batch,)                   — HR estimado em bpm
"""
import argparse
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


DEVICE = get_device()

SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 250
LR          = 5e-4
HIDDEN      = 256
LAYERS      = 3
DROPOUT     = 0.3
PATIENCE    = 30
HUBER_DELTA  = 1.5
WEIGHT_DECAY = 1e-4
LR_FACTOR    = 0.5
LR_PATIENCE  = 7
LR_MIN       = 1e-5
GRAD_CLIP    = 1.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def load_data(input_dir: Path) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    X_train = np.load(input_dir / "X_train.npz")["X"].astype(np.float32)
    X_val   = np.load(input_dir / "X_val.npz")["X"].astype(np.float32)
    X_test  = np.load(input_dir / "X_test.npz")["X"].astype(np.float32)
    y_train = np.load(input_dir / "y_train.npy").astype(np.float32)
    y_val   = np.load(input_dir / "y_val.npy").astype(np.float32)
    y_test  = np.load(input_dir / "y_test.npy").astype(np.float32)

    # Garante shape (N, T, C) — adiciona dim se vier (N, T)
    if X_train.ndim == 2:
        X_train = X_train[:, :, np.newaxis]
        X_val   = X_val[:, :, np.newaxis]
        X_test  = X_test[:, :, np.newaxis]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    return train_loader, val_loader, test_loader, y_val, y_test


class TemporalMLP(nn.Module):
    """
    1. Projection: Linear(n_subcarriers → HIDDEN) por timestep
    2. Backbone:   (LAYERS-1) blocos Linear→LayerNorm→GELU→Dropout por timestep
    3. Attention:  escalar por timestep → softmax → média ponderada
    4. Regressor:  Linear(HIDDEN → 1)
    """

    def __init__(self, n_subcarriers: int, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()

        # Projeção inicial: mapeia cada timestep de n_subcarriers → hidden
        self.proj = nn.Sequential(
            nn.Linear(n_subcarriers, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        # Blocos adicionais aplicados por timestep (shared weights)
        blocks: list[nn.Module] = []
        for _ in range(layers - 1):
            blocks += [
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT),
            ]
        self.backbone = nn.Sequential(*blocks)

        # Atenção temporal: projeta cada timestep em um escalar
        self.attn = nn.Linear(hidden, 1)

        # Regressor final
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, n_subcarriers)
        h = self.proj(x)          # (batch, T, hidden)
        h = self.backbone(h)      # (batch, T, hidden)  — aplicado por timestep

        # Atenção temporal
        scores = self.attn(h).squeeze(-1)          # (batch, T)
        weights = torch.softmax(scores, dim=-1)    # (batch, T)
        context = (h * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)

        return self.regressor(context).squeeze(-1)  # (batch,)


def train(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    train_loader, val_loader, test_loader, y_val_true, y_test_true = load_data(input_dir)

    sample_x = next(iter(train_loader))[0]
    # shape: (batch, T, n_subcarriers)
    n_subcarriers = sample_x.shape[2]
    T             = sample_x.shape[1]

    model = TemporalMLP(n_subcarriers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE} | T={T} | n_subcarriers={n_subcarriers} | Parameters: {n_params:,}")

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    history          = {"train_loss": [], "val_mae": []}
    best_val_mae     = float("inf")
    patience_counter = 0
    ckpt_path        = output_dir / "best_model_mlp_temporal.pt"
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

    # Val
    preds_val = []
    with torch.no_grad():
        for bx, _ in val_loader:
            preds_val.append(model(bx.to(DEVICE)).cpu().numpy())
    y_pred_val = np.concatenate(preds_val)
    val_mae  = float(mean_absolute_error(y_val_true, y_pred_val))
    val_rmse = float(np.sqrt(np.mean((y_val_true - y_pred_val) ** 2)))
    val_r2   = float(r2_score(y_val_true, y_pred_val))

    # Test
    preds_test = []
    with torch.no_grad():
        for bx, _ in test_loader:
            preds_test.append(model(bx.to(DEVICE)).cpu().numpy())
    y_pred_test = np.concatenate(preds_test)
    test_mae  = float(mean_absolute_error(y_test_true, y_pred_test))
    test_rmse = float(np.sqrt(np.mean((y_test_true - y_pred_test) ** 2)))
    test_r2   = float(r2_score(y_test_true, y_pred_test))

    print("\n── MLP Temporal Results ─────────────────")
    print(f"Val  - MAE: {val_mae:.4f} BPM | RMSE: {val_rmse:.4f} BPM | R²: {val_r2:.4f}")
    print(f"Test - MAE: {test_mae:.4f} BPM | RMSE: {test_rmse:.4f} BPM | R²: {test_r2:.4f}")
    print(f"Fit Time: {elapsed:.0f}s | Epochs: {epoch + 1}")

    metrics = {
        "val_mae":        val_mae,
        "val_rmse":       val_rmse,
        "val_r2":         val_r2,
        "test_mae":       test_mae,
        "test_rmse":      test_rmse,
        "test_r2":        test_r2,
        "fit_time_s":     elapsed,
        "epochs_trained": epoch + 1,
        "best_val_mae":   best_val_mae,
        "T":              T,
        "n_subcarriers":  n_subcarriers,
        "n_params":       n_params,
    }

    with open(output_dir / "metrics_mlp_temporal.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "history_mlp_temporal.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {output_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Treino de MLP Temporal (sem flatten)")
    parser.add_argument("--input_dir",  type=Path, default=Path("data/saida_10s"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/mlp_temporal_10s"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.input_dir, args.output_dir)
