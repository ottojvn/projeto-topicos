import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1).astype(np.float64)


def load_data(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = flatten(np.load(input_dir / "X_train.npz")["X"])
    X_val   = flatten(np.load(input_dir / "X_val.npz")["X"])
    X_test  = flatten(np.load(input_dir / "X_test.npz")["X"])
    y_train = np.load(input_dir / "y_train.npy").astype(np.float64)
    y_val   = np.load(input_dir / "y_val.npy").astype(np.float64)
    y_test  = np.load(input_dir / "y_test.npy").astype(np.float64)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(input_dir)
    n_samples, n_features = X_train.shape

    model = LinearRegression()
    t_start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t_start

    rank = int(model.rank_)
    print(f"Samples: {n_samples:,} | Features: {n_features:,} | Rank: {rank:,}")
    if rank < n_features:
        print("Warning: X is rank-deficient; minimum-norm solution used.")

    # Avaliação em validação e teste
    y_pred_val = model.predict(X_val)
    val_mae  = float(mean_absolute_error(y_val, y_pred_val))
    val_rmse = float(np.sqrt(np.mean((y_val - y_pred_val) ** 2)))
    val_r2   = float(r2_score(y_val, y_pred_val))

    y_pred_test = model.predict(X_test)
    test_mae  = float(mean_absolute_error(y_test, y_pred_test))
    test_rmse = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
    test_r2   = float(r2_score(y_test, y_pred_test))

    print("\n── Linear Regression Results ────────────")
    print(f"Val  - MAE: {val_mae:.4f} BPM | RMSE: {val_rmse:.4f} BPM | R²: {val_r2:.4f}")
    print(f"Test - MAE: {test_mae:.4f} BPM | RMSE: {test_rmse:.4f} BPM | R²: {test_r2:.4f}")
    print(f"Fit Time: {elapsed:.2f}s")

    model_path = output_dir / "model_linear.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    metrics = {
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "n_samples": n_samples,
        "n_features": n_features,
        "rank": rank,
        "fit_time_s": elapsed,
    }

    metrics_path = output_dir / "metrics_linear.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Treino de Regressão Linear")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/saida_10s"),
        help="Diretório contendo os arquivos .npz e .npy",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/linear_10s"),
        help="Diretório de saída para salvar métricas e modelo",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.input_dir, args.output_dir)
