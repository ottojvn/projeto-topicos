from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("preprocess")

DEFAULT_FS   = 500 / 60.0
BANDPASS_LOW = 0.8
BANDPASS_HIGH = 2.17
FILTER_ORDER  = 3
SAVGOL_WINDOW = 15
SAVGOL_POLY   = 3
TRIM_S        = 10
TARGET_WINDOW = int(20 * DEFAULT_FS)

TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
RANDOM_STATE = 42

INVALID_SUBJECTS: frozenset[int] = frozenset({
    9, 10, 35, 59, 61, 62, 63, 64, 65, 66,
    67, 68, 69, 70, 71, 81, 90, 98, 106, 203,
})


def csi_to_amplitude(x: np.ndarray) -> np.ndarray:
    return np.abs(x).astype(np.float32) if np.iscomplexobj(x) else x.astype(np.float32)


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0)


def bandpass_filter(x: np.ndarray, fs: float) -> np.ndarray:
    b, a = butter(FILTER_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype="band", fs=fs)
    out = np.empty_like(x)
    for i in range(x.shape[1]):
        out[:, i] = filtfilt(b, a, x[:, i])
    return out


def savgol_smooth(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < SAVGOL_WINDOW:
        return x
    out = np.empty_like(x)
    for i in range(x.shape[1]):
        out[:, i] = savgol_filter(x[:, i], SAVGOL_WINDOW, SAVGOL_POLY)
    return out


def preprocess_signal(x: np.ndarray, fs: float) -> np.ndarray:
    x = csi_to_amplitude(x)
    x = remove_dc(x)
    x = bandpass_filter(x, fs)
    x = savgol_smooth(x)
    return x


def infer_fs(ts: Optional[np.ndarray]) -> Optional[float]:
    if ts is None:
        return None
    try:
        ts = np.array(ts, dtype=float).ravel()
    except Exception:
        return None
    if ts.size < 2:
        return None
    dt = np.mean(np.diff(ts))
    return 1.0 / dt if np.isfinite(dt) and dt > 0 else None


def load_one_npz(fp: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    z   = np.load(fp, allow_pickle=True)
    csi = z["csi"]
    if csi.ndim == 1:
        csi = csi[:, np.newaxis]
    elif csi.ndim > 2:
        csi = csi.reshape(csi.shape[0], -1)
    ts = z["ts"] if "ts" in z.files else None
    return csi_to_amplitude(csi), ts


def load_fault_intervals(fp: Path) -> list[tuple[float, float]]:
    if not fp.exists():
        return []
    intervals = []
    with open(fp) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                intervals.append((float(parts[0]), float(parts[1])))
    return intervals


def load_smartwatch_gt(fp: Path) -> Optional[pd.DataFrame]:
    with open(fp) as f:
        js = json.load(f)

    rows: list[tuple] = []

    if isinstance(js, dict) and "Data" in js:
        for rec in js["Data"]:
            hr = rec.get("HeartRate") or rec.get("Value")
            t  = rec.get("StartTime") or rec.get("Time")
            if hr is not None and t is not None:
                rows.append((pd.to_datetime(t.replace(" ", "T")), float(hr)))

    elif isinstance(js, dict) and "heart_rate" in js and "start_time" in js:
        for t, hr in zip(js["start_time"], js["heart_rate"]):
            rows.append((pd.to_datetime(t.replace(" ", "T")), float(hr)))

    elif isinstance(js, list):
        for rec in js:
            hr = rec.get("HeartRate") or rec.get("Value")
            t  = rec.get("StartTime") or rec.get("Time")
            if hr is not None and t is not None:
                rows.append((pd.to_datetime(t.replace(" ", "T")), float(hr)))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["datetime", "hr_bpm"])
    df["time"] = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()
    return df[["time", "hr_bpm"]]


def find_matching_gt(gt_root: Path, base_name: str) -> Optional[Path]:
    base_clean = re.sub(r"_bw_.*$", "", base_name)
    return next(gt_root.rglob(f"{base_clean}_HeartRateData.json"), None)


def extract_position_from_filename(fp: Path) -> Optional[int]:
    m = re.match(r"^(\d+)_", fp.name)
    return int(m.group(1)) if m else None


def extract_subject_from_gt(gt_path: Path) -> int:
    return int(gt_path.parent.name)


def _is_in_fault(t: float, intervals: list[tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in intervals)


def sliding_window_with_gt(
    X: np.ndarray,
    ts: Optional[np.ndarray],
    gt_df: pd.DataFrame,
    fs: float,
    window: int,
    step: int,
    fault_intervals: list[tuple[float, float]],
    *,
    max_gt_gap_s: float = 0.75,
    hr_jump_bpm: float = 25.0,
    local_hr_window_s: float = 3.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:

    stats: Counter = Counter()

    try:
        ts_arr = np.array(ts, dtype=float).ravel()
        if ts_arr.size != len(X) or not np.all(np.isfinite(ts_arr)):
            raise ValueError
    except Exception:
        ts_arr = np.arange(len(X)) / fs
        stats["ts_fallback"] += 1

    gt_t  = gt_df["time"].to_numpy(dtype=float)
    gt_hr = gt_df["hr_bpm"].to_numpy(dtype=float)

    if gt_t.size < 2:
        return None, None, {"reason": "gt_too_small"}

    X_out, y_out, dt_list = [], [], []

    for start in range(0, len(X) - window + 1, step):
        end  = start + window
        win  = X[start:end]
        t_c  = float(ts_arr[start:end].mean())

        if _is_in_fault(t_c, fault_intervals):
            stats["drop_fault"] += 1
            continue

        idx    = int(np.argmin(np.abs(gt_t - t_c)))
        abs_dt = abs(float(gt_t[idx]) - t_c)

        if abs_dt > max_gt_gap_s:
            stats["drop_gap"] += 1
            continue

        mask = (gt_t >= t_c - local_hr_window_s) & (gt_t <= t_c + local_hr_window_s)
        if mask.sum() >= 3 and (gt_hr[mask].max() - gt_hr[mask].min()) > hr_jump_bpm:
            stats["drop_unstable"] += 1
            continue

        win = (win - win.mean()) / (win.std() + 1e-8)
        X_out.append(win)
        y_out.append(float(gt_hr[idx]))
        dt_list.append(float(gt_t[idx]) - t_c)

    if not X_out:
        return None, None, {"reason": "no_windows", "stats": dict(stats)}

    dt_arr = np.abs(dt_list)
    diag = {
        "accepted":     len(X_out),
        "mean_abs_dt":  float(dt_arr.mean()),
        "p95_abs_dt":   float(np.percentile(dt_arr, 95)),
        "stats":        dict(stats),
    }
    return np.stack(X_out, dtype=np.float32), np.array(y_out, dtype=np.float32), diag


def build_windows(
    dataset_path: Path,
    gt_dir: Path,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    fault_intervals = load_fault_intervals(gt_dir.parent / "faltas.txt")
    files = sorted(dataset_path.rglob("*.npz"))
    log.info(f"CSI files found: {len(files)}")

    all_X, all_y, all_p, all_s = [], [], [], []

    for fp in files:
        pos = extract_position_from_filename(fp)
        if pos is None:
            continue

        gt_fp = find_matching_gt(gt_dir, fp.stem)
        if gt_fp is None:
            continue

        subject = extract_subject_from_gt(gt_fp)
        if subject in INVALID_SUBJECTS:
            continue

        gt_df = load_smartwatch_gt(gt_fp)
        if gt_df is None or len(gt_df) == 0:
            continue
        gt_df["time"] -= gt_df["time"].iloc[0]

        X_raw, ts_raw = load_one_npz(fp)

        try:
            ts = np.array(ts_raw, dtype=float).ravel() - np.array(ts_raw, dtype=float).ravel()[0]
        except Exception:
            ts = None

        fs_eff = infer_fs(ts) or fs
        cut    = int(TRIM_S * fs_eff)

        X = preprocess_signal(X_raw, fs_eff)
        if X.shape[0] <= cut:
            continue
        X  = X[cut:]
        ts = ts[cut:] if ts is not None and ts.size == X_raw.shape[0] else None

        window = int(window_sec * fs_eff)
        step   = max(1, int(step_sec * fs_eff))

        if X.shape[0] < window:
            continue

        Xw, yw, diag = sliding_window_with_gt(
            X, ts, gt_df, fs_eff, window, step, fault_intervals,
        )
        if Xw is None:
            log.warning(f"NO WINDOWS | {fp.name} | {diag}")
            continue

        if Xw.shape[1] != TARGET_WINDOW:
            log.warning(f"Window shape mismatch {Xw.shape[1]} → {TARGET_WINDOW} | {fp.name}")
            Xw = Xw[:, :TARGET_WINDOW, :]

        log.info(
            f"OK | {fp.name} | pos={pos} subj={subject} | fs={fs_eff:.1f} Hz"
            f" | windows={len(Xw)} | mean_dt={diag['mean_abs_dt']:.3f}s"
        )
        all_X.append(Xw)
        all_y.append(yw)
        all_p.append(np.full(len(Xw), pos,     dtype=np.int32))
        all_s.append(np.full(len(Xw), subject, dtype=np.int32))

    if not all_X:
        raise RuntimeError("No windows generated. Check dataset and GT paths.")

    return (
        np.concatenate(all_X, dtype=np.float32),
        np.concatenate(all_y, dtype=np.float32),
        np.concatenate(all_p, dtype=np.int32),
        np.concatenate(all_s, dtype=np.int32),
    )


def split_and_save(
    X: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    s: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng      = np.random.default_rng(RANDOM_STATE)
    subjects = np.unique(s)
    rng.shuffle(subjects)

    n_test = max(1, round(len(subjects) * TEST_SIZE))
    n_val  = max(1, round(len(subjects) * VAL_SIZE))

    test_subjs  = subjects[:n_test]
    val_subjs   = subjects[n_test:n_test + n_val]
    train_subjs = subjects[n_test + n_val:]

    log.info(f"Split — train: {len(train_subjs)} | val: {len(val_subjs)} | test: {len(test_subjs)} subjects")

    def _save(name: str, subj_set: np.ndarray) -> None:
        mask = np.isin(s, subj_set)
        idx  = np.where(mask)[0]
        np.savez_compressed(out_dir / f"X_{name}.npz", X=X[mask])
        np.save(out_dir / f"y_{name}.npy",            y[mask])
        np.save(out_dir / f"positions_{name}.npy",    p[mask])
        np.save(out_dir / f"subject_{name}.npy",      s[mask])
        np.save(out_dir / f"idx_{name}.npy",          idx)
        np.save(out_dir / f"subjects_{name}_ids.npy", subj_set)
        log.info(f"  {name:5s}: {mask.sum():>6} windows | subjects: {sorted(subj_set)}")

    _save("train", train_subjs)
    _save("val",   val_subjs)
    _save("test",  test_subjs)


def main() -> None:
    ap = argparse.ArgumentParser(description="CSI preprocessing pipeline")
    ap.add_argument("--dataset_path", required=True, help="Root dir of .npz CSI files")
    ap.add_argument("--gt_dir",       required=True, help="Root dir of smartwatch GT JSON files")
    ap.add_argument("--out_dir",      required=True, help="Output directory for tensors")
    ap.add_argument("--fs",           type=float, default=DEFAULT_FS, help="Fallback sample rate (Hz)")
    ap.add_argument("--window_sec",   type=float, default=20.0,       help="Sliding window length (s)")
    ap.add_argument("--step_sec",     type=float, default=0.5,        help="Sliding window step (s)")
    args = ap.parse_args()

    X, y, p, s = build_windows(
        Path(args.dataset_path),
        Path(args.gt_dir),
        args.fs,
        args.window_sec,
        args.step_sec,
    )

    log.info(f"Dataset — X: {X.shape} | HR: [{y.min():.0f}, {y.max():.0f}] BPM | mean={y.mean():.1f}")
    split_and_save(X.astype(np.float16), y, p, s, Path(args.out_dir))


if __name__ == "__main__":
    main()
