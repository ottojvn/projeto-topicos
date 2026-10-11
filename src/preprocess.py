from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import shutil
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
    if fp.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(fp)
            if df.empty or "start_time" not in df.columns or "heart_rate" not in df.columns:
                return None
            df["datetime"] = pd.to_datetime(df["start_time"])
            df["time"] = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()
            df["hr_bpm"] = df["heart_rate"].astype(float)
            return df[["time", "hr_bpm"]]
        except Exception as e:
            log.warning(f"Erro ao ler CSV {fp}: {e}")
            return None

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
    gt_file = next(gt_root.rglob(f"{base_clean}_HeartRateData.csv"), None)
    if gt_file is None:
        gt_file = next(gt_root.rglob(f"{base_clean}_HeartRateData.json"), None)
    return gt_file


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


def compute_subject_split(
    files: list[Path],
    gt_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    subjects_set = set()
    for fp in files:
        if extract_position_from_filename(fp) is None:
            continue
        gt_fp = find_matching_gt(gt_dir, fp.stem)
        if gt_fp is None:
            continue
        subj = extract_subject_from_gt(gt_fp)
        if subj not in INVALID_SUBJECTS:
            subjects_set.add(subj)

    rng = np.random.default_rng(RANDOM_STATE)
    subjects = np.array(sorted(subjects_set), dtype=np.int32)
    rng.shuffle(subjects)

    n_test = max(1, round(len(subjects) * TEST_SIZE))
    n_val  = max(1, round(len(subjects) * VAL_SIZE))

    test_subjs  = subjects[:n_test]
    val_subjs   = subjects[n_test:n_test + n_val]
    train_subjs = subjects[n_test + n_val:]

    subj_to_split: dict[int, str] = {}
    for s in train_subjs:
        subj_to_split[int(s)] = "train"
    for s in val_subjs:
        subj_to_split[int(s)] = "val"
    for s in test_subjs:
        subj_to_split[int(s)] = "test"

    return train_subjs, val_subjs, test_subjs, subj_to_split


def run_pipeline(
    dataset_path: Path,
    gt_dir: Path,
    out_dir: Path,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> None:
    fault_intervals = load_fault_intervals(gt_dir.parent / "faltas.txt")
    files = sorted(dataset_path.rglob("*.npz"))
    log.info(f"CSI files found: {len(files)}")

    train_subjs, val_subjs, test_subjs, subj_to_split = compute_subject_split(files, gt_dir)
    log.info(
        f"Split — train: {len(train_subjs)} | val: {len(val_subjs)} | test: {len(test_subjs)} subjects"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "_temp_chunks"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    chunk_counts = {"train": 0, "val": 0, "test": 0}
    window_counts = {"train": 0, "val": 0, "test": 0}
    all_y_vals: list[float] = []

    for fp in files:
        pos = extract_position_from_filename(fp)
        if pos is None:
            continue

        gt_fp = find_matching_gt(gt_dir, fp.stem)
        if gt_fp is None:
            continue

        subject = extract_subject_from_gt(gt_fp)
        if subject in INVALID_SUBJECTS or subject not in subj_to_split:
            continue

        split_name = subj_to_split[subject]

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
        del X_raw
        if X.shape[0] <= cut:
            del X
            continue
        X  = X[cut:]
        ts = ts[cut:] if ts is not None and ts.size == (X.shape[0] + cut) else None

        window = int(window_sec * fs_eff)
        step   = max(1, int(step_sec * fs_eff))

        if X.shape[0] < window:
            del X
            continue

        Xw, yw, diag = sliding_window_with_gt(
            X, ts, gt_df, fs_eff, window, step, fault_intervals,
        )
        del X, ts, gt_df

        if Xw is None:
            log.warning(f"NO WINDOWS | {fp.name} | {diag}")
            continue

        target_window = int(window_sec * fs)
        if Xw.shape[1] != target_window:
            log.warning(f"Window shape mismatch {Xw.shape[1]} → {target_window} | {fp.name}")
            if Xw.shape[1] > target_window:
                Xw = Xw[:, :target_window, :]
            else:
                pad_width = target_window - Xw.shape[1]
                Xw = np.pad(Xw, ((0, 0), (0, pad_width), (0, 0)), mode="edge")

        log.info(
            f"OK | {fp.name} | pos={pos} subj={subject} [{split_name}] | fs={fs_eff:.1f} Hz"
            f" | windows={len(Xw)} | mean_dt={diag['mean_abs_dt']:.3f}s"
        )

        n_w = len(Xw)
        c_idx = chunk_counts[split_name]
        chunk_file = temp_dir / f"{split_name}_{c_idx:06d}.npz"
        np.savez(
            chunk_file,
            X=Xw.astype(np.float16),
            y=yw,
            p=np.full(n_w, pos, dtype=np.int32),
            s=np.full(n_w, subject, dtype=np.int32),
        )
        chunk_counts[split_name] += 1
        window_counts[split_name] += n_w
        all_y_vals.extend(yw.tolist())

        del Xw, yw
        gc.collect()

    if sum(window_counts.values()) == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("No windows generated. Check dataset and GT paths.")

    y_arr = np.array(all_y_vals, dtype=np.float32)
    log.info(
        f"Dataset total windows: {len(y_arr)} | "
        f"HR: [{y_arr.min():.0f}, {y_arr.max():.0f}] BPM | mean={y_arr.mean():.1f}"
    )
    del all_y_vals, y_arr
    gc.collect()

    split_subj_map = {
        "train": train_subjs,
        "val": val_subjs,
        "test": test_subjs,
    }

    current_idx_offset = 0
    for split_name in ("train", "val", "test"):
        n_chunks = chunk_counts[split_name]
        subj_set = split_subj_map[split_name]
        if n_chunks == 0:
            log.warning(f"Split {split_name} has 0 chunks!")
            continue

        split_X: list[np.ndarray] = []
        split_y: list[np.ndarray] = []
        split_p: list[np.ndarray] = []
        split_s: list[np.ndarray] = []

        for c_i in range(n_chunks):
            chunk_fp = temp_dir / f"{split_name}_{c_i:06d}.npz"
            with np.load(chunk_fp) as data:
                split_X.append(data["X"])
                split_y.append(data["y"])
                split_p.append(data["p"])
                split_s.append(data["s"])
            chunk_fp.unlink(missing_ok=True)

        X_cat = np.concatenate(split_X, axis=0)
        y_cat = np.concatenate(split_y, axis=0)
        p_cat = np.concatenate(split_p, axis=0)
        s_cat = np.concatenate(split_s, axis=0)
        del split_X, split_y, split_p, split_s
        gc.collect()

        n_split = len(X_cat)
        idx_split = np.arange(current_idx_offset, current_idx_offset + n_split, dtype=np.int64)
        current_idx_offset += n_split

        np.savez_compressed(out_dir / f"X_{split_name}.npz", X=X_cat)
        np.save(out_dir / f"y_{split_name}.npy",            y_cat)
        np.save(out_dir / f"positions_{split_name}.npy",    p_cat)
        np.save(out_dir / f"subject_{split_name}.npy",      s_cat)
        np.save(out_dir / f"idx_{split_name}.npy",          idx_split)
        np.save(out_dir / f"subjects_{split_name}_ids.npy", subj_set)

        log.info(f"  {split_name:5s}: {n_split:>6} windows | subjects: {sorted(subj_set)}")

        del X_cat, y_cat, p_cat, s_cat, idx_split
        gc.collect()

    shutil.rmtree(temp_dir, ignore_errors=True)
    log.info("Preprocessing complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="CSI preprocessing pipeline")
    ap.add_argument("--dataset_path", required=True, help="Root dir of .npz CSI files")
    ap.add_argument("--gt_dir",       required=True, help="Root dir of smartwatch GT JSON files")
    ap.add_argument("--out_dir",      required=True, help="Output directory for tensors")
    ap.add_argument("--fs",           type=float, default=DEFAULT_FS, help="Fallback sample rate (Hz)")
    ap.add_argument("--window_sec",   type=float, default=20.0,       help="Sliding window length (s)")
    ap.add_argument("--step_sec",     type=float, default=0.5,        help="Sliding window step (s)")
    args = ap.parse_args()

    run_pipeline(
        dataset_path=Path(args.dataset_path),
        gt_dir=Path(args.gt_dir),
        out_dir=Path(args.out_dir),
        fs=args.fs,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
    )


if __name__ == "__main__":
    main()
