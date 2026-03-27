#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional
from tensorflow.keras.utils import pad_sequences


FRAME_CSV_PATH        = os.path.join("frame_level", "pose_keypoints.csv")
MANUAL_SEGMENTS_PATH  = os.path.join("rep_level",  "manual_segments.csv")
AUTO_SEGMENTS_OUTPATH = os.path.join("rep_level",  "auto_segments_ensemble.csv")
MODEL_DIR             = "models"


N_MODELS = 3  
ENSEMBLE_VOTE_THR = 0.6  

FPS_FALLBACK = 25.0


BASE_PROB_THR      = 0.55
BASE_GAP_MERGE_FR  = 15
BASE_LEN_MIN_SCALE = 0.90
BASE_LEN_MAX_SCALE = 1.50
BASE_KNEE_ROM_MIN  = 22.0


MIN_CONFIDENCE = 0.65  # Minimum confidence to accept a segment



def save_csv_overwrite(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    df.to_csv(path, index=False)




def load_data():
    if not os.path.exists(FRAME_CSV_PATH):
        raise FileNotFoundError(f"Could not find {FRAME_CSV_PATH}")
    if not os.path.exists(MANUAL_SEGMENTS_PATH):
        raise FileNotFoundError(f"Could not find {MANUAL_SEGMENTS_PATH}")

    print(f"📥 Loading data...")
    df_frames = pd.read_csv(FRAME_CSV_PATH)
    df_gt = pd.read_csv(MANUAL_SEGMENTS_PATH)

    needed_cols = {"file", "frame", "hip_y", "knee_deg"}
    if not needed_cols.issubset(df_frames.columns):
        raise ValueError(f"Missing columns in pose_keypoints.csv")

    common_files = sorted(set(df_frames["file"]).intersection(set(df_gt["file"])))
    df_frames = df_frames[df_frames["file"].isin(common_files)].copy()
    df_gt = df_gt[df_gt["file"].isin(common_files)].copy()

    print(f"✅ Found {len(common_files)} labelled videos")
    return df_frames, df_gt, common_files



def safe_normalize(series: np.ndarray) -> np.ndarray:
    s = pd.Series(series, dtype="float64").ffill().bfill()
    if s.isna().all():
        return np.zeros_like(series, dtype=float)
    mean, std = float(s.mean()), float(s.std())
    if std < 1e-6:
        return np.zeros_like(series, dtype=float)
    return ((s - mean) / std).values


def compute_velocity(series: np.ndarray) -> np.ndarray:
    if len(series) < 3:
        return np.zeros_like(series)
    return np.gradient(series)


def build_sequences(df_frames: pd.DataFrame, df_gt: pd.DataFrame, files: list):
    X_list, y_list, meta = [], [], []

    for f in files:
        df_f = df_frames[df_frames["file"] == f].copy().sort_values("frame")
        frames = df_f["frame"].values.astype(int)

        # Extract and normalize features
        hip_y = pd.Series(df_f["hip_y"].astype(float)).ffill().bfill().values
        knee_deg = pd.Series(df_f["knee_deg"].astype(float)).ffill().bfill().values

        hip_y_norm = safe_normalize(hip_y)
        knee_deg_norm = safe_normalize(knee_deg)
        hip_vel = safe_normalize(compute_velocity(hip_y))
        knee_vel = safe_normalize(compute_velocity(knee_deg))

        # Stack features
        X = np.stack([hip_y_norm, knee_deg_norm, hip_vel, knee_vel], axis=-1)

        # Build labels
        y = np.zeros(len(frames), dtype=int)
        df_g = df_gt[df_gt["file"] == f]
        for _, row in df_g.iterrows():
            s, e = int(row["start_frame"]), int(row["end_frame"])
            mask = (frames >= s) & (frames <= e)
            y[mask] = 1

        X_list.append(X)
        y_list.append(y)
        meta.append({"file": f, "frames": frames})

    return X_list, y_list, meta




def build_model(input_dim: int, seed: int) -> tf.keras.Model:
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(None, input_dim)),
        Bidirectional(LSTM(96, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(48, return_sequences=True)),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def prepare_padded(X_list, y_list):
    max_len = max(seq.shape[0] for seq in X_list)
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype="float32", padding="post", value=0.0)
    y_padded = pad_sequences(y_list, maxlen=max_len, dtype="float32", padding="post", value=0.0)
    y_padded = np.expand_dims(y_padded, -1)
    return X_padded, y_padded




def assess_signal_quality(hip_y: np.ndarray, knee_deg: np.ndarray) -> float:
    
    quality_scores = []
    
    # Check for NaN ratio
    hip_valid = np.isfinite(hip_y).sum() / len(hip_y)
    knee_valid = np.isfinite(knee_deg).sum() / len(knee_deg)
    quality_scores.append((hip_valid + knee_valid) / 2)
    
    # Check for variability (more variability = better for rep detection)
    hip_std = np.nanstd(hip_y)
    knee_std = np.nanstd(knee_deg)
    hip_range = np.nanmax(hip_y) - np.nanmin(hip_y) if np.isfinite(hip_y).any() else 0
    knee_range = np.nanmax(knee_deg) - np.nanmin(knee_deg) if np.isfinite(knee_deg).any() else 0
    
    variability_score = min(1.0, (hip_range / 100 + knee_range / 100) / 2)
    quality_scores.append(variability_score)
    
    return float(np.mean(quality_scores))


def get_adaptive_thresholds(signal_quality: float):
   
    if signal_quality > 0.8:  # High quality
        prob_thr = BASE_PROB_THR - 0.05
        gap_merge = BASE_GAP_MERGE_FR + 5
        len_min = BASE_LEN_MIN_SCALE - 0.1
    elif signal_quality < 0.5:  # Low quality
        prob_thr = BASE_PROB_THR + 0.10
        gap_merge = BASE_GAP_MERGE_FR - 5
        len_min = BASE_LEN_MIN_SCALE + 0.1
    else:  # Medium quality
        prob_thr = BASE_PROB_THR
        gap_merge = BASE_GAP_MERGE_FR
        len_min = BASE_LEN_MIN_SCALE
    
    return {
        'prob_thr': prob_thr,
        'gap_merge': max(5, gap_merge),
        'len_min_scale': len_min,
        'len_max_scale': BASE_LEN_MAX_SCALE,
        'knee_rom_min': BASE_KNEE_ROM_MIN
    }




def ensemble_predict(models: list, X: np.ndarray, smoothing_win: int = 9) -> tuple:
   
    all_probs = []
    
    for model in models:
        X_in = np.expand_dims(X, axis=0)
        probs = model.predict(X_in, verbose=0)[0, :, 0]
        
        # Smooth predictions
        probs_smooth = pd.Series(probs).rolling(smoothing_win, center=True, min_periods=1).mean().values
        all_probs.append(probs_smooth)
    
    # Ensemble statistics
    probs_stack = np.stack(all_probs, axis=0)
    probs_mean = np.mean(probs_stack, axis=0)
    probs_std = np.std(probs_stack, axis=0)
    
    # Confidence: high when models agree (low std)
    confidence = 1.0 - probs_std
    
    return probs_mean, probs_std, confidence




def segments_from_ensemble(probs_mean: np.ndarray,
                           confidence: np.ndarray,
                           fps: float,
                           thresholds: dict,
                           knee_series: pd.Series = None):
    
    if fps <= 0:
        fps = FPS_FALLBACK
    
    prob_thr = thresholds['prob_thr']
    gap_merge_fr = thresholds['gap_merge']
    len_min_scale = thresholds['len_min_scale']
    len_max_scale = thresholds['len_max_scale']
    knee_rom_min = thresholds['knee_rom_min']
    
    
    mask = (probs_mean >= prob_thr) & (confidence >= MIN_CONFIDENCE)
    n = len(mask)
    
    if not mask.any():
        return []
    
  
    segments = []
    in_rep = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_rep:
            in_rep = True
            start = i
        elif not val and in_rep:
            in_rep = False
            end = i - 1
            segments.append([start, end])
    if in_rep:
        segments.append([start, n - 1])
    
    if not segments:
        return []
    
    
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e - 1 <= gap_merge_fr:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    
    segments = merged
    
    
    lengths = np.array([e - s + 1 for s, e in segments], dtype=float)
    if len(lengths) == 0:
        return []
    
    med_len = float(np.median(lengths))
    min_len = max(int(0.8 * fps), int(len_min_scale * med_len))
    max_len = int(len_max_scale * med_len) if med_len > 0 else n
    
    filtered = []
    for s, e in segments:
        L = e - s + 1
        if L < min_len or L > max_len:
            continue
        
      
        seg_conf = confidence[s:e+1]
        avg_conf = float(np.mean(seg_conf))
        if avg_conf < MIN_CONFIDENCE:
            continue
        
       
        if knee_series is not None and knee_rom_min > 0.0:
            seg_knee = np.array(knee_series.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < knee_rom_min:
                    continue
        
        filtered.append((int(s), int(e)))
    
    return filtered




def fallback_segment(df_file: pd.DataFrame, fps: float = None):
    
    if fps is None or fps <= 0:
        fps = FPS_FALLBACK
    
    if "hip_y" not in df_file.columns:
        return []
    
    hip_y = df_file["hip_y"].values.astype(float)
    hip_y = pd.Series(hip_y).ffill().bfill().values
    
    # Normalize
    hip_min, hip_max = np.nanmin(hip_y), np.nanmax(hip_y)
    if hip_max - hip_min < 1e-6:
        return []
    hip_norm = (hip_y - hip_min) / (hip_max - hip_min)
    
    inv = -hip_norm
    peaks, _ = find_peaks(inv, prominence=0.03, distance=int(fps * 0.7))
    
    if len(peaks) == 0:
        return []
    
    # Build segments around peaks
    reps = []
    for center in peaks:
        half_win = int(fps * 1.5)
        s = max(0, center - half_win)
        e = min(len(hip_y) - 1, center + half_win)
        reps.append((s, e))
    
   
    reps_sorted = sorted(reps, key=lambda x: x[0])
    merged = []
    for s, e in reps_sorted:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    
    return [(int(s), int(e)) for s, e in merged]




def mask_from_segments(segments, n_frames):
    mask = np.zeros(n_frames, dtype=int)
    for s, e in segments:
        s = max(0, min(n_frames - 1, s))
        e = max(0, min(n_frames - 1, e))
        mask[s:e+1] = 1
    return mask


def iou_masks(a, b):
    inter = np.logical_and(a == 1, b == 1).sum()
    union = np.logical_or(a == 1, b == 1).sum()
    return float(inter) / union if union > 0 else 0.0


def pairwise_iou(gt_segments, pred_segments):
    if len(gt_segments) == 0 or len(pred_segments) == 0:
        return 0.0
    ious = []
    for gs, ge in gt_segments:
        best = 0.0
        for ps, pe in pred_segments:
            s = max(gs, ps)
            e = min(ge, pe)
            inter = max(0, e - s + 1)
            union = (ge - gs + 1) + (pe - ps + 1) - inter
            if union > 0:
                best = max(best, inter / union)
        ious.append(best)
    return float(np.mean(ious)) if ious else 0.0




def main():
    df_frames, df_gt, files = load_data()
    
    print("\n🔧 Building sequences...")
    X_list, y_list, meta = build_sequences(df_frames, df_gt, files)
    X_padded, y_padded = prepare_padded(X_list, y_list)
    
    # Train/val split
    idx = np.arange(len(files))
    idx_train, idx_val = train_test_split(idx, test_size=0.25, random_state=42, shuffle=True)
    
    X_train, y_train = X_padded[idx_train], y_padded[idx_train]
    X_val, y_val = X_padded[idx_val], y_padded[idx_val]
    
    # Train ensemble
    models = []
    print(f"\n🧠 Training ensemble of {N_MODELS} models...")
    
    for i in range(N_MODELS):
        print(f"\n--- Training Model {i+1}/{N_MODELS} (seed={42+i}) ---")
        model = build_model(input_dim=X_padded.shape[-1], seed=42+i)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=4,
            verbose=1
        )
        
        models.append(model)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"rep_segmenter_ensemble_{i}.keras")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(model_path)
        print(f"💾 Saved model {i+1} to {model_path}")
    
    print("\n Ensemble training complete!\n")
    
    # Evaluation
    all_rows = []
    mask_ious, pair_ious, count_diffs = [], [], []
    
    print("🔍 Evaluating with ensemble + adaptive thresholds:\n")
    
    for i, info in enumerate(meta):
        f = info["file"]
        frames = info["frames"]
        n_frames = len(frames)
        
        df_f = df_frames[df_frames["file"] == f].copy().sort_values("frame")
        X = X_list[i]
        
        # Assess signal quality
        hip_y = df_f["hip_y"].values.astype(float)
        knee_deg = df_f["knee_deg"].values.astype(float)
        signal_quality = assess_signal_quality(hip_y, knee_deg)
        
        # Get adaptive thresholds
        thresholds = get_adaptive_thresholds(signal_quality)
        
        # Ensemble prediction
        probs_mean, probs_std, confidence = ensemble_predict(models, X)
        
        # Segment extraction
        knee_series = df_f["knee_deg"].astype(float)
        seg_pred = segments_from_ensemble(
            probs_mean, confidence, FPS_FALLBACK, thresholds, knee_series
        )
        
        # Fallback if needed
        if not seg_pred:
            seg_pred = fallback_segment(df_f, FPS_FALLBACK)
        
        pred_segments_global = [(int(frames[s]), int(frames[e])) for (s, e) in seg_pred]
        
        # Ground truth
        df_g = df_gt[df_gt["file"] == f]
        gt_segments = [(int(r["start_frame"]), int(r["end_frame"])) for _, r in df_g.iterrows()]
        
        # Metrics
        if gt_segments:
            gt_idx_segments = []
            for (gs, ge) in gt_segments:
                idx_s = np.where(frames == gs)[0]
                idx_e = np.where(frames == ge)[0]
                if len(idx_s) == 0 or len(idx_e) == 0:
                    continue
                gt_idx_segments.append((int(idx_s[0]), int(idx_e[-1])))
            gt_mask = mask_from_segments(gt_idx_segments, n_frames)
        else:
            gt_mask = np.zeros(n_frames, dtype=int)
        
        pred_mask = mask_from_segments(seg_pred, n_frames)
        
        m_iou = iou_masks(gt_mask, pred_mask)
        p_iou = pairwise_iou(gt_segments, pred_segments_global)
        c_diff = len(pred_segments_global) - len(gt_segments)
        
        mask_ious.append(m_iou)
        pair_ious.append(p_iou)
        count_diffs.append(c_diff)
        
        print(f"File: {f}")
        print(f"  Signal Quality: {signal_quality:.2f}")
        print(f"  Adaptive Prob Threshold: {thresholds['prob_thr']:.2f}")
        print(f"  GT   : {gt_segments}")
        print(f"  Pred : {pred_segments_global}")
        print(f"  mask_iou={m_iou:.3f}, pair_iou={p_iou:.3f}, count_diff={c_diff}\n")
        
        for (s, e) in pred_segments_global:
            all_rows.append({"file": f, "start_frame": s, "end_frame": e})
    
    # Final metrics
    avg_mask_iou = float(np.mean(mask_ious)) if mask_ious else 0.0
    avg_pair_iou = float(np.mean(pair_ious)) if pair_ious else 0.0
    avg_count_diff = float(np.mean(count_diffs)) if count_diffs else 0.0
    
    print("="*80)
    print("📊 ENSEMBLE MODEL - Global Metrics:")
    print("="*80)
    print(f"  avg_mask_iou  : {avg_mask_iou:.3f}")
    print(f"  avg_pair_iou  : {avg_pair_iou:.3f}")
    print(f"  avg_count_diff: {avg_count_diff:.3f}")
    print("="*80)
    
    df_out = pd.DataFrame(all_rows, columns=["file", "start_frame", "end_frame"])
    save_csv_overwrite(df_out, AUTO_SEGMENTS_OUTPATH)
    print(f"\n💾 Predictions saved to: {AUTO_SEGMENTS_OUTPATH}")


if __name__ == "__main__":
    main()