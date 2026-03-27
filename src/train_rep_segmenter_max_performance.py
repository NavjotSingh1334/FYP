#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout, 
                                      Masking, BatchNormalization, GaussianNoise)
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



FRAME_CSV_PATH        = os.path.join("frame_level", "pose_keypoints.csv")
MANUAL_SEGMENTS_PATH  = os.path.join("rep_level",  "manual_segments.csv")
AUTO_SEGMENTS_OUTPATH = os.path.join("rep_level",  "auto_segments_max_performance.csv")
MODEL_DIR             = "models"
BEST_MODEL_PATH       = os.path.join(MODEL_DIR, "rep_segmenter_best.keras")

FPS_FALLBACK = 25.0

PROB_THR       = 0.62
GAP_MERGE_FR   = 18
LEN_MIN_SCALE  = 0.95
LEN_MAX_SCALE  = 1.40
KNEE_ROM_MIN   = 23.0


N_FOLDS = 5  # 5-fold cross-validation
EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 4


AUG_NOISE_FACTOR = 0.02
AUG_TIME_WARP_RANGE = (0.9, 1.1)
AUG_SCALE_RANGE = (0.95, 1.05)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)




def save_csv_overwrite(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    df.to_csv(path, index=False)




def augment_temporal_warp(X: np.ndarray, y: np.ndarray, warp_range: tuple) -> tuple:
    """
    Temporal warping: speed up or slow down the sequence
    """
    warp_factor = np.random.uniform(*warp_range)
    orig_len = len(X)
    new_len = int(orig_len * warp_factor)
    
    if new_len < 10:  # Safety check
        return X, y
    
    old_indices = np.arange(orig_len)
    new_indices = np.linspace(0, orig_len - 1, new_len)
    
    X_warped = np.zeros((new_len, X.shape[1]))
    for feat_idx in range(X.shape[1]):
        interp_func = interp1d(old_indices, X[:, feat_idx], kind='linear', fill_value='extrapolate')
        X_warped[:, feat_idx] = interp_func(new_indices)
    
    # Warp labels
    y_interp = interp1d(old_indices, y.astype(float), kind='nearest', fill_value='extrapolate')
    y_warped = (y_interp(new_indices) > 0.5).astype(int)
    
    return X_warped, y_warped


def augment_add_noise(X: np.ndarray, noise_factor: float) -> np.ndarray:
    
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def augment_scale(X: np.ndarray, scale_range: tuple) -> np.ndarray:
    
    scale_factor = np.random.uniform(*scale_range)
    return X * scale_factor


def augment_sequence(X: np.ndarray, y: np.ndarray, 
                     apply_warp: bool = True,
                     apply_noise: bool = True,
                     apply_scale: bool = True) -> tuple:
    
    
    
    if apply_warp and np.random.rand() > 0.5:
        X, y = augment_temporal_warp(X, y, AUG_TIME_WARP_RANGE)
    
    
    if apply_noise and np.random.rand() > 0.5:
        X = augment_add_noise(X, AUG_NOISE_FACTOR)
    
    # Scaling
    if apply_scale and np.random.rand() > 0.5:
        X = augment_scale(X, AUG_SCALE_RANGE)
    
    return X, y




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


def compute_acceleration(series: np.ndarray) -> np.ndarray:
    if len(series) < 3:
        return np.zeros_like(series)
    return np.gradient(np.gradient(series))


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
        raise ValueError(f"Missing required columns")
    
    optional_cols = ["hip_deg", "ankle_deg", "trunk_deg"]
    available_optional = [col for col in optional_cols if col in df_frames.columns]

    common_files = sorted(set(df_frames["file"]).intersection(set(df_gt["file"])))
    df_frames = df_frames[df_frames["file"].isin(common_files)].copy()
    df_gt = df_gt[df_gt["file"].isin(common_files)].copy()

    print(f"✅ Found {len(common_files)} labelled videos")
    print(f"✅ Optional features: {', '.join(available_optional) if available_optional else 'none'}")
    
    return df_frames, df_gt, common_files, available_optional


def build_sequences(df_frames: pd.DataFrame, df_gt: pd.DataFrame, 
                   files: list, optional_cols: list,
                   augment: bool = False):
    
    X_list, y_list, meta = [], [], []

    for f in files:
        df_f = df_frames[df_frames["file"] == f].copy().sort_values("frame")
        frames = df_f["frame"].values.astype(int)

        # Extract features
        hip_y = pd.Series(df_f["hip_y"].astype(float)).ffill().bfill().values
        knee_deg = pd.Series(df_f["knee_deg"].astype(float)).ffill().bfill().values

        # Base features
        features = [
            safe_normalize(hip_y),
            safe_normalize(knee_deg),
            safe_normalize(compute_velocity(hip_y)),
            safe_normalize(compute_velocity(knee_deg)),
            safe_normalize(compute_acceleration(hip_y)),
            safe_normalize(compute_acceleration(knee_deg))
        ]
        
        # Optional features
        for col in optional_cols:
            if col in df_f.columns:
                val = pd.Series(df_f[col].astype(float)).ffill().bfill().values
                features.extend([
                    safe_normalize(val),
                    safe_normalize(compute_velocity(val))
                ])
        
        X = np.stack(features, axis=-1)
        
        # Build labels
        y = np.zeros(len(frames), dtype=int)
        df_g = df_gt[df_gt["file"] == f]
        for _, row in df_g.iterrows():
            s, e = int(row["start_frame"]), int(row["end_frame"])
            mask = (frames >= s) & (frames <= e)
            y[mask] = 1
        
        # Apply augmentation if requested
        if augment:
            X, y = augment_sequence(X, y)
        
        X_list.append(X)
        y_list.append(y)
        meta.append({"file": f, "frames": frames})

    return X_list, y_list, meta




def build_model(input_dim: int, class_weights: dict = None) -> tf.keras.Model:
   
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(None, input_dim)),
        GaussianNoise(0.01),  # Regularization through noise
        
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        
        LSTM(32, return_sequences=True, dropout=0.2),
        
        Dense(1, activation="sigmoid")
    ])
    
    # Use weighted loss if class imbalance
    if class_weights:
        loss = tf.keras.losses.BinaryCrossentropy()
    else:
        loss = "binary_crossentropy"
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def compute_sample_weights(y_list):
    
    # Flatten all labels
    all_labels = np.concatenate([y.flatten() for y in y_list])
    
    # Compute class weights
    classes = np.array([0, 1])
    class_weights_array = compute_class_weight('balanced', classes=classes, y=all_labels)
    
    print(f"📊 Class distribution: {np.bincount(all_labels)}")
    print(f"📊 Class weights: {class_weights_array}")
    
    # Create sample weights for each sequence
    sample_weights_list = []
    for y in y_list:
        weights = np.where(y == 1, class_weights_array[1], class_weights_array[0])
        sample_weights_list.append(weights)
    
    return sample_weights_list, {0: class_weights_array[0], 1: class_weights_array[1]}


def prepare_padded(X_list, y_list, sample_weights_list=None):
    
    max_len = max(seq.shape[0] for seq in X_list)
    
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype="float32", padding="post", value=0.0)
    y_padded = pad_sequences(y_list, maxlen=max_len, dtype="float32", padding="post", value=0.0)
    y_padded = np.expand_dims(y_padded, -1)
    
    if sample_weights_list:
        sw_padded = pad_sequences(sample_weights_list, maxlen=max_len, dtype="float32", padding="post", value=0.0)
        sw_padded = np.expand_dims(sw_padded, -1)
        return X_padded, y_padded, sw_padded
    
    return X_padded, y_padded, None


def multi_stage_filtering(probs: np.ndarray, 
                          knee_series: pd.Series,
                          fps: float) -> list:
   
    if fps <= 0:
        fps = FPS_FALLBACK
    
    # Stage 1: Initial threshold
    mask = probs >= PROB_THR
    
    if not mask.any():
        return []
    
    # Stage 2: Find segments
    segments = []
    in_rep = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_rep:
            in_rep = True
            start = i
        elif not val and in_rep:
            in_rep = False
            segments.append([start, i - 1])
    if in_rep:
        segments.append([start, len(mask) - 1])
    
    if not segments:
        return []
    
    # Stage 3: Merge gaps
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e - 1 <= GAP_MERGE_FR:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    
    # Stage 4: Length filtering with adaptive bounds
    lengths = np.array([e - s + 1 for s, e in merged], dtype=float)
    if len(lengths) == 0:
        return []
    
    med_len = float(np.median(lengths))
    min_len = max(int(0.8 * fps), int(LEN_MIN_SCALE * med_len))
    max_len = int(LEN_MAX_SCALE * med_len) if med_len > 0 else len(mask)
    
    filtered = []
    for s, e in merged:
        L = e - s + 1
        if L < min_len or L > max_len:
            continue
        
        # Stage 5: Knee ROM check
        if knee_series is not None and KNEE_ROM_MIN > 0.0:
            seg_knee = np.array(knee_series.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < KNEE_ROM_MIN:
                    continue
        
        # Stage 6: Confidence check (average probability in segment)
        seg_probs = probs[s:e+1]
        avg_conf = float(np.mean(seg_probs))
        if avg_conf < 0.70:  # Require high average confidence
            continue
        
        filtered.append((int(s), int(e)))
    
    return filtered


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
    df_frames, df_gt, files, optional_cols = load_data()
    
    print("\n🔧 Building sequences with augmentation...")
    X_list_orig, y_list_orig, meta = build_sequences(df_frames, df_gt, files, optional_cols, augment=False)
    
    # Compute class weights
    print("\n⚖️ Computing class weights...")
    sample_weights_list, class_weights = compute_sample_weights(y_list_orig)
    
    # 5-Fold Cross-Validation
    print(f"\n🔄 Starting {N_FOLDS}-Fold Cross-Validation...")
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_scores = []
    best_fold_score = 0
    best_model = None
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(files)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'='*80}")
        
        # Split data
        X_train_list = [X_list_orig[i] for i in train_idx]
        y_train_list = [y_list_orig[i] for i in train_idx]
        sw_train_list = [sample_weights_list[i] for i in train_idx]
        
        X_val_list = [X_list_orig[i] for i in val_idx]
        y_val_list = [y_list_orig[i] for i in val_idx]
        
        # Apply augmentation to training data only
        print("🎨 Applying data augmentation to training set...")
        X_train_aug, y_train_aug, sw_train_aug = [], [], []
        for X, y, sw in zip(X_train_list, y_train_list, sw_train_list):
            # Original
            X_train_aug.append(X)
            y_train_aug.append(y)
            sw_train_aug.append(sw)
            
            # Augmented version 1
            X_aug1, y_aug1 = augment_sequence(X.copy(), y.copy())
            X_train_aug.append(X_aug1)
            y_train_aug.append(y_aug1)
            sw_aug1 = np.where(y_aug1 == 1, class_weights[1], class_weights[0])
            sw_train_aug.append(sw_aug1)
        
        print(f"   Training samples: {len(X_train_aug)} (original: {len(X_train_list)})")
        
        # Pad sequences
        X_train_pad, y_train_pad, sw_train_pad = prepare_padded(X_train_aug, y_train_aug, sw_train_aug)
        X_val_pad, y_val_pad, _ = prepare_padded(X_val_list, y_val_list)
        
        # Build model
        model = build_model(input_dim=X_train_pad.shape[-1], class_weights=class_weights)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PATIENCE//2, verbose=1, min_lr=1e-6)
        ]
        
        # Train
        print(f"\n🧠 Training fold {fold + 1}...")
        history = model.fit(
            X_train_pad, y_train_pad,
            sample_weight=sw_train_pad,
            validation_data=(X_val_pad, y_val_pad),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            callbacks=callbacks
        )
        
        # Evaluate
        val_loss = min(history.history['val_loss'])
        fold_scores.append(val_loss)
        
        print(f"\n📊 Fold {fold + 1} validation loss: {val_loss:.4f}")
        
        # Keep best model
        if val_loss < best_fold_score or fold == 0:
            best_fold_score = val_loss
            best_model = model
            print(f"    New best model!")
    
    # Summary
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Mean validation loss: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Best fold loss: {best_fold_score:.4f}")
    
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model.save(BEST_MODEL_PATH)
    print(f"\n💾 Saved best model to: {BEST_MODEL_PATH}")
    
    
  
    print("FINAL EVALUATION")
    
    
    all_rows = []
    mask_ious, pair_ious, count_diffs = [], [], []
    
    for i, info in enumerate(meta):
        f = info["file"]
        frames = info["frames"]
        n_frames = len(frames)
        
        df_f = df_frames[df_frames["file"] == f].copy().sort_values("frame")
        X = X_list_orig[i]
        X_in = np.expand_dims(X, axis=0)
        
        # Predict
        probs = best_model.predict(X_in, verbose=0)[0, :, 0]
        
        # Smooth
        probs_smooth = pd.Series(probs).rolling(11, center=True, min_periods=1).mean().values
        
        # Multi-stage filtering
        knee_series = df_f["knee_deg"].astype(float)
        seg_pred = multi_stage_filtering(probs_smooth, knee_series, FPS_FALLBACK)
        
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
        
        print(f"\nFile: {f}")
        print(f"  GT   : {gt_segments}")
        print(f"  Pred : {pred_segments_global}")
        print(f"  mask_iou={m_iou:.3f}, pair_iou={p_iou:.3f}, count_diff={c_diff}")
        
        for (s, e) in pred_segments_global:
            all_rows.append({"file": f, "start_frame": s, "end_frame": e})
    
    # Final metrics
    avg_mask_iou = float(np.mean(mask_ious)) if mask_ious else 0.0
    avg_pair_iou = float(np.mean(pair_ious)) if pair_ious else 0.0
    avg_count_diff = float(np.mean(count_diffs)) if count_diffs else 0.0
    
    print(f"\n{'='*80}")
    print("📊 MAXIMUM PERFORMANCE MODEL - Final Metrics:")
    print(f"{'='*80}")
    print(f"  avg_mask_iou  : {avg_mask_iou:.3f}")
    print(f"  avg_pair_iou  : {avg_pair_iou:.3f}")
    print(f"  avg_count_diff: {avg_count_diff:.3f}")
    print(f"{'='*80}")
    
    # Save predictions
    df_out = pd.DataFrame(all_rows, columns=["file", "start_frame", "end_frame"])
    save_csv_overwrite(df_out, AUTO_SEGMENTS_OUTPATH)
    print(f"\n💾 Predictions saved to: {AUTO_SEGMENTS_OUTPATH}")


if __name__ == "__main__":
    main()