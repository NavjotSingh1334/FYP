

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import pad_sequences


FRAME_CSV_PATH        = os.path.join("frame_level", "pose_keypoints.csv")
MANUAL_SEGMENTS_PATH  = os.path.join("rep_level",  "manual_segments.csv")
AUTO_SEGMENTS_OUTPATH = os.path.join("rep_level",  "auto_segments_lstm.csv")

LSTM_MODEL_PATH = os.path.join("models", "rep_segmenter_lstm.keras")

FPS_FALLBACK          = 25.0 


PROB_THR       = 0.50   
GAP_MERGE_FR   = 10     
LEN_MIN_SCALE  = 0.80   
LEN_MAX_SCALE  = 1.60   
KNEE_ROM_MIN   = 20.0   


FPS_FALLBACK_FB   = 25.0
PROM_FB           = 0.02
MIN_REP_SEC_FB    = 0.8
MAX_REP_SEC_FB    = 3.5
KNEE_ROM_MIN_FB   = 15.0

RANDOM_SEED       = 42
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




def load_data():
    if not os.path.exists(FRAME_CSV_PATH):
        raise FileNotFoundError(f"Could not find {FRAME_CSV_PATH}")

    if not os.path.exists(MANUAL_SEGMENTS_PATH):
        raise FileNotFoundError(f"Could not find {MANUAL_SEGMENTS_PATH}")

    print(f"📥 Loading frame-level data from {FRAME_CSV_PATH}")
    df_frames = pd.read_csv(FRAME_CSV_PATH)

    needed_cols = {"file", "frame", "hip_y", "knee_deg"}
    if not needed_cols.issubset(df_frames.columns):
        missing = needed_cols - set(df_frames.columns)
        raise ValueError(f"pose_keypoints.csv missing columns: {missing}")

    print(f"📥 Loading manual segments from {MANUAL_SEGMENTS_PATH}")
    df_gt = pd.read_csv(MANUAL_SEGMENTS_PATH)

    if not {"file", "start_frame", "end_frame"}.issubset(df_gt.columns):
        raise ValueError("manual_segments.csv must have columns: file,start_frame,end_frame")

    
    common_files = sorted(set(df_frames["file"]).intersection(set(df_gt["file"])))
    df_frames = df_frames[df_frames["file"].isin(common_files)].copy()
    df_gt     = df_gt[df_gt["file"].isin(common_files)].copy()

    print(f"✅ Found {len(common_files)} labelled video(s).")
    return df_frames, df_gt, common_files




def build_sequences(df_frames: pd.DataFrame,
                    df_gt: pd.DataFrame,
                    files: list):
    
    X_list = []
    y_list = []
    meta   = []  

    for f in files:
        df_f = df_frames[df_frames["file"] == f].copy()
        df_f = df_f.sort_values("frame").reset_index(drop=True)
        frames = df_f["frame"].values.astype(int)

       
        hip = df_f["hip_y"].astype(float)
        knee = df_f["knee_deg"].astype(float)

       
        hip = hip.ffill().bfill()
        knee = knee.ffill().bfill()

     
        if hip.isna().all():
            hip = pd.Series(np.zeros(len(hip)))
        if knee.isna().all():
            knee = pd.Series(np.zeros(len(knee)))

      
        hip_norm = (hip - hip.mean()) / (hip.std() + 1e-6)
        knee_norm = (knee - knee.mean()) / (knee.std() + 1e-6)

        X = np.stack([hip_norm.values, knee_norm.values], axis=-1)

      
        y = np.zeros(len(frames), dtype=int)
        df_g = df_gt[df_gt["file"] == f]
        for _, row in df_g.iterrows():
            s = int(row["start_frame"])
            e = int(row["end_frame"])
            mask = (frames >= s) & (frames <= e)
            y[mask] = 1

        X_list.append(X)
        y_list.append(y)
        meta.append({"file": f, "frames": frames})

    return X_list, y_list, meta




def build_model(input_dim: int) -> tf.keras.Model:
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(None, input_dim)),
        LSTM(64, return_sequences=True),
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
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype="float32",
                             padding="post", value=0.0)
    y_padded = pad_sequences(y_list, maxlen=max_len, dtype="float32",
                             padding="post", value=0.0)
    y_padded = np.expand_dims(y_padded, -1)  # shape: (N, T, 1)
    return X_padded, y_padded




def smooth_probs(probs: np.ndarray, win: int = 7) -> np.ndarray:

    s = pd.Series(probs)
    s = s.rolling(win, center=True, min_periods=1).mean()
    return s.values.astype(float)


def segments_from_probs(probs: np.ndarray,
                        fps: float,
                        len_min_scale: float,
                        len_max_scale: float,
                        prob_thr: float,
                        gap_merge_fr: int,
                        knee_series: pd.Series = None,
                        knee_rom_min: float = 0.0):
    
    if fps <= 0:
        fps = FPS_FALLBACK

  
    mask = probs >= prob_thr
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
    min_len = max(3, int(len_min_scale * med_len))
    max_len = int(len_max_scale * med_len) if med_len > 0 else n

    filtered = []
    for s, e in segments:
        L = e - s + 1
        if L < min_len or L > max_len:
            continue

      
        if knee_series is not None and knee_rom_min > 0.0:
            seg_knee = np.array(knee_series.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < knee_rom_min:
                    continue
        filtered.append((int(s), int(e)))

    return filtered




def _norm_series_safe(y):
    s = pd.Series(y, dtype="float64")
    s = s.ffill().bfill()
    if s.isna().all():
        return np.zeros_like(s.values, dtype=float)
    mn = float(s.min())
    mx = float(s.max())
    if mx - mn < 1e-6:
        return np.zeros_like(s.values, dtype=float)
    return ((s - mn) / (mx - mn)).values


def fallback_segment_from_hip(df_file: pd.DataFrame, fps: float = None):
    
    if fps is None or fps <= 0:
        fps = FPS_FALLBACK_FB

    if "hip_y" not in df_file.columns:
        return []

    y_raw = df_file["hip_y"].values.astype(float)
    y_norm = _norm_series_safe(y_raw)
    inv = -y_norm 

    min_dist_frames = int(MIN_REP_SEC_FB * fps * 0.6)
    if min_dist_frames < 3:
        min_dist_frames = 3

    peaks, _ = find_peaks(inv, prominence=PROM_FB, distance=min_dist_frames)
    if len(peaks) == 0:
        return []

    n = len(y_norm)
    reps = []

    for center in peaks:
        half_win = int(MAX_REP_SEC_FB * fps / 2)
        left = max(0, center - half_win)
        right = min(n - 1, center + half_win)

        local = y_norm[left:right+1]
        if not np.isfinite(local).any():
            continue

        valley_val = y_norm[center]
        local_max = float(np.nanmax(local))
        thr = valley_val + 0.4 * (local_max - valley_val)

        s = center
        while s > left and y_norm[s] <= thr:
            s -= 1
        e = center
        while e < right and y_norm[e] <= thr:
            e += 1

        length = e - s + 1
        min_len = int(MIN_REP_SEC_FB * fps)
        max_len = int(MAX_REP_SEC_FB * fps)

        if length < min_len:
            extra = min_len - length
            s = max(0, s - extra // 2)
            e = min(n - 1, e + extra - extra // 2)
            length = e - s + 1
        if length > max_len:
            half = max_len // 2
            s = max(0, center - half)
            e = min(n - 1, s + max_len - 1)
            length = e - s + 1

        if length < min_len or length > max_len:
            continue

        reps.append((s, e))

    reps_sorted = sorted(reps, key=lambda x: x[0])
    merged = []
    for s, e in reps_sorted:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    reps_clean = []
    knee = df_file.get("knee_deg", None)
    for s, e in merged:
        if knee is not None:
            seg_knee = np.array(knee.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < KNEE_ROM_MIN_FB:
                    continue
        reps_clean.append((int(s), int(e)))

    return reps_clean



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


    X_list, y_list, meta = build_sequences(df_frames, df_gt, files)

  
    X_padded, y_padded = prepare_padded(X_list, y_list)


    idx = np.arange(len(files))
    idx_train, idx_val = train_test_split(idx, test_size=0.25,
                                          random_state=RANDOM_SEED, shuffle=True)

    X_train, y_train = X_padded[idx_train], y_padded[idx_train]
    X_val, y_val     = X_padded[idx_val],   y_padded[idx_val]

    model = build_model(input_dim=X_padded.shape[-1])
    print(model.summary())

    print("🧠 Training LSTM rep segmenter...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=4,
        verbose=2
    )

    print("\n✅ Training complete.\n")

    
    os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    print(f"💾 Saved trained LSTM model to: {LSTM_MODEL_PATH}\n")


    all_rows = []
    mask_ious = []
    pair_ious = []
    count_diffs = []

    print("🔍 Per-file check vs manual labels:\n")

    for i, info in enumerate(meta):
        f = info["file"]
        frames = info["frames"]
        n_frames = len(frames)

        df_f = df_frames[df_frames["file"] == f].copy().sort_values("frame")
        X = X_list[i] 
        X_in = np.expand_dims(X, axis=0)

      
        probs = model.predict(X_in, verbose=0)[0, :, 0]
        probs = smooth_probs(probs, win=7)

   
        knee_series = df_f["knee_deg"].astype(float)
        seg_lstm = segments_from_probs(
            probs,
            fps=FPS_FALLBACK,
            len_min_scale=LEN_MIN_SCALE,
            len_max_scale=LEN_MAX_SCALE,
            prob_thr=PROB_THR,
            gap_merge_fr=GAP_MERGE_FR,
            knee_series=knee_series,
            knee_rom_min=KNEE_ROM_MIN
        )

        if not seg_lstm:
            seg_pred = fallback_segment_from_hip(df_f, fps=FPS_FALLBACK)
           
        else:
            seg_pred = seg_lstm

        
        pred_segments_global = [(int(frames[s]), int(frames[e])) for (s, e) in seg_pred]

        
        df_g = df_gt[df_gt["file"] == f]
        gt_segments = [(int(r["start_frame"]), int(r["end_frame"])) for _, r in df_g.iterrows()]

        
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

        m_iou  = iou_masks(gt_mask, pred_mask)
        p_iou  = pairwise_iou(gt_segments, pred_segments_global)
        c_diff = len(pred_segments_global) - len(gt_segments)

        mask_ious.append(m_iou)
        pair_ious.append(p_iou)
        count_diffs.append(c_diff)

        print(f"File: {f}")
        print(f"  GT   : {gt_segments}")
        print(f"  Pred : {pred_segments_global}")
        print(f"  mask_iou={m_iou:.3f}, pair_iou={p_iou:.3f}, count_diff={c_diff}\n")

        for (s, e) in pred_segments_global:
            all_rows.append({"file": f, "start_frame": s, "end_frame": e})

   
    avg_mask_iou   = float(np.mean(mask_ious)) if mask_ious else 0.0
    avg_pair_iou   = float(np.mean(pair_ious)) if pair_ious else 0.0
    avg_count_diff = float(np.mean(count_diffs)) if count_diffs else 0.0

    print("📊 Global metrics vs manual labels:")
    print(f"  avg_mask_iou  : {avg_mask_iou:.3f}")
    print(f"  avg_pair_iou  : {avg_pair_iou:.3f}")
    print(f"  avg_count_diff: {avg_count_diff:.3f}")

    
    df_out = pd.DataFrame(all_rows, columns=["file", "start_frame", "end_frame"])
    save_csv_overwrite(df_out, AUTO_SEGMENTS_OUTPATH)
    print(f"\n💾 Predicted segments saved to {AUTO_SEGMENTS_OUTPATH}")


if __name__ == "__main__":
    main()