

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.signal import find_peaks
from InquirerPy import inquirer

import tensorflow as tf


LSTM_MODEL_PATH = os.path.join("models", "rep_segmenter_lstm.keras")


KEYPOINTS_CSV = os.path.join("frame_level", "pose_keypoints.csv")


OUTPUT_REPS_FOLDER = "segmented_reps"
AUTO_SEGMENTS_CSV = os.path.join("rep_level", "auto_segments.csv")


PROB_THR = 0.50
GAP_MERGE_FR = 10
LEN_MIN_SCALE = 0.80
LEN_MAX_SCALE = 1.60
KNEE_ROM_MIN = 20.0


FPS_FALLBACK = 25.0
PROM_FB = 0.02
MIN_REP_SEC_FB = 0.8
MAX_REP_SEC_FB = 3.5
KNEE_ROM_MIN_FB = 15.0

PAD_FRAMES_BEFORE = 5
PAD_FRAMES_AFTER = 5



def choose_video_folder():
    cwd = os.getcwd()
    choices = []
    

    priority = ['output_pose_videos', 'all_videos', 'videos', 'raw_videos']
    
    for name in priority:
        path = os.path.join(cwd, name)
        if os.path.isdir(path):
        
            vids = [f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            choices.append({"name": f"{name}/ ({len(vids)} videos)", "value": path})
    

    for name in sorted(os.listdir(cwd)):
        full = os.path.join(cwd, name)
        if os.path.isdir(full) and name not in priority:
            if not name.startswith('.') and name not in [
                '__pycache__', 'models', 'frame_level', 'rep_level',
                'segmented_reps', 'deleted_videos'
            ]:
                vids = [f for f in os.listdir(full) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if vids:
                    choices.append({"name": f"{name}/ ({len(vids)} videos)", "value": full})
    
    if not choices:
        print("❌ No video folders found")
        return None
    
    return inquirer.select(
        message="📁 Select folder containing videos to segment:",
        choices=choices
    ).execute()


def smooth_probs(probs, win=7):
    s = pd.Series(probs)
    s = s.rolling(win, center=True, min_periods=1).mean()
    return s.values.astype(float)


def segments_from_probs(probs, knee_series=None):
    mask = probs >= PROB_THR
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
            segments.append([start, i - 1])
    
    if in_rep:
        segments.append([start, n - 1])
    
    if not segments:
        return []
    
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e - 1 <= GAP_MERGE_FR:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    
    segments = merged

    lengths = np.array([e - s + 1 for s, e in segments], dtype=float)
    if len(lengths) == 0:
        return []
    
    med_len = float(np.median(lengths))
    min_len = max(3, int(LEN_MIN_SCALE * med_len))
    max_len = int(LEN_MAX_SCALE * med_len) if med_len > 0 else n
    
    filtered = []
    for s, e in segments:
        L = e - s + 1
        if L < min_len or L > max_len:
            continue
        
        if knee_series is not None and KNEE_ROM_MIN > 0:
            seg_knee = np.array(knee_series.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < KNEE_ROM_MIN:
                    continue
        
        filtered.append((int(s), int(e)))
    
    return filtered


def fallback_segment_from_hip(df_file, fps=None):
    if fps is None or fps <= 0:
        fps = FPS_FALLBACK
    
    if "hip_y" not in df_file.columns:
        return []
    
    y_raw = df_file["hip_y"].values.astype(float)
    

    s = pd.Series(y_raw, dtype="float64").ffill().bfill()
    if s.isna().all():
        return []
    
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-6:
        return []
    
    y_norm = ((s - mn) / (mx - mn)).values
    inv = -y_norm 
    
    min_dist = int(MIN_REP_SEC_FB * fps * 0.6)
    if min_dist < 3:
        min_dist = 3
    
    peaks, _ = find_peaks(inv, prominence=PROM_FB, distance=min_dist)
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
        
        s_idx = center
        while s_idx > left and y_norm[s_idx] <= thr:
            s_idx -= 1
        e_idx = center
        while e_idx < right and y_norm[e_idx] <= thr:
            e_idx += 1
        
        length = e_idx - s_idx + 1
        min_len = int(MIN_REP_SEC_FB * fps)
        max_len = int(MAX_REP_SEC_FB * fps)
        
        if length < min_len:
            extra = min_len - length
            s_idx = max(0, s_idx - extra // 2)
            e_idx = min(n - 1, e_idx + extra - extra // 2)
        
        if length > max_len:
            half = max_len // 2
            s_idx = max(0, center - half)
            e_idx = min(n - 1, s_idx + max_len - 1)
        
        reps.append((s_idx, e_idx))
    
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


def extract_rep_clip(video_path, start_frame, end_frame, output_path, fps=None):

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps or FPS_FALLBACK
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_padded = max(0, start_frame - PAD_FRAMES_BEFORE)
    end_padded = min(total_frames - 1, end_frame + PAD_FRAMES_AFTER)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_padded)
    
    for _ in range(end_padded - start_padded + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    
    cap.release()
    writer.release()
    
    return True


def main():
    print("\n" + "="*60)
    print("       SEGMENT VIDEOS INTO INDIVIDUAL REPS")
    print("="*60)
    
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"❌ Model not found: {LSTM_MODEL_PATH}")
        print("   Run train_rep_segmenter_lstm.py first!")
        return

    if not os.path.exists(KEYPOINTS_CSV):
        print(f"❌ Keypoints not found: {KEYPOINTS_CSV}")
        print("   Run extract_pose_keypoints.py first!")
        return

    print(f"\n🧠 Loading model: {LSTM_MODEL_PATH}")
    model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    print("   ✅ Model loaded")
 
    print(f"\n📊 Loading keypoints: {KEYPOINTS_CSV}")
    df_keypoints = pd.read_csv(KEYPOINTS_CSV)
    print(f"   {len(df_keypoints)} rows, {df_keypoints['file'].nunique()} videos")
 
    video_folder = choose_video_folder()
    if not video_folder:
        return
 
    video_files = [f for f in os.listdir(video_folder) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    print(f"\n📁 Video folder: {video_folder}")
    print(f"🎬 Videos found: {len(video_files)}")
    
    if "pose_videos" in video_folder:
        print("\n⚠️  Detected pose_videos folder - will map names back to originals")
    

    os.makedirs(OUTPUT_REPS_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(AUTO_SEGMENTS_CSV), exist_ok=True)
    
  
    all_segments = []
    total_reps = 0
    videos_processed = 0
    videos_skipped = 0
    
    print(f"\n🔄 Processing videos...\n")
    
    for video_file in tqdm(video_files, desc="Segmenting"):
        video_path = os.path.join(video_folder, video_file)
        
        if video_file.startswith("pose_"):
            keypoint_name = video_file[5:] 
        else:
            keypoint_name = video_file
        
       
        df_vid = df_keypoints[df_keypoints['file'] == keypoint_name].copy()
        
        if df_vid.empty:
          
            df_vid = df_keypoints[df_keypoints['file'] == video_file].copy()
        
        if df_vid.empty:
            videos_skipped += 1
            continue
        
        df_vid = df_vid.sort_values('frame').reset_index(drop=True)
        frames = df_vid['frame'].values.astype(int)
        n_frames = len(frames)
        
        if n_frames < 10:
            videos_skipped += 1
            continue
        
      
        hip = df_vid['hip_y'].astype(float).ffill().bfill()
        knee = df_vid['knee_deg'].astype(float).ffill().bfill()
        
        if hip.isna().all():
            hip = pd.Series(np.zeros(len(hip)))
        if knee.isna().all():
            knee = pd.Series(np.zeros(len(knee)))
        
        hip_norm = (hip - hip.mean()) / (hip.std() + 1e-6)
        knee_norm = (knee - knee.mean()) / (knee.std() + 1e-6)
        
        X = np.stack([hip_norm.values, knee_norm.values], axis=-1)
        X_in = np.expand_dims(X, axis=0)
        
      
        probs = model.predict(X_in, verbose=0)[0, :, 0]
        probs = smooth_probs(probs, win=7)
        
     
        knee_series = df_vid['knee_deg'].astype(float)
        segments = segments_from_probs(probs, knee_series)
        
     
        if not segments:
            segments = fallback_segment_from_hip(df_vid)
        
        if not segments:
            videos_skipped += 1
            continue
        

        segments_global = [(int(frames[s]), int(frames[e])) for s, e in segments]
        
      
        base_name = os.path.splitext(keypoint_name)[0]
        
        for rep_idx, (start_f, end_f) in enumerate(segments_global):
            output_name = f"{base_name}_rep{rep_idx:02d}.mp4"
            output_path = os.path.join(OUTPUT_REPS_FOLDER, output_name)
            
            success = extract_rep_clip(video_path, start_f, end_f, output_path)
            
            if success:
                all_segments.append({
                    'source_file': keypoint_name,
                    'rep_id': rep_idx,
                    'start_frame': start_f,
                    'end_frame': end_f,
                    'output_file': output_name
                })
                total_reps += 1
        
        videos_processed += 1
    
  
    df_segments = pd.DataFrame(all_segments)
    df_segments.to_csv(AUTO_SEGMENTS_CSV, index=False)
    

    print(f"\n📊 Results:")
    print(f"   Videos processed: {videos_processed}")
    print(f"   Videos skipped (no keypoints/reps): {videos_skipped}")
    print(f"   Total reps extracted: {total_reps}")
    print(f"   Average reps per video: {total_reps/max(1,videos_processed):.1f}")
    print(f"\n💾 Outputs:")
    print(f"   Rep clips: {OUTPUT_REPS_FOLDER}/")
    print(f"   Segments CSV: {AUTO_SEGMENTS_CSV}")


if __name__ == "__main__":
    main()