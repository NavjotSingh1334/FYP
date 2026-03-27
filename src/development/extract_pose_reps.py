import os, math
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from InquirerPy import inquirer
import mediapipe as mp


import tensorflow as tf
from tensorflow.keras.models import load_model


OUTPUT_VIDEOS     = "output_videos"
FRAME_CSV_PATH    = os.path.join("frame_level", "pose_keypoints.csv")
SEGMENTS_CSV_PATH = os.path.join("rep_level", "rep_segments.csv")
REPS_CSV_PATH     = os.path.join("rep_level", "rep_features.csv")
QC_REPORT_PATH    = os.path.join("reports", "qc_report.csv")
DEBUG_HIP_PATH    = os.path.join("reports", "debug_hip_series.csv")

# Clear ALL files inside output_videos each run
CLEAR_OUTPUT_VIDEOS = True

# Light QC
INFO_MIN_WIDTH, INFO_MIN_HEIGHT = 240, 240
BLUR_WARN_THRESHOLD = 40.0

# Hard reject
MIN_DURATION_SEC_HARD = 1.0

# Coverage gates
VALID_FRAME_RATIO_MIN = 0.60
JOINT_COVERAGE_MIN    = 0.65


LSTM_MODEL_PATH   = os.path.join("models", "rep_segmenter_lstm.keras")
FPS_FALLBACK      = 25.0  # used if fps is zero


PROB_THR          = 0.50   # probability threshold for "in rep" (was 0.45)
GAP_MERGE_FR      = 10     # merge small gaps shorter than this (frames) (was 8)
LEN_MIN_SCALE     = 0.80   # min length = LEN_MIN_SCALE * median rep length (was 0.6)
LEN_MAX_SCALE     = 1.60   # max length = LEN_MAX_SCALE * median rep length (was 1.8)
KNEE_ROM_MIN      = 20.0   # drop segments where knee ROM < this (deg)

# Fallback rule-only (used ONLY if LSTM finds no reps)
PROM_FB           = 0.02
MIN_REP_SEC_FB    = 1.0
MAX_REP_SEC_FB    = 4.0
KNEE_ROM_MIN_FB   = 15.0

INTERP_MAX_GAP_FRAMES   = 6
SMOOTH_WINDOW_FRAMES    = 5
FALLBACK_MIN_HIP_RANGE_NORM = 0.06  # last-last resort (almost never used)


# Rep padding (extra context in rep videos)
REP_PADDING_SEC = 0.35  # tweak 0.25–0.45 if needed

# Rule thresholds (fixed)
KNEE_SHALLOW_DEG         = 140.0
TRUNK_LEAN_EXCESSIVE_DEG = 35.0
ANKLE_LIMITED_DEG        = 15.0

# Visibility
VIEW_TIP_THRESHOLD = 0.25
VIS_THR            = 0.6

os.makedirs(OUTPUT_VIDEOS, exist_ok=True)
os.makedirs(os.path.dirname(FRAME_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SEGMENTS_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(QC_REPORT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DEBUG_HIP_PATH), exist_ok=True)

# Full wipe of output_videos contents each run
if CLEAR_OUTPUT_VIDEOS:
    for f in os.listdir(OUTPUT_VIDEOS):
        fpath = os.path.join(OUTPUT_VIDEOS, f)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"[WARN] Could not delete {fpath}: {e}")

def save_csv_safely(df_new, path):
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(inplace=True)  
    else:
        df_combined = df_new
    df_combined.to_csv(path, index=False)



def choose_input_folder():
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)

    choices = []

    # Prefer folders named "videos" if they exist
    cand1 = os.path.join(cwd, "videos")
    cand2 = os.path.join(parent, "videos")

    if os.path.isdir(cand1):
        choices.append(("videos (./videos)", cand1))
    if os.path.isdir(cand2) and cand2 != cand1:
        choices.append(("videos (../videos)", cand2))

    # Add subdirectories of current dir as additional options
    for name in sorted(os.listdir(cwd)):
        full = os.path.join(cwd, name)
        if os.path.isdir(full) and full not in [c[1] for c in choices]:
            choices.append((name, full))

    if not choices:
        print("⚠️ No subfolders found. Using current directory.")
        return cwd

    answer = inquirer.select(
        message="📁 Select folder containing squat videos:",
        choices=[{"name": n, "value": p} for (n, p) in choices],
    ).execute()

    return answer


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
P = mp_pose.PoseLandmark
CORE_PAIRS = [
    P.RIGHT_HIP, P.RIGHT_KNEE, P.RIGHT_ANKLE, P.RIGHT_SHOULDER,
    P.LEFT_HIP,  P.LEFT_KNEE,  P.LEFT_ANKLE,  P.LEFT_SHOULDER
]


def _to_xy(p):
    return (float(p[0]), float(p[1])) if p and len(p) == 2 else None

def angle_3pts(a, b, c):
    A, B, C = _to_xy(a), _to_xy(b), _to_xy(c)
    if None in (A, B, C):
        return np.nan
    ax, ay = A; bx, by = B; cx, cy = C
    abx, aby = ax - bx, ay - by
    cbx, cby = cx - bx, cy - by
    nab = math.hypot(abx, aby); ncb = math.hypot(cbx, cby)
    if nab == 0 or ncb == 0:
        return np.nan
    cosang = max(-1.0, min(1.0, (abx*cbx + aby*cby) / (nab * ncb)))
    return math.degrees(math.acos(cosang))

def trunk_angle_deg(shoulder_xy, hip_xy):
    S, H = _to_xy(shoulder_xy), _to_xy(hip_xy)
    if None in (S, H):
        return np.nan
    sx, sy = S; hx, hy = H
    vx, vy = sx - hx, sy - hy
    if vx == 0 and vy == 0:
        return np.nan
    ang = abs(math.degrees(math.atan2(vy, vx)))
    return abs(90 - ang)

def interp_nans(series, max_gap=6):
    s = pd.Series(series, dtype="float64")
    return list(s.interpolate(limit=max_gap, limit_direction="both"))

def smooth_series(series, win=5):
    s = pd.Series(series, dtype="float64")
    return list(s.rolling(win, center=True, min_periods=1).median())



def smooth_probs(probs, win=5):
    s = pd.Series(probs, dtype="float64")
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

    # Find contiguous regions where mask == 1
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

    # Merge small gaps
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e - 1 <= gap_merge_fr:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    segments = merged

    # Length-based filtering
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

        # optional knee ROM filter
        if knee_series is not None and knee_rom_min > 0.0:
            seg_knee = np.array(knee_series.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < knee_rom_min:
                    continue
        filtered.append((int(s), int(e)))

    return filtered

def _norm_series_safe(y):
    s = pd.Series(y, dtype="float64").ffill().bfill()
    if s.isna().all():
        return np.zeros_like(s.values, dtype=float)
    mn = float(s.min())
    mx = float(s.max())
    if mx - mn < 1e-6:
        return np.zeros_like(s.values, dtype=float)
    return ((s - mn) / (mx - mn)).values

def fallback_segment_from_hip(hip_y_series, knee_series, fps: float = None):
    
    if fps is None or fps <= 0:
        fps = FPS_FALLBACK

    y_raw = np.asarray(hip_y_series, dtype=float)
    if len(y_raw) < 10:
        return []

    y_norm = _norm_series_safe(y_raw)
    inv = -y_norm  # squat bottom → peak in inv

    min_dist_frames = int(MIN_REP_SEC_FB * fps * 0.6)
    if min_dist_frames < 3:
        min_dist_frames = 3

    peaks, _ = find_peaks(inv, prominence=PROM_FB, distance=min_dist_frames)
    if len(peaks) == 0:
        # last-last resort: if we clearly see hip range, one big rep
        hip_range = float(np.nanmax(y_norm) - np.nanmin(y_norm)) if len(y_norm) else 0.0
        if hip_range > FALLBACK_MIN_HIP_RANGE_NORM:
            return [(0, len(y_norm) - 1)]
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

    # merge overlaps
    reps_sorted = sorted(reps, key=lambda x: x[0])
    merged = []
    for s, e in reps_sorted:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    reps_clean = []
    knee = pd.Series(knee_series, dtype="float64") if knee_series is not None else None
    for s, e in merged:
        if knee is not None:
            seg_knee = np.array(knee.values[s:e+1], dtype=float)
            if np.isfinite(seg_knee).any():
                rom = float(np.nanmax(seg_knee) - np.nanmin(seg_knee))
                if rom < KNEE_ROM_MIN_FB:
                    continue
        reps_clean.append((int(s), int(e)))

    return reps_clean

def run_lstm_segmentation(hip_y_series, knee_deg_series, fps, model):
    
    hip = pd.Series(hip_y_series, dtype="float64").ffill().bfill()
    knee = pd.Series(knee_deg_series, dtype="float64").ffill().bfill()

    if hip.isna().all():
        hip = pd.Series(np.zeros(len(hip)))
    if knee.isna().all():
        knee = pd.Series(np.zeros(len(knee)))

    hip_norm  = (hip  - hip.mean())  / (hip.std()  + 1e-6)
    knee_norm = (knee - knee.mean()) / (knee.std() + 1e-6)

    X = np.stack([hip_norm.values, knee_norm.values], axis=-1)  # [T, 2]
    X_in = np.expand_dims(X, axis=0)  # [1, T, 2]

    probs = model.predict(X_in, verbose=0)[0, :, 0]
    probs = smooth_probs(probs, win=5)

    segs = segments_from_probs(
        probs,
        fps=fps if fps > 0 else FPS_FALLBACK,
        len_min_scale=LEN_MIN_SCALE,
        len_max_scale=LEN_MAX_SCALE,
        prob_thr=PROB_THR,
        gap_merge_fr=GAP_MERGE_FR,
        knee_series=knee,
        knee_rom_min=KNEE_ROM_MIN
    )
    return segs


def friendly_feedback(flags, view_score):
    cues = {
        "shallow_depth": "Go a little deeper — sit back more.",
        "excessive_lean": "Keep your chest up — brace your core.",
        "limited_ankle": "Keep heels down — work on ankle mobility."
    }
    order = [f for f in ["shallow_depth", "excessive_lean", "limited_ankle"] if f in flags]
    msg = " | ".join([cues[f] for f in order[:2]]) if order else "Nice rep — form looks solid!"
    if view_score is not None and view_score > VIEW_TIP_THRESHOLD:
        msg = f"{msg}  Tip: a more side-on camera angle will improve accuracy."
    return msg

def blur_score_bgr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def frame_valid_visibility(lm, idxs, vis_thr=0.6):
    vis = [lm[i].visibility for i in idxs]
    return float(np.mean(vis)) >= vis_thr

def view_score_sample(lm):
    return abs(lm[P.RIGHT_SHOULDER].x - lm[P.LEFT_SHOULDER].x)


INPUT_FOLDER = choose_input_folder()
if not INPUT_FOLDER or not os.path.isdir(INPUT_FOLDER):
    print("❌ Invalid folder. Exiting.")
    raise SystemExit

videos = [f for f in os.listdir(INPUT_FOLDER)
          if f.lower().endswith((".mp4", ".avi", ".mov"))]

if not videos:
    print(f"❌ No video files found in: {INPUT_FOLDER}")
    raise SystemExit

print(f"\n📁 Using folder: {INPUT_FOLDER}")
print(f"🎞 Found {len(videos)} video file(s).\n")


if not os.path.exists(LSTM_MODEL_PATH):
    print(f"❌ Could not find LSTM model at: {LSTM_MODEL_PATH}")
    print("   Train & save it with train_rep_segmenter_lstm.py first.")
    raise SystemExit

print(f"🧠 Loading LSTM rep segmenter from {LSTM_MODEL_PATH} ...")
lstm_model = load_model(LSTM_MODEL_PATH)
print("✅ LSTM model loaded.\n")


frame_rows, segment_rows, rep_rows, qc_rows, debug_hip = [], [], [], [], []


with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    for fname in videos:
        path = os.path.join(INPUT_FOLDER, fname)
        cap  = cv2.VideoCapture(path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 320
        H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
        nframes  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = nframes / (fps if fps > 0 else FPS_FALLBACK)

        ok, first_frame = cap.read()
        blur_first = blur_score_bgr(first_frame) if ok else None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        qc_reason = []
        if W < INFO_MIN_WIDTH or H < INFO_MIN_HEIGHT:
            qc_reason.append("very_low_res")
        if blur_first is not None and blur_first < BLUR_WARN_THRESHOLD:
            qc_reason.append("very_blurry")

         
        if not ok:
            qc_rows.append({
                "file": fname, "width": W, "height": H, "fps": fps,
                "duration_sec": duration, "blur_score": blur_first,
                "valid_frame_ratio": 0.0,
                "hip_cov": 0.0, "knee_cov": 0.0, "ankle_cov": 0.0,
                "view_score_median": None,
                "decision": "rejected",
                "reasons": "unreadable"
            })
            print(f"⛔ Skipping {fname}: completely unreadable.")
            cap.release()
            continue

        overlay_frames = []
        hip_y_series, knee_deg_series, hip_deg_series = [], [], []
        trunk_deg_series, ankle_deg_series, side_used_list = [], [], []
        valid_mask = []
        hip_vis_frames = knee_vis_frames = ankle_vis_frames = 0
        view_samples = []

        print(f"🎥 Processing {fname} ({W}x{H} @ {fps:.1f}fps)")
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            knee_deg = hip_deg = trunk_deg = ankle_deg = np.nan
            use_left = 0
            hip_y    = np.nan

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark

                is_valid = frame_valid_visibility(lm, CORE_PAIRS, vis_thr=VIS_THR)
                valid_mask.append(1 if is_valid else 0)

                if is_valid and len(view_samples) < 30:
                    view_samples.append(view_score_sample(lm))

                vlh = lm[P.LEFT_HIP].visibility
                vlk = lm[P.LEFT_KNEE].visibility
                vla = lm[P.LEFT_ANKLE].visibility
                vls = lm[P.LEFT_SHOULDER].visibility

                vrh = lm[P.RIGHT_HIP].visibility
                vrk = lm[P.RIGHT_KNEE].visibility
                vra = lm[P.RIGHT_ANKLE].visibility
                vrs = lm[P.RIGHT_SHOULDER].visibility

                hip_vis_frames   += 1 if (vlh >= VIS_THR or vrh >= VIS_THR) else 0
                knee_vis_frames  += 1 if (vlk >= VIS_THR or vrk >= VIS_THR) else 0
                ankle_vis_frames += 1 if (vla >= VIS_THR or vra >= VIS_THR) else 0

                left_sum  = vlh + vlk + vla
                right_sum = vrh + vrk + vra
                use_left  = int(left_sum >= right_sum)

                if use_left:
                    hip_pt   = (lm[P.LEFT_HIP].x,   lm[P.LEFT_HIP].y);   v_hip   = vlh
                    knee_pt  = (lm[P.LEFT_KNEE].x,  lm[P.LEFT_KNEE].y);  v_knee  = vlk
                    ankle_pt = (lm[P.LEFT_ANKLE].x, lm[P.LEFT_ANKLE].y); v_ankle = vla
                    shoulder = (lm[P.LEFT_SHOULDER].x, lm[P.LEFT_SHOULDER].y); v_sh = vls
                else:
                    hip_pt   = (lm[P.RIGHT_HIP].x,   lm[P.RIGHT_HIP].y);   v_hip   = vrh
                    knee_pt  = (lm[P.RIGHT_KNEE].x,  lm[P.RIGHT_KNEE].y);  v_knee  = vrk
                    ankle_pt = (lm[P.RIGHT_ANKLE].x, lm[P.RIGHT_ANKLE].y); v_ankle = vra
                    shoulder = (lm[P.RIGHT_SHOULDER].x, lm[P.RIGHT_SHOULDER].y); v_sh = vrs

                hip_y = float(hip_pt[1])

                if v_hip >= VIS_THR and v_knee >= VIS_THR and v_ankle >= VIS_THR:
                    knee_deg = angle_3pts(hip_pt, knee_pt, ankle_pt)

                if v_sh >= VIS_THR and v_hip >= VIS_THR and v_knee >= VIS_THR:
                    hip_deg = angle_3pts(shoulder, hip_pt, knee_pt)

                if v_sh >= VIS_THR and v_hip >= VIS_THR:
                    trunk_deg = trunk_angle_deg(shoulder, hip_pt)

                if v_knee >= VIS_THR and v_ankle >= VIS_THR:
                    ankle_proxy = (ankle_pt[0], ankle_pt[1] - 0.05)
                    ankle_deg = angle_3pts(knee_pt, ankle_pt, ankle_proxy)

                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            else:
                valid_mask.append(0)

            overlay_frames.append(frame.copy())

            frame_rows.append({
                "file": fname, "frame": frame_idx, "use_left": use_left,
                "hip_y": hip_y, "knee_deg": knee_deg, "hip_deg": hip_deg,
                "trunk_deg": trunk_deg, "ankle_deg": ankle_deg,
                "valid": valid_mask[-1]
            })

            hip_y_series.append(hip_y)
            knee_deg_series.append(knee_deg)
            hip_deg_series.append(hip_deg)
            trunk_deg_series.append(trunk_deg)
            ankle_deg_series.append(ankle_deg)
            side_used_list.append(use_left)

            if frame_idx < 120:
                debug_hip.append({"file": fname, "frame": frame_idx, "hip_y": hip_y})

            frame_idx += 1

        cap.release()

        total_frames = max(1, len(valid_mask))
        valid_ratio  = sum(valid_mask) / total_frames
        hip_cov   = hip_vis_frames   / total_frames
        knee_cov  = knee_vis_frames  / total_frames
        ankle_cov = ankle_vis_frames / total_frames
        view_score_median = float(np.median(view_samples)) if view_samples else None

        decision = "accepted"
        reasons = list(qc_reason)
        
        if valid_ratio < VALID_FRAME_RATIO_MIN:
            reasons.append("insufficient_valid_frames")
        if min(hip_cov, knee_cov, ankle_cov) < JOINT_COVERAGE_MIN:
            reasons.append("low_joint_coverage")

        qc_rows.append({
            "file": fname, "width": W, "height": H, "fps": fps,
            "duration_sec": duration, "blur_score": blur_first,
            "valid_frame_ratio": valid_ratio,
            "hip_cov": hip_cov, "knee_cov": knee_cov, "ankle_cov": ankle_cov,
            "view_score_median": view_score_median,
            "decision": decision, "reasons": ",".join(reasons) if reasons else ""
        })

        
        try:
            raw_hip = np.array(hip_y_series, dtype=float)
            if np.all(np.isnan(raw_hip)):
                hip_motion = 0.0
            else:
                hip_motion = float(np.nanmax(raw_hip) - np.nanmin(raw_hip))
        except ValueError:
            hip_motion = 0.0

        if hip_motion < 0.05:
           
            print(f"⚠️  Warning for {fname}: very low hip motion detected (motion={hip_motion:.3f})")
            qc_rows[-1]["reasons"] = qc_rows[-1]["reasons"] + ",low_hip_motion" if qc_rows[-1]["reasons"] else "low_hip_motion"

        
        seg_lstm = run_lstm_segmentation(hip_y_series, knee_deg_series, fps, lstm_model)

        if not seg_lstm:
            # fallback: rule-based on hip_y
            seg_pred = fallback_segment_from_hip(hip_y_series, knee_deg_series, fps=fps)
        else:
            seg_pred = seg_lstm

        # still nothing? (very rare) → last resort: one full rep if hip moved enough
        if not seg_pred:
            hip_range = float(np.nanmax(raw_hip) - np.nanmin(raw_hip)) if len(hip_y_series) else 0.0
            if hip_range >= FALLBACK_MIN_HIP_RANGE_NORM:
                seg_pred = [(0, len(hip_y_series) - 1)]

        reps = seg_pred

        
        def _clean(x):
            return smooth_series(interp_nans(x, INTERP_MAX_GAP_FRAMES), SMOOTH_WINDOW_FRAMES)

        hip_y_series     = _clean(hip_y_series)
        knee_deg_series  = _clean(knee_deg_series)
        hip_deg_series   = _clean(hip_deg_series)
        trunk_deg_series = _clean(trunk_deg_series)
        ankle_deg_series = _clean(ankle_deg_series)

        base_name, _ = os.path.splitext(fname)

        for rep_id, (s, e) in enumerate(reps):
            rep_valid_ratio = float(np.nanmean(valid_mask[s:e+1]))
            if rep_valid_ratio < 0.55:
                continue

            segment_rows.append({"file": fname, "rep_id": rep_id,
                                 "start_frame": s, "end_frame": e})

            fps_eff = fps if fps > 0 else FPS_FALLBACK
            pad_frames = int(REP_PADDING_SEC * fps_eff)
            s_vid = max(0, s - pad_frames)
            e_vid = min(len(overlay_frames) - 1, e + pad_frames)

            rep_video_name = f"pose_{base_name}_rep{rep_id}.mp4"
            rep_video_path = os.path.join(OUTPUT_VIDEOS, rep_video_name)

            out_rep = cv2.VideoWriter(
                rep_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_eff,
                (W, H)
            )

            for f_idx in range(s_vid, e_vid + 1):
                out_rep.write(overlay_frames[f_idx])
            out_rep.release()
            print(f"🎬 Saved rep video → {rep_video_path}")

            
            k = np.array(knee_deg_series[s:e+1], dtype=float)
            h = np.array(hip_deg_series[s:e+1], dtype=float)
            t = np.array(trunk_deg_series[s:e+1], dtype=float)
            a = np.array(ankle_deg_series[s:e+1], dtype=float)

            flags = []
            knee_min  = float(np.nanmin(k))
            knee_mean = float(np.nanmean(k))
            hip_min   = float(np.nanmin(h))
            hip_mean  = float(np.nanmean(h))
            trunk_max = float(np.nanmax(t))
            trunk_mean= float(np.nanmean(t))
            ankle_max = float(np.nanmax(a))
            knee_rom  = float(np.nanmax(k) - np.nanmin(k))
            hip_rom   = float(np.nanmax(h) - np.nanmin(h))

            if knee_min  > KNEE_SHALLOW_DEG:          flags.append("shallow_depth")
            if trunk_max > TRUNK_LEAN_EXCESSIVE_DEG:  flags.append("excessive_lean")
            if ankle_max < ANKLE_LIMITED_DEG:         flags.append("limited_ankle")

            rep_rows.append({
                "file": fname,
                "rep_id": rep_id,
                "rep_video": rep_video_name,
                "frames": (e - s + 1),
                "knee_min":  knee_min,
                "knee_mean": knee_mean,
                "hip_min":   hip_min,
                "hip_mean":  hip_mean,
                "trunk_max": trunk_max,
                "trunk_mean":trunk_mean,
                "ankle_max": ankle_max,
                "knee_rom":  knee_rom,
                "hip_rom":   hip_rom,
                "view_score": view_score_median,
                "label_rule": "risky" if flags else "safe",
                "flags":      ",".join(flags),
                "feedback":   friendly_feedback(flags, view_score_median),
            })


save_csv_safely(pd.DataFrame(frame_rows),   FRAME_CSV_PATH)
save_csv_safely(pd.DataFrame(segment_rows), SEGMENTS_CSV_PATH)
save_csv_safely(pd.DataFrame(rep_rows),     REPS_CSV_PATH)
save_csv_safely(pd.DataFrame(qc_rows),      QC_REPORT_PATH)
save_csv_safely(pd.DataFrame(debug_hip),    DEBUG_HIP_PATH)

print("\n📄 Saved:")
print("  •", FRAME_CSV_PATH)
print("  •", SEGMENTS_CSV_PATH)
print("  •", REPS_CSV_PATH)
print("  •", QC_REPORT_PATH)
print("  •", DEBUG_HIP_PATH)
print("\n✅ Done.")