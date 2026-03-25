

import streamlit as st
import os, sys, json, time, tempfile, glob
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple

st.set_page_config(
    page_title="Squat Form Analyser",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FRAME_LEVEL_DIR = os.path.join(PROJECT_ROOT, "frame_level")
REP_LEVEL_DIR = os.path.join(PROJECT_ROOT, "rep_level")
OUTPUT_POSE_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "output_pose_videos")

KEYPOINTS_CSV = os.path.join(FRAME_LEVEL_DIR, "pose_keypoints.csv")
KEYPOINTS_RAW_CSV = os.path.join(FRAME_LEVEL_DIR, "pose_keypoints_raw.csv")
LABELS_CSV = os.path.join(REP_LEVEL_DIR, "rep_labels_merged.csv")
SEGMENTS_CSV = os.path.join(REP_LEVEL_DIR, "auto_segments.csv")
THRESHOLDS_JSON = os.path.join(MODELS_DIR, "feedback_thresholds.json")
COMPARISON_JSON = os.path.join(RESULTS_DIR, "model_comparison.json")


MODEL_REGISTRY = {
    "LightGBM": {
        "type": "baseline",
        "classifier_path": os.path.join(MODELS_DIR, "lightgbm_classifier.joblib"),
        "scaler_path": None,
        "features_path": os.path.join(MODELS_DIR, "lightgbm_features.joblib"),
        "label_encoder_path": os.path.join(MODELS_DIR, "lightgbm_label_encoder.joblib"),
        "results_json": os.path.join(RESULTS_DIR, "lightgbm_results.json"),
        "description": "Gradient boosting on hand-crafted biomechanical features",
    },
    "Random Forest": {
        "type": "baseline",
        "classifier_path": os.path.join(MODELS_DIR, "random_forest_classifier.joblib"),
        "scaler_path": None,
        "features_path": os.path.join(MODELS_DIR, "random_forest_features.joblib"),
        "label_encoder_path": os.path.join(MODELS_DIR, "random_forest_label_encoder.joblib"),
        "results_json": os.path.join(RESULTS_DIR, "random_forest_results.json"),
        "description": "Ensemble of decision trees on biomechanical features",
    },
    "MLP": {
        "type": "baseline",
        "classifier_path": os.path.join(MODELS_DIR, "mlp_classifier.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "mlp_scaler.joblib"),
        "features_path": None,
        "label_encoder_path": os.path.join(MODELS_DIR, "mlp_label_encoder.joblib"),
        "results_json": os.path.join(RESULTS_DIR, "mlp_results.json"),
        "description": "Multi-layer perceptron on biomechanical features",
    },
    "SVM": {
        "type": "baseline",
        "classifier_path": os.path.join(MODELS_DIR, "svm_classifier.joblib"),
        "scaler_path": os.path.join(MODELS_DIR, "svm_scaler.joblib"),
        "features_path": None,
        "label_encoder_path": os.path.join(MODELS_DIR, "svm_label_encoder.joblib"),
        "results_json": os.path.join(RESULTS_DIR, "svm_results.json"),
        "description": "Support Vector Machine with RBF kernel",
    },
    "STGCN++ Pretrained": {
        "type": "gcn",
        "classifier_path": os.path.join(MODELS_DIR, "stgcn_pretrained_finetuned.pt"),
        "architecture": "stgcnpp",
        # FIX #4: Results JSON is in models/, not results/
        "results_json": os.path.join(MODELS_DIR, "stgcn_pretrained_results.json"),
        "description": "Spatio-Temporal GCN++ pretrained on NTU120, fine-tuned",
    },
    "MS-G3D Pretrained": {
        "type": "gcn",
        "classifier_path": os.path.join(MODELS_DIR, "msg3d_pretrained_finetuned.pt"),
        "architecture": "msg3d",
        # FIX #4: Results JSON is in models/, not results/
        "results_json": os.path.join(MODELS_DIR, "msg3d_pretrained_results.json"),
        "description": "Multi-Scale G3D pretrained on NTU120, fine-tuned",
    },
}


COCO_JOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),
    (6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #16213e 100%);
        border: 1px solid #2a2a4a; border-radius: 12px;
        padding: 20px; text-align: center; transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem; font-weight: 700; color: #00d4aa;
    }
    .metric-card .label { font-size: 0.85rem; color: #8888aa; margin-top: 4px; }

    .badge-safe {
        background: #00d4aa; color: #0f0f0f;
        padding: 4px 14px; border-radius: 12px; font-size: 1rem; font-weight: 700;
    }
    .badge-risky {
        background: #ff4444; color: white;
        padding: 4px 14px; border-radius: 12px; font-size: 1rem; font-weight: 700;
    }
    .badge-concern {
        background: #ff4444; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    .badge-warning {
        background: #ffaa00; color: #1a1a1a;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    .badge-info {
        background: #4488ff; color: white;
        padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }

    .rep-card {
        background: #1a1a2e; border: 1px solid #2a2a4a;
        border-radius: 12px; padding: 16px; margin-bottom: 12px;
    }
    .rep-card-safe { border-left: 4px solid #00d4aa; }
    .rep-card-risky { border-left: 4px solid #ff4444; }

    .main-header {
        font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #00d4aa, #4488ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #8888aa; font-size: 1rem; margin-top: -8px; margin-bottom: 24px; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)



def angle_3pts(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return np.nan
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))

def trunk_lean(shoulder, hip):
    if shoulder is None or hip is None:
        return np.nan
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return np.nan
    return abs(float(np.degrees(np.arctan2(dx, -dy))))

def get_pt(kps, scores, idx, thr=0.3):
    if idx >= len(kps) or scores[idx] < thr:
        return None
    return (float(kps[idx][0]), float(kps[idx][1]))



@st.cache_resource
def load_pose_model():
    try:
        from rtmlib import Body
        model = Body(mode='balanced', to_openpose=False)
        return model, 'rtmlib'
    except ImportError:
        pass
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8m-pose.pt')
        return model, 'yolov8'
    except ImportError:
        pass
    return None, None


def extract_pose_from_frame(frame, model, backend):
    if backend == 'rtmlib':
        kps, scores = model(frame)
        if kps is None or len(kps) == 0:
            return None, None
        kps = kps[0] if len(kps.shape) == 3 else kps
        scores = scores[0] if len(scores.shape) == 2 else scores
        if len(kps) < 17:
            return None, None
        return kps[:17], scores[:17]
    elif backend == 'yolov8':
        results = model(frame, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return None, None
        k = results[0].keypoints
        if k.xy is None or len(k.xy) == 0:
            return None, None
        kps = k.xy[0].cpu().numpy()
        scores = k.conf[0].cpu().numpy() if k.conf is not None else np.ones(17)
        return kps, scores
    return None, None


def draw_skeleton_on_frame(frame, keypoints, scores, threshold=0.3):
    vis = frame.copy()
    for i, j in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            if scores[i] >= threshold and scores[j] >= threshold:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
    for i in range(min(len(keypoints), 17)):
        if scores[i] >= threshold:
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            color = (0, 0, 255) if i >= 11 else (255, 0, 0)
            cv2.circle(vis, (x, y), 5, color, -1)
    return vis


def process_video_for_ui(video_path, model, backend, progress_bar=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fname = os.path.basename(video_path)

    frame_data, raw_data, skeleton_frames = [], [], []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kps, scores = extract_pose_from_frame(frame, model, backend)

        if kps is not None:
            left_conf = sum(scores[i] for i in [5,11,13,15] if i < len(scores))
            right_conf = sum(scores[i] for i in [6,12,14,16] if i < len(scores))
            use_left = left_conf >= right_conf
            if use_left:
                sh,hi,kn,an = get_pt(kps,scores,5), get_pt(kps,scores,11), get_pt(kps,scores,13), get_pt(kps,scores,15)
            else:
                sh,hi,kn,an = get_pt(kps,scores,6), get_pt(kps,scores,12), get_pt(kps,scores,14), get_pt(kps,scores,16)

            feat = {
                'file': fname, 'frame': idx,
                'hip_y': hi[1]/h if hi else np.nan,
                'knee_deg': angle_3pts(hi, kn, an),
                'hip_deg': angle_3pts(sh, hi, kn),
                'trunk_deg': trunk_lean(sh, hi),
                'ankle_deg': angle_3pts(kn, an, (an[0], an[1] + 0.05 * h)) if kn and an else np.nan,
                'valid': 1,
            }
            frame_data.append(feat)

            raw_entry = {'file': fname, 'frame': idx}
            for j, name in enumerate(COCO_JOINTS):
                if j < len(kps):
                    raw_entry[f'{name}_x'] = kps[j][0]
                    raw_entry[f'{name}_y'] = kps[j][1]
                    raw_entry[f'{name}_conf'] = scores[j]
            raw_data.append(raw_entry)
            skeleton_frames.append(draw_skeleton_on_frame(frame, kps, scores))
        else:
            frame_data.append({'file': fname, 'frame': idx, 'hip_y': np.nan, 'knee_deg': np.nan, 'valid': 0})
            skeleton_frames.append(frame.copy())

        idx += 1
        if progress_bar:
            progress_bar.progress(min(idx / max(total, 1), 1.0), text=f"Extracting keypoints: {idx}/{total}")

    cap.release()
    return pd.DataFrame(frame_data), pd.DataFrame(raw_data), (skeleton_frames, fps, w, h)



@st.cache_resource
def load_segmenter():
    try:
        import tensorflow as tf
    except (ImportError, ModuleNotFoundError):
        st.error(
            "❌ **TensorFlow is not installed.** The LSTM rep segmenter requires it.\n\n"
            "**Fix (run in Admin PowerShell):**\n"
            "```\n"
            'reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
            "/v LongPathsEnabled /t REG_DWORD /d 1 /f\n"
            "```\n"
            "Then restart your terminal and run:\n"
            "```\n"
            "pip install tensorflow --no-cache-dir\n"
            "```"
        )
        return None

   
    for ext in ['.keras', '.h5']:
        path = os.path.join(MODELS_DIR, f"rep_segmenter_lstm{ext}")
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path)
            except Exception as e:
                st.warning(f"Could not load {os.path.basename(path)}: {e}")
                continue

    st.warning("Segmenter model not found. Expected `models/rep_segmenter_lstm.keras` or `.h5`.")
    return None


def segment_reps(df_kp, segmenter_model):
    if segmenter_model is None:
        return []
    df = df_kp.sort_values('frame').reset_index(drop=True)
    if len(df) < 10:
        return []

    hip = df['hip_y'].astype(float).ffill().bfill()
    knee = df['knee_deg'].astype(float).ffill().bfill()
    if hip.isna().all():
        hip = pd.Series(np.zeros(len(hip)))
    if knee.isna().all():
        knee = pd.Series(np.zeros(len(knee)))

    hip_norm = (hip - hip.mean()) / (hip.std() + 1e-6)
    knee_norm = (knee - knee.mean()) / (knee.std() + 1e-6)
    X = np.stack([hip_norm.values, knee_norm.values], axis=-1)[np.newaxis]

    probs = segmenter_model.predict(X, verbose=0)[0, :, 0]
    probs = pd.Series(probs).rolling(7, center=True, min_periods=1).mean().values

    mask = probs >= 0.5
    segments = []
    in_rep, start = False, 0
    for i, v in enumerate(mask):
        if v and not in_rep:
            in_rep, start = True, i
        elif not v and in_rep:
            in_rep = False
            segments.append([start, i - 1])
    if in_rep:
        segments.append([start, len(mask) - 1])

    merged = [segments[0]] if segments else []
    for s, e in segments[1:]:
        if s - merged[-1][1] <= 10:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    if merged:
        lengths = [e - s + 1 for s, e in merged]
        med = np.median(lengths)
        min_l, max_l = max(3, int(0.8 * med)), int(1.6 * med)
        merged = [(s, e) for s, e in merged if min_l <= e - s + 1 <= max_l]

    frames = df['frame'].values.astype(int)
    return [(int(frames[s]), int(frames[e])) for s, e in merged if s < len(frames) and e < len(frames)]



@st.cache_resource
def load_classifier(model_name):
    reg = MODEL_REGISTRY.get(model_name)
    if not reg:
        return None, f"Model '{model_name}' not in registry"

    path = reg['classifier_path']
    if not os.path.exists(path):
        return None, f"Model file not found: {os.path.basename(path)}"

    mtype = reg['type']

    if mtype == 'baseline':
        import joblib
        clf = joblib.load(path)
        scaler = joblib.load(reg['scaler_path']) if reg.get('scaler_path') and os.path.exists(reg['scaler_path']) else None
        features = joblib.load(reg['features_path']) if reg.get('features_path') and os.path.exists(reg['features_path']) else None
        le = joblib.load(reg['label_encoder_path']) if reg.get('label_encoder_path') and os.path.exists(reg['label_encoder_path']) else None
        return {'clf': clf, 'scaler': scaler, 'features': features, 'le': le, 'type': 'baseline'}, None

    elif mtype == 'gcn':
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        arch = reg.get('architecture', '')

        model = None
        if arch == 'stgcnpp':
            model = _build_stgcnpp()
        elif arch == 'msg3d':
            model = _build_msg3d()

        if model is not None:
            sd = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            model.load_state_dict(sd, strict=False)
            model = model.to(device)
            model.eval()

        return {'model': model, 'device': device, 'type': 'gcn', 'arch': arch}, None

    return None, f"Unknown model type: {mtype}"


def _build_stgcnpp():
   
    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from train_classifier_st_gcn import PYSKL_STGCNPP
        return PYSKL_STGCNPP(in_ch=3, nj=17, nc=2)
    except ImportError as e:
        st.warning(f"STGCN++ import failed: {e}")
        return None
    except Exception as e:
        st.warning(f"STGCN++ build failed: {e}")
        return None


def _build_msg3d():
    
    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        # Filename has a DASH: train_classifier_msg3d-pretrained.py
        # Python can't import dashed names directly, so use importlib
        import importlib.util
        for fname in ['train_classifier_msg3d-pretrained.py', 'train_classifier_msg3d_pretrained.py']:
            fpath = os.path.join(PROJECT_ROOT, fname)
            if os.path.exists(fpath):
                spec = importlib.util.spec_from_file_location("msg3d_module", fpath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, 'PYSKL_MSG3D'):
                    # MS-G3D uses: in_ch, num_joints, num_person, num_classes
                    return mod.PYSKL_MSG3D(in_ch=3, num_joints=17, num_person=2, num_classes=2)
        st.warning("MS-G3D: Could not find train_classifier_msg3d-pretrained.py")
        return None
    except Exception as e:
        st.warning(f"MS-G3D build failed: {e}")
        return None


def extract_rep_features(df_kp, start_f, end_f):
    rep = df_kp[(df_kp['frame'] >= start_f) & (df_kp['frame'] <= end_f)]
    if len(rep) < 5:
        return None

    feats = {}
    for col in ['hip_y', 'knee_deg', 'hip_deg', 'trunk_deg', 'ankle_deg']:
        if col not in rep.columns:
            continue
        vals = rep[col].astype(float).dropna()
        if len(vals) == 0:
            for s in ['_mean','_std','_min','_max','_range']:
                feats[f'{col}{s}'] = 0
        else:
            feats[f'{col}_mean'] = vals.mean()
            feats[f'{col}_std'] = vals.std()
            feats[f'{col}_min'] = vals.min()
            feats[f'{col}_max'] = vals.max()
            feats[f'{col}_range'] = vals.max() - vals.min()

    feats['duration_frames'] = len(rep)
    return feats


def extract_gcn_sequence(df_raw, start_f, end_f, seq_len=64):
    rep = df_raw[(df_raw['frame'] >= start_f) & (df_raw['frame'] <= end_f)].sort_values('frame')
    if len(rep) < 10:
        return None

    frames_data = []
    for _, fr in rep.iterrows():
        joints = []
        for jn in COCO_JOINTS:
            x = fr.get(f'{jn}_x', 0); y = fr.get(f'{jn}_y', 0)
            x = 0 if pd.isna(x) else float(x)
            y = 0 if pd.isna(y) else float(y)
            c = fr.get(f'{jn}_conf', fr.get(f'{jn}_score', 1.0))
            c = 1.0 if pd.isna(c) else float(c)
            joints.append([x, y, c])
        frames_data.append(joints)
    frames_data = np.array(frames_data, dtype=np.float32)

  
    hip_c = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
    frames_data[:, :, :2] -= hip_c[:, np.newaxis, :]
    sc = (frames_data[:, 5, :2] + frames_data[:, 6, :2]) / 2
    hc = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
    torso = np.linalg.norm(sc - hc, axis=1).mean()
    if torso > 1e-6:
        frames_data[:, :, :2] /= torso

    if len(frames_data) != seq_len:
        old_idx = np.linspace(0, 1, len(frames_data))
        new_idx = np.linspace(0, 1, seq_len)
        resampled = np.zeros((seq_len, 17, 3), dtype=np.float32)
        for j in range(17):
            for c in range(3):
                resampled[:, j, c] = np.interp(new_idx, old_idx, frames_data[:, j, c])
        frames_data = resampled

    return frames_data


def classify_rep_baseline(model_data, features_dict):
    clf = model_data['clf']
    scaler = model_data['scaler']
    feature_names = model_data['features']
    le = model_data['le']

    if feature_names is not None:
        X = np.array([[features_dict.get(f, 0) for f in feature_names]])
    else:
        X = np.array([list(features_dict.values())])

    X = np.nan_to_num(X, nan=0.0)
    if scaler is not None:
        X = scaler.transform(X)

    pred = clf.predict(X)[0]
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X)[0]
        conf = max(proba)
    elif hasattr(clf, 'decision_function'):
        dec = clf.decision_function(X)[0]
        conf = 1.0 / (1.0 + np.exp(-abs(dec)))
    else:
        conf = 0.5

    if le is not None:
        label = le.inverse_transform([pred])[0]
    else:
        label = "risky" if pred == 1 else "safe"

    return label, float(conf)


def classify_rep_gcn(model_data, sequence):
    import torch
    model = model_data.get('model')
    device = model_data['device']

    if model is None:
        return "unknown", 0.0

    with torch.no_grad():
        
        x = torch.FloatTensor(sequence).unsqueeze(0).to(device) 
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        risky_prob = prob[0, 1].item()
        label = "risky" if risky_prob >= 0.5 else "safe"
        conf = risky_prob if label == "risky" else 1.0 - risky_prob

    return label, conf


@st.cache_resource
def load_feedback_generator():
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from generate_feedback import FeedbackGenerator
        if os.path.exists(THRESHOLDS_JSON):
            return FeedbackGenerator.load_calibrated(THRESHOLDS_JSON)
        return FeedbackGenerator()
    except ImportError:
        return None


def load_model_results(model_name):
    kfold_path = os.path.join(MODELS_DIR, "kfold_results.json")
    if os.path.exists(kfold_path):
        try:
            with open(kfold_path) as f:
                kfold_data = json.load(f)
            models = kfold_data.get('models', {})
            if model_name in models:
                agg = models[model_name].get('aggregated', {})
                if agg:
                 
                    flat = {'_source': 'kfold', '_n_folds': models[model_name].get('n_folds', 5)}
                    for k, v in agg.items():
                        if isinstance(v, dict) and 'mean' in v:
                            flat[k] = v['mean']
                            flat[f'{k}_std'] = v['std']
                            flat[f'{k}_ci_lower'] = v.get('ci_lower')
                            flat[f'{k}_ci_upper'] = v.get('ci_upper')
                        else:
                            flat[k] = v
                    return flat
        except Exception:
            pass


    reg = MODEL_REGISTRY.get(model_name, {})
    rpath = reg.get('results_json', '')

    if rpath and os.path.exists(rpath):
        with open(rpath) as f:
            data = json.load(f)
        if any(k in data for k in ['f1_risky', 'f1_safe', 'balanced_accuracy', 'roc_auc']):
            return data
        for key in ['test_results', 'metrics', 'results', 'evaluation']:
            if key in data and isinstance(data[key], dict):
                return data[key]
        return data


    if os.path.exists(COMPARISON_JSON):
        with open(COMPARISON_JSON) as f:
            comp = json.load(f)
        metrics = comp.get('metrics', {})
        if model_name in metrics:
            return metrics[model_name]

    return {}


def load_comparison_data():
   
    if os.path.exists(COMPARISON_JSON):
        with open(COMPARISON_JSON) as f:
            return json.load(f)
    return None


def render_metric_card(label, value, fmt=".1f"):
    if isinstance(value, float) and np.isnan(value):
        display = "N/A"
    elif isinstance(value, float):
        display = f"{value:{fmt}}"
    elif isinstance(value, int):
        display = str(value)
    else:
        display = str(value)
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{display}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_rep_feedback(rep_idx, prediction, confidence, feedback_report):
    badge_class = "badge-safe" if prediction == "safe" else "badge-risky"
    card_class = "rep-card-safe" if prediction == "safe" else "rep-card-risky"
 
    st.markdown(f"""
    <div class="rep-card {card_class}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:1.2rem; font-weight:700;">Rep {rep_idx + 1}</span>
            <span class="{badge_class}">{prediction.upper()}</span>
        </div>
        <div style="color:#8888aa; font-size:0.9rem;">Confidence: {confidence:.0%}</div>
    </div>
    """, unsafe_allow_html=True)
 
    if feedback_report:
        if hasattr(feedback_report, 'issues') and feedback_report.issues:
            hard_issues = [i for i in feedback_report.issues
                          if not getattr(i, 'is_borderline', False)]
            borderline_issues = [i for i in feedback_report.issues
                                 if getattr(i, 'is_borderline', False)]
 
            
            for issue in hard_issues[:3]:
                sev = issue.severity if hasattr(issue, 'severity') else 'info'
                msg = issue.message if hasattr(issue, 'message') else str(issue)
                cue = issue.coaching_cue if hasattr(issue, 'coaching_cue') else ''
                st.markdown(f"""
                <div style="margin: 4px 0 4px 12px;">
                    <span class="badge-{sev}">{sev.upper()}</span>
                    <span style="margin-left:8px; color:#ccc;">{msg}</span>
                </div>
                """, unsafe_allow_html=True)
                if cue:
                    st.markdown(f"<div style='margin: 2px 0 8px 24px; color:#00d4aa; font-size:0.85rem;'>💡 {cue}</div>", unsafe_allow_html=True)
 
           
            if borderline_issues:
                header_text = "Form notes (rep classified as safe):" if prediction == "safe" else "Possible contributing factors:"
                st.markdown(f"<div style='margin: 8px 0 4px 12px; color:#8888aa; font-size:0.8rem;'>{header_text}</div>", unsafe_allow_html=True)
                for issue in borderline_issues[:2]:
                    msg = issue.message if hasattr(issue, 'message') else str(issue)
                    cue = issue.coaching_cue if hasattr(issue, 'coaching_cue') else ''
                    st.markdown(f"""
                    <div style="margin: 2px 0 2px 12px;">
                        <span class="badge-info">BORDERLINE</span>
                        <span style="margin-left:8px; color:#aaaacc; font-size:0.9rem;">{msg}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if cue:
                        st.markdown(f"<div style='margin: 2px 0 6px 24px; color:#00d4aa; font-size:0.85rem;'>💡 {cue}</div>", unsafe_allow_html=True)
 
        
        temporal_note = getattr(feedback_report, 'temporal_pattern_note', '')
        if temporal_note:
            st.markdown(f"""
            <div style="margin: 8px 0 4px 12px; padding: 10px; background: #1a1a2e;
                        border-left: 3px solid #4488ff; border-radius: 4px;">
                <div style="color:#4488ff; font-size:0.8rem; font-weight:600; margin-bottom:4px;">
                    📊 TEMPORAL PATTERN DETECTED
                </div>
                <div style="color:#aaaacc; font-size:0.85rem;">{temporal_note}</div>
            </div>
            """, unsafe_allow_html=True)
 
        if hasattr(feedback_report, 'positive_notes') and feedback_report.positive_notes:
            for note in feedback_report.positive_notes[:2]:
                st.markdown(f"<div style='margin:2px 0 2px 12px; color:#00d4aa;'>✓ {note}</div>", unsafe_allow_html=True)


_ffmpeg_checked = {'found': None, 'path': None, 'error': None}

def _find_ffmpeg():
    """Find ffmpeg once, cache result."""
    import shutil
    if _ffmpeg_checked['found'] is not None:
        return _ffmpeg_checked['path']

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        for candidate in [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
            os.path.expanduser(r'~\ffmpeg\bin\ffmpeg.exe'),
            os.path.expanduser(r'~\scoop\shims\ffmpeg.exe'),
        ]:
            if os.path.isfile(candidate):
                ffmpeg = candidate
                break
    _ffmpeg_checked['found'] = ffmpeg is not None
    _ffmpeg_checked['path'] = ffmpeg
    return ffmpeg


def _reencode_to_h264(video_path):
    """Re-encode a video to H.264 MP4 for browser compatibility."""
    import subprocess

    out_path = video_path + '.h264.mp4'
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        return out_path

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _ffmpeg_checked['error'] = 'ffmpeg not found on PATH'
        return None

    try:
        cmd = [
            ffmpeg, '-y', '-i', video_path,
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-crf', '23', '-an', out_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)

        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            return out_path

   
        stderr_text = result.stderr.decode('utf-8', errors='replace')[-500:] if result.stderr else ''
        _ffmpeg_checked['error'] = f'ffmpeg returned {result.returncode}: {stderr_text}'

        if os.path.exists(out_path):
            os.remove(out_path)
    except subprocess.TimeoutExpired:
        _ffmpeg_checked['error'] = 'ffmpeg timed out (>120s)'
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception as e:
        _ffmpeg_checked['error'] = str(e)
        if os.path.exists(out_path):
            os.remove(out_path)
    return None


def _is_browser_playable(video_path):
    
    ext = os.path.splitext(video_path)[1].lower()
    if ext in ('.webm',):
        return True

    if ext in ('.avi', '.mkv', '.mov'):
        return False
    if ext not in ('.mp4', '.m4v'):
        return False
    
    try:
        with open(video_path, 'rb') as f:
            header = f.read(min(32768, os.path.getsize(video_path)))
        
        if b'avc1' in header or b'avc3' in header or b'hev1' in header or b'hvc1' in header:
            return True
        
        if b'mp4v' in header or b'XVID' in header or b'xvid' in header:
            return False
        
        if b'vp09' in header:
            return True
        
        return False
    except Exception:
        return False


def display_video(video_path):
    
    if video_path is None:
        st.info("Video preview unavailable.")
        return
    try:
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            st.info("Video file is empty or missing.")
            return

        actual_path = video_path
        if not _is_browser_playable(video_path):
            reencoded = _reencode_to_h264(video_path)
            if reencoded:
                actual_path = reencoded
            else:
                err = _ffmpeg_checked.get('error', 'unknown')
                ffpath = _ffmpeg_checked.get('path', 'not found')
                st.warning(
                    f"Cannot play "
                    f"ffmpeg: `{ffpath}` | Error: {err}"
                )
                return

        video_bytes = Path(actual_path).read_bytes()
        if len(video_bytes) > 0:
            st.video(video_bytes)
        else:
            st.info("Re-encoded video is empty.")
    except Exception as e:
        st.warning(f"Could not preview: {os.path.basename(video_path)} — {e}")


def write_video_to_temp(frames, fps, w, h):
    
    if not frames:
        return None

   
    try:
        import imageio.v3 as iio
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.close()
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        iio.imwrite(tmp.name, rgb_frames, fps=fps,
                     codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
        if os.path.getsize(tmp.name) > 0:
            return tmp.name
    except Exception:
        pass

   
    try:
        import imageio
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.close()
        writer = imageio.get_writer(tmp.name, fps=fps, codec='libx264',
                                    output_params=['-pix_fmt', 'yuv420p'])
        for f in frames:
            writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        writer.close()
        if os.path.getsize(tmp.name) > 0:
            return tmp.name
    except Exception:
        pass

   
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
        tmp.close()
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            if os.path.getsize(tmp.name) > 0:
                return tmp.name
    except Exception:
        pass

    return None



def page_analyse():
    st.markdown('<div class="main-header">🏋️ Analyse Squat Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a video or select from training data → automatic analysis</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Model Selection")
        available_models = [name for name, reg in MODEL_REGISTRY.items()
                           if os.path.exists(reg['classifier_path'])]
        if not available_models:
            st.warning("No models found. Run training scripts first.")
            available_models = list(MODEL_REGISTRY.keys())

        selected_model = st.selectbox("Classification Model", available_models)
        reg = MODEL_REGISTRY.get(selected_model, {})
        st.caption(reg.get('description', ''))
        st.markdown(f"**Type:** `{reg.get('type', '?')}`")

       
        results = load_model_results(selected_model)
        if results:
            st.markdown("---")
            is_kfold = results.get('_source') == 'kfold'
            n_folds = results.get('_n_folds', 5)
            if is_kfold:
                st.markdown(f"### Performance ({n_folds}-Fold CV)")
            else:
                st.markdown("### Model Performance")
            key_metrics = ['f1_risky', 'f1_safe', 'balanced_accuracy', 'roc_auc', 'mcc']
            for k in key_metrics:
                v = results.get(k)
                if v is not None and isinstance(v, (int, float)):
                    std = results.get(f'{k}_std')
                    if is_kfold and std is not None:
                        st.metric(k.replace('_', ' ').title(), f"{v:.3f} ± {std:.3f}")
                    else:
                        st.metric(k.replace('_', ' ').title(), f"{v:.3f}")

    tab_upload, tab_training = st.tabs(["Upload Video", "Training Data"])

    with tab_upload:
        uploaded = st.file_uploader("Upload a squat video", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Original Video")
              
                display_video(video_path)

            if st.button("🔬 Analyse Video", type="primary", use_container_width=True):
                _run_pipeline(video_path, selected_model, col2)

    with tab_training:
        video_dirs = []
        for d in [OUTPUT_POSE_VIDEOS_DIR, os.path.join(PROJECT_ROOT, "output_videos"),
                   os.path.join(PROJECT_ROOT, "segmented_reps"),
                   os.path.join(PROJECT_ROOT, "all_videos")]:
            if os.path.isdir(d):
                video_dirs.append(d)

        if not video_dirs:
            st.info("No training video directories found.")
        else:
            chosen_dir = st.selectbox("Video Folder", video_dirs,
                                       format_func=lambda x: os.path.basename(x),
                                       key="td_folder")
            vids = sorted([f for f in os.listdir(chosen_dir)
                          if f.lower().endswith(('.mp4', '.avi', '.mov'))
                          and '.h264.mp4' not in f.lower()])
            if vids:
                chosen_vid = st.selectbox(f"Select Video ({len(vids)} available)", vids,
                                          key="td_video")
                video_path = os.path.join(chosen_dir, chosen_vid)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### Selected Video")
                    display_video(video_path)
                if st.button("🔬 Analyse Selected Video", type="primary", use_container_width=True):
                    _run_pipeline(video_path, selected_model, col2)
            else:
                st.info("No videos found in this folder.")


def _run_pipeline(video_path, model_name, results_col):
    status = st.status("Running analysis pipeline...", expanded=True)

    with status:
        st.write("**Step 1/4:** Loading pose estimation model...")
        pose_model, backend = load_pose_model()
        if pose_model is None:
            st.error("No pose estimation backend available. Install rtmlib or ultralytics.")
            return
        st.write(f"Loaded ({backend})")

        st.write("**Step 2/4:** Extracting keypoints...")
        progress = st.progress(0, text="Extracting keypoints...")
        df_kp, df_raw, skel_info = process_video_for_ui(video_path, pose_model, backend, progress)
        if df_kp is None or len(df_kp) == 0:
            st.error("Failed to extract keypoints from video.")
            return
        skeleton_frames, fps, w, h = skel_info
        st.write(f"{len(df_kp)} frames processed")

       
        st.write("**Step 3/4:** Segmenting repetitions...")
        segmenter = load_segmenter()
        reps = segment_reps(df_kp, segmenter)
        if not reps:
            st.warning("No reps detected. Showing full video analysis instead.")
            reps = [(0, len(df_kp) - 1)]
        st.write(f"✅ {len(reps)} rep(s) found")

        st.write("**Step 4/4:** Classifying and generating feedback...")
        model_data, err = load_classifier(model_name)
        if err:
            st.warning(f"Could not load {model_name}: {err}")
            model_data = None

        feedback_gen = load_feedback_generator()

    with results_col:
        st.markdown("#### Analysis Results")
        st.markdown(f"**Model:** {model_name} | **Reps detected:** {len(reps)}")

        if skeleton_frames:
            tmp_path = write_video_to_temp(skeleton_frames, fps, w, h)
            with st.expander("🦴 Skeleton Overlay Video", expanded=False):
                display_video(tmp_path)

    st.markdown("---")
    st.markdown("### Per-Rep Analysis")

    for i, (sf, ef) in enumerate(reps):
        with st.container():
            c1, c2 = st.columns([1, 2])

            prediction, confidence = "unknown", 0.0
            if model_data:
                try:
                    mtype = model_data['type']
                    if mtype == 'baseline':
                        feats = extract_rep_features(df_kp, sf, ef)
                        if feats:
                            prediction, confidence = classify_rep_baseline(model_data, feats)
                    elif mtype == 'gcn':
                        if df_raw is not None and len(df_raw) > 0:
                            seq = extract_gcn_sequence(df_raw, sf, ef)
                            if seq is not None:
                                prediction, confidence = classify_rep_gcn(model_data, seq)
                except Exception as e:
                    st.warning(f"Classification error (Rep {i+1}): {e}")

            fb_report = None
            if feedback_gen and df_raw is not None and len(df_raw) > 0:
                rep_raw = df_raw[(df_raw['frame'] >= sf) & (df_raw['frame'] <= ef)].sort_values('frame')
                if len(rep_raw) >= 5:
                    kps_arr = []
                    for _, fr in rep_raw.iterrows():
                        joints = []
                        for jn in COCO_JOINTS:
                            x = fr.get(f'{jn}_x', 0); y = fr.get(f'{jn}_y', 0)
                            c = fr.get(f'{jn}_conf', 1.0)
                            joints.append([float(x or 0), float(y or 0), float(c or 1.0)])
                        kps_arr.append(joints)
                    kps_arr = np.array(kps_arr, dtype=np.float32)
                    try:
                      
                        _mtype = model_data.get('type', 'baseline') if model_data else 'baseline'
                        fb_report = feedback_gen.analyse_rep(
                            kps_arr, prediction=prediction,
                            confidence=confidence, model_name=model_name,
                            model_type=_mtype,
                        )
                    except Exception:
                        fb_report = None

            with c1:
                rep_frames = skeleton_frames[sf:ef+1] if sf < len(skeleton_frames) else []
                if rep_frames:
                    tmp_path = write_video_to_temp(rep_frames, fps, w, h)
                    display_video(tmp_path)

            with c2:
                render_rep_feedback(i, prediction, confidence, fb_report)

            st.markdown("---")



def page_training_data():
    st.markdown('<div class="main-header">📂 Training Data Browser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Browse your labelled squat dataset</div>', unsafe_allow_html=True)

    if not os.path.exists(LABELS_CSV):
        st.warning(f"Labels file not found: {LABELS_CSV}")
        return

    df_labels = pd.read_csv(LABELS_CSV)
    df_valid = df_labels[df_labels['label'].isin(['safe', 'risky'])].copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Reps", len(df_valid), "d")
    with col2:
        n_safe = len(df_valid[df_valid['label'] == 'safe'])
        render_metric_card("Safe Reps", n_safe, "d")
    with col3:
        n_risky = len(df_valid[df_valid['label'] == 'risky'])
        render_metric_card("Risky Reps", n_risky, "d")
    with col4:
        rejected = len(df_labels[~df_labels['label'].isin(['safe', 'risky'])])
        render_metric_card("Rejected", rejected, "d")

    st.markdown("<br>", unsafe_allow_html=True)

    label_filter = st.selectbox("Filter by label", ["All", "safe", "risky"])
    if label_filter != "All":
        df_show = df_valid[df_valid['label'] == label_filter]
    else:
        df_show = df_valid

    display_cols = [c for c in ['file', 'rep_id', 'label'] if c in df_show.columns]
    st.dataframe(df_show[display_cols].reset_index(drop=True), width="stretch", height=400)


    vid_dirs = [OUTPUT_POSE_VIDEOS_DIR, os.path.join(PROJECT_ROOT, "output_videos"),
                os.path.join(PROJECT_ROOT, "segmented_reps")]
    for vd in vid_dirs:
        if os.path.isdir(vd):
            st.markdown("### 🎬 Video Preview")
            rep_vids = sorted([f for f in os.listdir(vd)
                               if f.lower().endswith(('.mp4', '.avi'))
                               and '.h264.mp4' not in f.lower()])
            if rep_vids:
                chosen = st.selectbox("Select rep video", rep_vids)
            
                display_video(os.path.join(vd, chosen))
            break


def page_model_comparison():
    st.markdown('<div class="main-header">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Side-by-side performance across all trained models</div>', unsafe_allow_html=True)

    
    kfold_path = os.path.join(MODELS_DIR, "kfold_results.json")
    has_kfold = os.path.exists(kfold_path)

    if has_kfold:
        tab_kfold, tab_single = st.tabs(["K-Fold Cross-Validation", "Single Split Results"])
        with tab_kfold:
            _render_kfold_results(kfold_path)
        with tab_single:
            comparison = load_comparison_data()
            if comparison and 'metrics' in comparison:
                _render_comparison_from_json(comparison)
            else:
                _render_comparison_from_individual()
    else:
        comparison = load_comparison_data()
        if comparison and 'metrics' in comparison:
            _render_comparison_from_json(comparison)
        else:
            _render_comparison_from_individual()
        st.markdown("---")
        st.info("Run `python run_kfold_evaluation.py` to generate robust K-Fold CV results with mean ± std across folds.")


def _render_kfold_results(kfold_path):
    
    with open(kfold_path) as f:
        data = json.load(f)

    n_folds = data.get('n_folds', 5)
    method = data.get('evaluation_method', f'{n_folds}-Fold CV')
    total_time = data.get('total_time_seconds', 0)

    st.markdown(f"### {method}")
    st.caption(f"Seed: {data.get('random_seed', 42)} | "
               f"Total time: {total_time / 60:.1f} min | "
               f"Generated: {data.get('timestamp', 'N/A')[:19]}")

    models = data.get('models', {})
    if not models:
        st.warning("No model results found in kfold_results.json")
        return

   
    st.markdown("### Performance Summary (mean ± std)")
    key_metrics = ['f1_risky', 'f1_safe', 'balanced_accuracy', 'mcc',
                   'roc_auc', 'precision_risky', 'recall_risky', 'specificity',
                   'cohens_kappa', 'g_mean', 'pr_auc', 'brier_score', 'log_loss']

    rows = []
    for mn, mdata in models.items():
        agg = mdata.get('aggregated', {})
        row = {'Model': mn}
        for k in key_metrics:
            if k in agg:
                m = agg[k]['mean']
                s = agg[k]['std']
                row[k] = f"{m:.3f} ± {s:.3f}"
            else:
                row[k] = "—"
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    if 'Model' in df_summary.columns:
        df_summary = df_summary.set_index('Model')

    st.dataframe(df_summary, width="stretch")

    st.markdown("### Metric Comparison Chart")
    chart_metrics = ['f1_risky', 'f1_safe', 'balanced_accuracy', 'mcc', 'roc_auc']
    available_chart = [m for m in chart_metrics if any(m in models[mn].get('aggregated', {}) for mn in models)]

    if available_chart:
        selected_metric = st.selectbox("Select metric to compare", available_chart,
                                        format_func=lambda x: x.replace('_', ' ').title(),
                                        key="kfold_chart_metric")

        chart_rows = []
        for mn, mdata in models.items():
            agg = mdata.get('aggregated', {})
            if selected_metric in agg:
                chart_rows.append({
                    'Model': mn,
                    'Mean': agg[selected_metric]['mean'],
                    'Std': agg[selected_metric]['std'],
                    'CI Lower': agg[selected_metric].get('ci_lower', 0),
                    'CI Upper': agg[selected_metric].get('ci_upper', 0),
                })

        if chart_rows:
            df_chart = pd.DataFrame(chart_rows).set_index('Model').sort_values('Mean', ascending=False)
            st.bar_chart(df_chart[['Mean']])

            st.markdown(f"**95% Confidence Intervals for {selected_metric.replace('_', ' ').title()}:**")
            ci_rows = []
            for r in chart_rows:
                ci_rows.append({
                    'Model': r['Model'],
                    'Mean': f"{r['Mean']:.4f}",
                    'Std': f"{r['Std']:.4f}",
                    '95% CI': f"[{r['CI Lower']:.4f}, {r['CI Upper']:.4f}]",
                })
            st.dataframe(pd.DataFrame(ci_rows).set_index('Model'), width="stretch")

    st.markdown("### Best Model Per Metric (by mean)")
    best_metrics = ['f1_risky', 'balanced_accuracy', 'mcc', 'roc_auc', 'f1_macro', 'g_mean']
    best_available = [m for m in best_metrics
                      if any(m in models[mn].get('aggregated', {}) for mn in models)]

    if best_available:
        cols = st.columns(min(len(best_available), 4))
        for i, metric in enumerate(best_available):
            with cols[i % len(cols)]:
                best_model, best_val = None, -1
                for mn, mdata in models.items():
                    agg = mdata.get('aggregated', {})
                    if metric in agg and agg[metric]['mean'] > best_val:
                        best_val = agg[metric]['mean']
                        best_model = mn
                if best_model:
                    st.metric(
                        metric.replace('_', ' ').title(),
                        f"{best_val:.3f}",
                        delta=best_model,
                    )

    st.markdown("### Per-Fold Breakdown")
    model_names = list(models.keys())
    selected_model = st.selectbox("Select model", model_names, key="kfold_fold_model")

    mdata = models.get(selected_model, {})
    per_fold = mdata.get('per_fold', [])
    if per_fold:
        fold_rows = []
        for i, fm in enumerate(per_fold):
            row = {'Fold': i + 1}
            for k in ['f1_risky', 'f1_safe', 'balanced_accuracy', 'mcc', 'roc_auc',
                       'precision_risky', 'recall_risky']:
                row[k.replace('_', ' ').title()] = round(fm.get(k, 0), 4)
            fold_rows.append(row)

        agg = mdata.get('aggregated', {})
        mean_row = {'Fold': 'Mean ± Std'}
        for k in ['f1_risky', 'f1_safe', 'balanced_accuracy', 'mcc', 'roc_auc',
                   'precision_risky', 'recall_risky']:
            if k in agg:
                mean_row[k.replace('_', ' ').title()] = f"{agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}"
        fold_rows.append(mean_row)

        df_folds = pd.DataFrame(fold_rows).set_index('Fold')
        st.dataframe(df_folds, width="stretch")

        if 'confusion_matrix_total' in agg:
            cm = np.array(agg['confusion_matrix_total'])
            st.markdown(f"**Aggregated Confusion Matrix (all folds combined):**")
            cm_df = pd.DataFrame(cm,
                                  index=['Actual Safe', 'Actual Risky'],
                                  columns=['Predicted Safe', 'Predicted Risky'])
            st.dataframe(cm_df, width="stretch")


def _render_comparison_from_json(comparison):
    
    metrics = comparison.get('metrics', {})

    st.markdown("### Performance Metrics")

    if metrics:
        rows = []
        for model_name, model_metrics in metrics.items():
            row = {'Model': model_name}
            row.update(model_metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        if 'Model' in df.columns:
            df = df.set_index('Model')

        display_order = [
            'f1_risky', 'f1_safe', 'f1_macro', 'balanced_accuracy', 'accuracy',
            'precision_risky', 'recall_risky', 'specificity',
            'mcc', 'cohens_kappa', 'g_mean',
            'roc_auc', 'pr_auc', 'brier_score', 'log_loss',
        ]
        available_cols = [c for c in display_order if c in df.columns]
        df_display = df[available_cols]

        numeric_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()

        lower_better = {'brier_score', 'log_loss'}
        higher_better = [c for c in numeric_cols if c not in lower_better]
        lower_cols = [c for c in numeric_cols if c in lower_better]

        styled = df_display.style.format({c: "{:.4f}" for c in numeric_cols})
        if higher_better:
            styled = styled.highlight_max(subset=higher_better, color='#1a3a2a')
        if lower_cols:
            styled = styled.highlight_min(subset=lower_cols, color='#1a3a2a')

        st.dataframe(styled, width="stretch")

        if len(available_cols) > 0:
            default_metric = 'f1_macro' if 'f1_macro' in available_cols else available_cols[0]
            default_idx = available_cols.index(default_metric)
            selected = st.selectbox("Compare metric", available_cols, index=default_idx)

            chart_data = df_display[[selected]].sort_values(selected, ascending=False)
            st.bar_chart(chart_data)

        st.markdown("### Best Model Per Metric")
        best_metrics = ['f1_risky', 'f1_safe', 'f1_macro', 'balanced_accuracy', 'mcc', 'roc_auc']
        best_available = [m for m in best_metrics if m in df_display.columns]

        if best_available:
            cols = st.columns(min(len(best_available), 4))
            for i, metric in enumerate(best_available):
                with cols[i % len(cols)]:
                    best_model = df_display[metric].idxmax()
                    best_val = df_display[metric].max()
                    st.metric(
                        metric.replace('_', ' ').title(),
                        f"{best_val:.4f}",
                        delta=best_model,
                    )

    mcnemar = comparison.get('mcnemar_results', [])
    if mcnemar:
        st.markdown("### Pairwise McNemar Tests")

        mc_rows = []
        for test in mcnemar:
            sig = test.get('significant', False)
            if isinstance(sig, str):
                sig = sig.lower() == 'true'

            mc_rows.append({
                'Model A': test.get('model_a', ''),
                'Model B': test.get('model_b', ''),
                'χ²': f"{test.get('chi2', 0):.3f}",
                'p-value': f"{test.get('p_value', 1):.6f}",
                'Significant': '✅ Yes' if sig else '❌ No',
                'b (A wrong, B right)': test.get('b', ''),
                'c (A right, B wrong)': test.get('c', ''),
            })

        mc_df = pd.DataFrame(mc_rows)
        st.dataframe(mc_df, width="stretch", hide_index=True)

        n_sig = sum(1 for r in mcnemar if
                    (isinstance(r.get('significant'), bool) and r['significant']) or
                    (isinstance(r.get('significant'), str) and r['significant'].lower() == 'true'))
        n_total = len(mcnemar)
        st.info(f"**{n_sig}/{n_total}** pairwise comparisons are statistically significant (p < 0.05)")

    st.markdown("---")
    st.markdown("### Model Details")
    for name, reg in MODEL_REGISTRY.items():
        exists = os.path.exists(reg['classifier_path'])
        status = "✅" if exists else "❌"
        with st.expander(f"{status} {name} — {reg['description']}"):
            st.markdown(f"**Type:** {reg['type']}")
            st.markdown(f"**Path:** `{os.path.basename(reg['classifier_path'])}`")
            st.markdown(f"**File exists:** {exists}")
            if name in metrics:
                model_metrics = metrics[name]
                for k, v in model_metrics.items():
                    if isinstance(v, float):
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v:.4f}")
                    elif isinstance(v, (int, bool)):
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")


def _render_comparison_from_individual():
    rows = []
    for name, reg in MODEL_REGISTRY.items():
        results = load_model_results(name)
        if results:
            row = {'Model': name, 'Type': reg['type']}
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    row[k.replace('_', ' ').title()] = v
            rows.append(row)

    if not rows:
        st.info("No model results found. Run training scripts and compare_models.py first.")
        st.markdown("**Expected files:**")
        st.code(
            "results/lightgbm_results.json\n"
            "results/mlp_results.json\n"
            "results/random_forest_results.json\n"
            "results/svm_results.json\n"
            "results/model_comparison.json"
        )
        return

    df_results = pd.DataFrame(rows)
    st.dataframe(df_results.set_index('Model'), width="stretch")

    metric_cols = [c for c in df_results.columns if c not in ('Model', 'Type')]
    if metric_cols:
        selected = st.selectbox("Compare metric", metric_cols)
        chart_data = df_results[['Model', selected]].set_index('Model').sort_values(selected, ascending=False)
        st.bar_chart(chart_data)

    st.markdown("---")
    st.markdown("### Model Details")
    for name, reg in MODEL_REGISTRY.items():
        exists = os.path.exists(reg['classifier_path'])
        status = "✅" if exists else "❌"
        with st.expander(f"{status} {name} — {reg['description']}"):
            st.markdown(f"**Type:** {reg['type']}")
            st.markdown(f"**Path:** `{reg['classifier_path']}`")
            st.markdown(f"**File exists:** {exists}")
            results = load_model_results(name)
            if results:
                st.json(results)

def main():
    inject_css()

    with st.sidebar:
        st.markdown("## 🏋️ Squat Analyser")
        st.markdown("---")
        page = st.radio("Navigate", ["🔬 Analyse", "📂 Training Data", "📊 Model Comparison"],
                        label_visibility="collapsed")

    if "Analyse" in page:
        page_analyse()
    elif "Training Data" in page:
        page_training_data()
    elif "Model Comparison" in page:
        page_model_comparison()


if __name__ == "__main__":
    main()