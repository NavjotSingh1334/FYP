#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import math
import json
import time
import importlib.util
import warnings
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

N_FOLDS = 5
RANDOM_SEED = 42


FRAME_LEVEL_DIR = "frame_level"
REP_LEVEL_DIR = "rep_level"
MODELS_DIR = "models"
RESULTS_DIR = "results"

KEYPOINTS_CSV = os.path.join(FRAME_LEVEL_DIR, "pose_keypoints.csv")
KEYPOINTS_RAW_CSV = os.path.join(FRAME_LEVEL_DIR, "pose_keypoints_raw.csv")
LABELS_CSV = os.path.join(REP_LEVEL_DIR, "rep_labels_merged.csv")
SEGMENTS_CSV = os.path.join(REP_LEVEL_DIR, "auto_segments.csv")

OUTPUT_PATH = os.path.join(MODELS_DIR, "kfold_results.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

GCN_CONFIG = {
    'sequence_length': 64,
    'num_joints': 17,
    'in_channels': 3,
    'augmentation': {
        'gaussian_noise_std': 0.01,
        'scale_range': (0.9, 1.1),
        'temporal_shift_range': (-3, 4),
        'horizontal_flip_prob': 0.3,
        'noise_prob': 0.5,
        'scale_prob': 0.5,
        'shift_prob': 0.3,
    },
    'use_smote': True,
    'smote_sampling_strategy': 0.75,
    'smote_k_neighbors': 5,
    'phase1_epochs': 30,
    'phase1_lr': 0.001,
    'phase1_patience': 12,
    'phase2_epochs': 25,
    'phase2_lr': 0.0001,
    'phase2_patience': 10,
    'batch_size': 32,
    'phase2_batch_size': 8,
    'weight_decay': 5e-4,
    'label_smoothing': 0.1,
    'gradient_clip_norm': 1.0,
    'random_seed': RANDOM_SEED,
}

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
]


def compute_confusion_matrix(y_true, y_pred, nc=2):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    cm = np.zeros((nc, nc), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision(yt, yp, pos=1):
    yt, yp = np.asarray(yt), np.asarray(yp)
    tp = np.sum((yp == pos) & (yt == pos))
    pp = np.sum(yp == pos)
    return tp / pp if pp > 0 else 0.0

def recall(yt, yp, pos=1):
    yt, yp = np.asarray(yt), np.asarray(yp)
    tp = np.sum((yp == pos) & (yt == pos))
    ap = np.sum(yt == pos)
    return tp / ap if ap > 0 else 0.0

def f1_score(yt, yp, pos=1):
    p, r = precision(yt, yp, pos), recall(yt, yp, pos)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def balanced_accuracy(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    recs = []
    for c in np.unique(yt):
        m = (yt == c)
        if m.sum() > 0:
            recs.append(np.sum(yp[m] == c) / m.sum())
    return np.mean(recs)

def accuracy(yt, yp):
    return np.mean(np.asarray(yt) == np.asarray(yp))

def specificity(yt, yp):
    cm = compute_confusion_matrix(yt, yp)
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def mcc(yt, yp):
    cm = compute_confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den > 0 else 0.0

def cohens_kappa(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    n = len(yt)
    acc = np.mean(yt == yp)
    pt = np.bincount(yt, minlength=2) / n
    pp = np.bincount(yp, minlength=2) / n
    pe = np.sum(pt * pp)
    return (acc - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

def g_mean(yt, yp):
    s = recall(yt, yp, pos=1)
    sp = specificity(yt, yp)
    return math.sqrt(s * sp)

def roc_auc(yt, yscores, nt=1000):
    yt, yscores = np.asarray(yt), np.asarray(yscores)
    tp_ = np.sum(yt == 1)
    tn_ = np.sum(yt == 0)
    if tp_ == 0 or tn_ == 0:
        return 0.5
    thresholds = np.linspace(yscores.min() - 0.001, yscores.max() + 0.001, nt)
    tpr_l, fpr_l = [], []
    for th in thresholds:
        preds = (yscores >= th).astype(int)
        tpr_l.append(np.sum((preds == 1) & (yt == 1)) / tp_)
        fpr_l.append(np.sum((preds == 1) & (yt == 0)) / tn_)
    idx = np.argsort(fpr_l)
    fpr_s, tpr_s = np.array(fpr_l)[idx], np.array(tpr_l)[idx]
    return float(np.trapz(tpr_s, fpr_s))

def pr_auc(yt, yscores, nt=1000):
    yt, yscores = np.asarray(yt), np.asarray(yscores)
    thresholds = np.linspace(yscores.max() + 0.001, yscores.min() - 0.001, nt)
    precs, recs = [], []
    for th in thresholds:
        preds = (yscores >= th).astype(int)
        tp = np.sum((preds == 1) & (yt == 1))
        fp = np.sum((preds == 1) & (yt == 0))
        fn = np.sum((preds == 0) & (yt == 1))
        precs.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    idx = np.argsort(recs)
    rs, ps = np.array(recs)[idx], np.array(precs)[idx]
    ap = 0.0
    for i in range(1, len(rs)):
        ap += (rs[i] - rs[i - 1]) * ps[i]
    return ap

def brier_score(yt, yprob):
    return float(np.mean((np.asarray(yprob) - np.asarray(yt)) ** 2))

def log_loss(yt, yprob, eps=1e-15):
    yt = np.asarray(yt, dtype=float)
    yp = np.clip(np.asarray(yprob), eps, 1 - eps)
    return -float(np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute all metrics for one fold."""
    m = {
        'accuracy': accuracy(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy(y_true, y_pred),
        'f1_risky': f1_score(y_true, y_pred, pos=1),
        'f1_safe': f1_score(y_true, y_pred, pos=0),
        'f1_macro': (f1_score(y_true, y_pred, 1) + f1_score(y_true, y_pred, 0)) / 2,
        'precision_risky': precision(y_true, y_pred, pos=1),
        'recall_risky': recall(y_true, y_pred, pos=1),
        'specificity': specificity(y_true, y_pred),
        'mcc': mcc(y_true, y_pred),
        'cohens_kappa': cohens_kappa(y_true, y_pred),
        'g_mean': g_mean(y_true, y_pred),
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        m['roc_auc'] = roc_auc(y_true, y_prob)
        m['pr_auc'] = pr_auc(y_true, y_prob)
        m['brier_score'] = brier_score(y_true, y_prob)
        m['log_loss'] = log_loss(y_true, y_prob)
    return m



def stratified_k_fold(y, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n = len(y)

    class_indices = {}
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        class_indices[cls] = idx

    fold_assignment = np.zeros(n, dtype=int)
    for cls, idx in class_indices.items():
        for i, sample_idx in enumerate(idx):
            fold_assignment[sample_idx] = i % n_folds

    folds = []
    for fold in range(n_folds):
        test_idx = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        folds.append((train_idx, test_idx))

    return folds


def apply_smote(X, y, target_ratio=0.75, k=5, seed=42):
    rng = np.random.RandomState(seed)
    safe_n = int(np.sum(y == 0))
    risky_n = int(np.sum(y == 1))
    target = int(safe_n * target_ratio)
    n_syn = max(0, target - risky_n)
    if n_syn == 0:
        return X, y

    if X.ndim > 2:
        orig_shape = X.shape[1:]
        flat = X.reshape(len(X), -1)
    else:
        orig_shape = None
        flat = X

    min_idx = np.where(y == 1)[0]
    min_flat = flat[min_idx]
    nm = len(min_idx)
    k_actual = min(k, nm - 1)
    k_actual = max(1, k_actual)


    nbrs = []
    for i in range(nm):
        d = np.sqrt(np.sum((min_flat - min_flat[i:i + 1]) ** 2, axis=1))
        d[i] = np.inf
        nbrs.append(np.argpartition(d, k_actual)[:k_actual])

    synthetic = []
    for _ in range(n_syn):
        src = rng.randint(0, nm)
        nbr = rng.choice(nbrs[src])
        lam = rng.uniform(0, 1)
        synthetic.append(min_flat[src] + lam * (min_flat[nbr] - min_flat[src]))

    synthetic = np.array(synthetic)
    if orig_shape is not None:
        synthetic = synthetic.reshape(-1, *orig_shape)
    syn_labels = np.ones(n_syn, dtype=y.dtype)

    X_aug = np.concatenate([X, synthetic], axis=0)
    y_aug = np.concatenate([y, syn_labels], axis=0)
    shuffle = rng.permutation(len(X_aug))
    return X_aug[shuffle], y_aug[shuffle]


def normalize_filename(f):
    f = os.path.basename(str(f))
    if f.startswith("pose_"):
        f = f[5:]
    return os.path.splitext(f)[0].lower().strip()


def load_all_data():

    print("  LOADING DATA")
    

    df_kp = pd.read_csv(KEYPOINTS_CSV)
    df_kp['file_norm'] = df_kp['file'].apply(normalize_filename)
    print(f"  Keypoints (processed): {len(df_kp):,} rows")

    df_raw = None
    if os.path.exists(KEYPOINTS_RAW_CSV):
        df_raw = pd.read_csv(KEYPOINTS_RAW_CSV)
        df_raw['file_norm'] = df_raw['file'].apply(normalize_filename)
        print(f"  Keypoints (raw): {len(df_raw):,} rows")

    df_lb = pd.read_csv(LABELS_CSV)
    df_lb = df_lb[df_lb['label'].isin(['safe', 'risky'])].copy()
    df_lb['file_norm'] = df_lb['file'].apply(normalize_filename)
    print(f"  Labels: {len(df_lb)} (safe={sum(df_lb['label'] == 'safe')}, risky={sum(df_lb['label'] == 'risky')})")

    df_sg = pd.read_csv(SEGMENTS_CSV)
    if 'source_file' in df_sg.columns:
        df_sg['file_norm'] = df_sg['source_file'].apply(normalize_filename)
    elif 'file' in df_sg.columns:
        df_sg['file_norm'] = df_sg['file'].apply(normalize_filename)
    print(f"  Segments: {len(df_sg)}")

    return df_kp, df_raw, df_lb, df_sg


def extract_baseline_features(df_kp, df_lb, df_sg):
    print("\n  Extracting baseline features...")
    kp_grouped = {n: g for n, g in df_kp.groupby('file_norm')}
    X_rows, y_list, meta = [], [], []
    skipped = 0

    for _, row in df_lb.iterrows():
        fn, rid, label = row['file_norm'], row.get('rep_id', 0), row['label']
        seg = df_sg[df_sg['file_norm'] == fn]
        if 'rep_id' in df_sg.columns:
            s2 = seg[seg['rep_id'] == rid]
            if len(s2) > 0:
                seg = s2
        if len(seg) == 0 or fn not in kp_grouped:
            skipped += 1
            continue

        sf = int(seg.iloc[0].get('start_frame', 0))
        ef = int(seg.iloc[0].get('end_frame', sf + 64))
        rep = kp_grouped[fn]
        rep = rep[(rep['frame'] >= sf) & (rep['frame'] <= ef)]
        if len(rep) < 5:
            skipped += 1
            continue

        feats = {}
        for col in ['hip_y', 'knee_deg', 'hip_deg', 'trunk_deg', 'ankle_deg']:
            if col not in rep.columns:
                for s in ['mean', 'std', 'min', 'max', 'range']:
                    feats[f'{col}_{s}'] = 0.0
                continue
            vals = rep[col].astype(float).dropna()
            if len(vals) == 0:
                for s in ['mean', 'std', 'min', 'max', 'range']:
                    feats[f'{col}_{s}'] = 0.0
            else:
                feats[f'{col}_mean'] = float(vals.mean())
                feats[f'{col}_std'] = float(vals.std())
                feats[f'{col}_min'] = float(vals.min())
                feats[f'{col}_max'] = float(vals.max())
                feats[f'{col}_range'] = float(vals.max() - vals.min())
        feats['duration_frames'] = len(rep)

        X_rows.append(feats)
        y_list.append(0 if label == 'safe' else 1)
        meta.append({'file': fn, 'rep_id': rid})

    feature_names = list(X_rows[0].keys()) if X_rows else []
    X = np.array([[row.get(f, 0) for f in feature_names] for row in X_rows], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array(y_list)

    print(f"  Baseline features: {X.shape[0]} samples, {X.shape[1]} features (skipped {skipped})")
    return X, y, feature_names, meta



def extract_gcn_sequences(df_raw, df_lb, df_sg, seq_len=64):
    if df_raw is None:
        print("  WARNING: No raw keypoints CSV, skipping GCN sequence extraction")
        return None, None, None

    print("\n  Extracting GCN sequences...")
    kp_grouped = {n: g for n, g in df_raw.groupby('file_norm')}
    sequences, labels = [], []
    skipped = Counter()

    for _, row in tqdm(df_lb.iterrows(), total=len(df_lb), desc="  GCN extract"):
        fn, rid = row['file_norm'], row.get('rep_id', 0)
        label = 0 if row['label'] == 'safe' else 1
        seg = df_sg[(df_sg['file_norm'] == fn)]
        if 'rep_id' in df_sg.columns:
            s2 = seg[seg['rep_id'] == rid]
            if len(s2) > 0:
                seg = s2
        if len(seg) == 0:
            skipped['no_segment'] += 1
            continue
        if fn not in kp_grouped:
            skipped['no_keypoints'] += 1
            continue

        sf = int(seg.iloc[0].get('start_frame', 0))
        ef = int(seg.iloc[0].get('end_frame', sf + 64))
        vk = kp_grouped[fn]
        rep_kp = vk[(vk['frame'] >= sf) & (vk['frame'] <= ef)].sort_values('frame')
        if len(rep_kp) < 10:
            skipped['too_short'] += 1
            continue

        frames_data = []
        for _, fr in rep_kp.iterrows():
            joints = []
            for kn in KEYPOINT_NAMES:
                x = fr.get(f'{kn}_x', 0)
                y_ = fr.get(f'{kn}_y', 0)
                x = 0 if pd.isna(x) else float(x)
                y_ = 0 if pd.isna(y_) else float(y_)
                c = fr.get(f'{kn}_score', fr.get(f'{kn}_conf', 1.0))
                c = 1.0 if pd.isna(c) else float(c)
                joints.append([x, y_, c])
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

        sequences.append(frames_data)
        labels.append(label)

    if not sequences:
        return None, None, None

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f"  GCN sequences: {sequences.shape} (skipped {sum(skipped.values())})")
    return sequences, labels

def train_baseline_fold(model_name, X_train, y_train, X_test, y_test, feature_names):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if model_name == "LightGBM":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=RANDOM_SEED, verbose=-1,
        )
    elif model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            min_samples_leaf=3, class_weight='balanced',
            random_state=RANDOM_SEED, n_jobs=-1,
        )
    elif model_name == "SVM":
        from sklearn.svm import SVC
        clf = SVC(
            kernel='rbf', C=10, gamma='scale', class_weight='balanced',
            probability=True, random_state=RANDOM_SEED,
        )
    elif model_name == "MLP":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    y_prob = None
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_te)[:, 1]
    elif hasattr(clf, 'decision_function'):
        dec = clf.decision_function(X_te)
        y_prob = 1.0 / (1.0 + np.exp(-dec))

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(['safe', 'risky'])
    artifacts = {
        'clf': clf,
        'scaler': scaler,
        'feature_names': feature_names,
        'label_encoder': le,
    }

    return y_pred, y_prob, artifacts


def _import_gcn_module(name, filename):
    for fn in [filename, filename.replace('-', '_')]:
        fpath = os.path.join('.', fn)
        if os.path.exists(fpath):
            spec = importlib.util.spec_from_file_location(name, fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def train_gcn_fold(model_name, X_train, y_train, X_test, y_test, config):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_tr, y_tr = apply_smote(X_train, y_train,
                              target_ratio=config.get('smote_sampling_strategy', 0.75),
                              k=config.get('smote_k_neighbors', 5),
                              seed=config['random_seed'])

    class SeqDataset(Dataset):
        def __init__(self, seqs, labs, augment=False):
            self.seqs = seqs
            self.labs = labs
            self.augment = augment

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            seq = self.seqs[idx].copy()
            if self.augment:
                aug = config.get('augmentation', {})
                if np.random.rand() < aug.get('noise_prob', 0.5):
                    seq[:, :, :2] += np.random.randn(*seq[:, :, :2].shape).astype(np.float32) * aug.get('gaussian_noise_std', 0.01)
                if np.random.rand() < aug.get('scale_prob', 0.5):
                    lo, hi = aug.get('scale_range', (0.9, 1.1))
                    seq[:, :, :2] *= np.random.uniform(lo, hi)
                if np.random.rand() < aug.get('shift_prob', 0.3):
                    lo, hi = aug.get('temporal_shift_range', (-3, 4))
                    seq = np.roll(seq, np.random.randint(lo, hi), axis=0)
                if np.random.rand() < aug.get('horizontal_flip_prob', 0.3):
                    seq[:, :, 0] = -seq[:, :, 0]
                    for l, r in FLIP_PAIRS:
                        seq[:, l, :], seq[:, r, :] = seq[:, r, :].copy(), seq[:, l, :].copy()
            return torch.FloatTensor(seq), self.labs[idx]

    train_ds = SeqDataset(X_tr, y_tr, augment=True)
    test_ds = SeqDataset(X_test, y_test, augment=False)
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    if model_name == "STGCN++ Pretrained":
        mod = _import_gcn_module('stgcn', 'train_classifier_st_gcn.py')
        if mod is None:
            print("    WARNING: Could not import STGCN++ module")
            return None, None, None
        model = mod.PYSKL_STGCNPP(in_ch=3, nj=17, nc=2).to(device)
        ckpt_path = os.path.join(MODELS_DIR, 'stgcnpp_ntu120_xsub_hrnet_j.pth')
    elif model_name == "MS-G3D Pretrained":
        mod = _import_gcn_module('msg3d', 'train_classifier_msg3d-pretrained.py')
        if mod is None:
            print("    WARNING: Could not import MS-G3D module")
            return None, None, None
        model = mod.PYSKL_MSG3D(in_ch=3, num_joints=17, num_classes=2).to(device)
        ckpt_path = os.path.join(MODELS_DIR, 'msg3d_pyskl_ntu120_xsub_hrnet_j.pth')
    else:
        raise ValueError(f"Unknown GCN model: {model_name}")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        model_sd = model.state_dict()
        loaded = 0
        for ck, cv in checkpoint.items():
            if 'cls_head' in ck:
                continue
            mk = ck if ck in model_sd else ck.replace('backbone.', '', 1)
            if mk in model_sd and model_sd[mk].shape == cv.shape:
                model_sd[mk] = cv
                loaded += 1
        model.load_state_dict(model_sd)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    use_amp = config.get('use_mixed_precision', True) and device.type == 'cuda'

    for name, param in model.named_parameters():
        param.requires_grad = 'cls_head' in name

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['phase1_lr'], weight_decay=config['weight_decay']
    )

    best_f1, best_state, wait = 0.0, None, 0
    for ep in range(config['phase1_epochs']):
        model.train()
        for seqs, labs in train_dl:
            seqs, labs = seqs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            optimizer.step()

        model.eval()
        all_pred, all_lab = [], []
        with torch.no_grad():
            for seqs, labs in test_dl:
                out = model(seqs.to(device))
                all_pred.extend(out.argmax(1).cpu().numpy())
                all_lab.extend(labs.numpy())
        f1r = f1_score(all_lab, all_pred, pos=1)
        if f1r > best_f1:
            best_f1, wait = f1r, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= config['phase1_patience']:
            break

    if best_state:
        model.load_state_dict(best_state)

    for param in model.parameters():
        param.requires_grad = True

    p2_dl = DataLoader(train_ds, batch_size=config.get('phase2_batch_size', 8),
                       shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['phase2_lr'],
                                   weight_decay=config['weight_decay'])
    wait = 0
    for ep in range(config['phase2_epochs']):
        model.train()
        for seqs, labs in p2_dl:
            seqs, labs = seqs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            optimizer.step()

        model.eval()
        all_pred, all_lab = [], []
        with torch.no_grad():
            for seqs, labs in test_dl:
                out = model(seqs.to(device))
                all_pred.extend(out.argmax(1).cpu().numpy())
                all_lab.extend(labs.numpy())
        f1r = f1_score(all_lab, all_pred, pos=1)
        if f1r > best_f1:
            best_f1, wait = f1r, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= config['phase2_patience']:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_pred, all_prob, all_lab = [], [], []
    with torch.no_grad():
        for seqs, labs in test_dl:
            out = model(seqs.to(device))
            probs = F.softmax(out.float(), dim=1)
            all_pred.extend(out.argmax(1).cpu().numpy())
            all_prob.extend(probs[:, 1].cpu().numpy())
            all_lab.extend(labs.numpy())

    saved_state = {k: v.cpu().clone() for k, v in model.state_dict().items()} if best_state else None

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(all_pred), np.array(all_prob), saved_state


def aggregate_fold_metrics(fold_metrics_list):
    if not fold_metrics_list:
        return {}

    keys = [k for k in fold_metrics_list[0].keys() if k != 'confusion_matrix']

    agg = {}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics_list if k in fm and fm[k] is not None]
        if not vals:
            continue
        vals = np.array(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # 95% CI: mean ± t * (std / sqrt(n))
        n = len(vals)
        # t-value for 95% CI with n-1 df (approximation for small n)
        t_val = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
        margin = t_val * (std / math.sqrt(n)) if n > 1 else 0.0
        agg[k] = {
            'mean': round(mean, 4),
            'std': round(std, 4),
            'ci_lower': round(mean - margin, 4),
            'ci_upper': round(mean + margin, 4),
            'per_fold': [round(v, 4) for v in vals],
        }

    # Aggregate confusion matrices
    cms = [np.array(fm['confusion_matrix']) for fm in fold_metrics_list if 'confusion_matrix' in fm]
    if cms:
        total_cm = sum(cms)
        agg['confusion_matrix_total'] = total_cm.tolist()

    return agg


def main():
    total_start = time.time()
    print("=" * 60)
    print(f"  STRATIFIED {N_FOLDS}-FOLD CROSS-VALIDATION")
    print(f"  All Models — Robust Evaluation")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Seed: {RANDOM_SEED}")
    print()

    np.random.seed(RANDOM_SEED)

    df_kp, df_raw, df_lb, df_sg = load_all_data()

    X_base, y_base, feat_names, meta = extract_baseline_features(df_kp, df_lb, df_sg)

    X_gcn, y_gcn = extract_gcn_sequences(df_raw, df_lb, df_sg)

    if X_gcn is not None:
        print(f"\n  Baseline samples: {len(X_base)}, GCN samples: {len(X_gcn)}")
        if len(X_base) != len(X_gcn):
            print("  WARNING: Sample counts differ. Using minimum overlap.")
            n_min = min(len(X_base), len(X_gcn))
            print(f"  Using {n_min} samples for consistent folds")

    print(f"\n  Generating {N_FOLDS} stratified folds...")
    folds_base = stratified_k_fold(y_base, n_folds=N_FOLDS, seed=RANDOM_SEED)
    folds_gcn = stratified_k_fold(y_gcn, n_folds=N_FOLDS, seed=RANDOM_SEED) if X_gcn is not None else None

    for i, (train_idx, test_idx) in enumerate(folds_base):
        tr_safe = np.sum(y_base[train_idx] == 0)
        tr_risky = np.sum(y_base[train_idx] == 1)
        te_safe = np.sum(y_base[test_idx] == 0)
        te_risky = np.sum(y_base[test_idx] == 1)
        print(f"  Fold {i + 1}: train={len(train_idx)} (S={tr_safe} R={tr_risky}), "
              f"test={len(test_idx)} (S={te_safe} R={te_risky})")

    baseline_models = ["LightGBM", "Random Forest", "SVM", "MLP"]
    gcn_models = ["STGCN++ Pretrained", "MS-G3D Pretrained"]
    all_models = baseline_models + gcn_models

    all_results = {}

    BASELINE_SAVE_PATHS = {
        "LightGBM": {
            'clf': 'lightgbm_classifier.joblib',
            'scaler': None,
            'features': 'lightgbm_features.joblib',
            'le': 'lightgbm_label_encoder.joblib',
        },
        "Random Forest": {
            'clf': 'random_forest_classifier.joblib',
            'scaler': None,
            'features': 'random_forest_features.joblib',
            'le': 'random_forest_label_encoder.joblib',
        },
        "SVM": {
            'clf': 'svm_classifier.joblib',
            'scaler': 'svm_scaler.joblib',
            'features': None,
            'le': 'svm_label_encoder.joblib',
        },
        "MLP": {
            'clf': 'mlp_classifier.joblib',
            'scaler': 'mlp_scaler.joblib',
            'features': None,
            'le': 'mlp_label_encoder.joblib',
        },
    }

    for model_name in baseline_models:
        print(f"\n{'=' * 60}")
        print(f"  {model_name} — {N_FOLDS}-Fold CV")
        print(f"{'=' * 60}")

        fold_metrics = []
        best_fold_f1 = -1
        best_fold_artifacts = None

        for fold_i, (train_idx, test_idx) in enumerate(folds_base):
            t0 = time.time()
            X_tr, X_te = X_base[train_idx], X_base[test_idx]
            y_tr, y_te = y_base[train_idx], y_base[test_idx]

            X_tr_aug, y_tr_aug = apply_smote(X_tr, y_tr, seed=RANDOM_SEED + fold_i)

            y_pred, y_prob, artifacts = train_baseline_fold(model_name, X_tr_aug, y_tr_aug, X_te, y_te, feat_names)
            metrics = compute_all_metrics(y_te, y_pred, y_prob)
            fold_metrics.append(metrics)

            if metrics['f1_risky'] > best_fold_f1:
                best_fold_f1 = metrics['f1_risky']
                best_fold_artifacts = artifacts
                best_fold_idx = fold_i

            elapsed = time.time() - t0
            print(f"  Fold {fold_i + 1}/{N_FOLDS}: "
                  f"F1r={metrics['f1_risky']:.3f} BA={metrics['balanced_accuracy']:.3f} "
                  f"MCC={metrics['mcc']:.3f} AUC={metrics.get('roc_auc', 0):.3f} "
                  f"({elapsed:.1f}s)")

        if best_fold_artifacts is not None:
            import joblib
            paths = BASELINE_SAVE_PATHS[model_name]
            joblib.dump(best_fold_artifacts['clf'], os.path.join(MODELS_DIR, paths['clf']))
            if paths['scaler']:
                joblib.dump(best_fold_artifacts['scaler'], os.path.join(MODELS_DIR, paths['scaler']))
            if paths['features']:
                joblib.dump(best_fold_artifacts['feature_names'], os.path.join(MODELS_DIR, paths['features']))
            if paths['le']:
                joblib.dump(best_fold_artifacts['label_encoder'], os.path.join(MODELS_DIR, paths['le']))
            print(f"  💾 Saved best model (fold {best_fold_idx + 1}, F1r={best_fold_f1:.3f})")

        agg = aggregate_fold_metrics(fold_metrics)
        all_results[model_name] = {
            'per_fold': fold_metrics,
            'aggregated': agg,
            'n_folds': N_FOLDS,
            'type': 'baseline',
        }
        print(f"\n  {model_name} SUMMARY:")
        for k in ['f1_risky', 'balanced_accuracy', 'mcc', 'roc_auc']:
            if k in agg:
                a = agg[k]
                print(f"    {k}: {a['mean']:.4f} ± {a['std']:.4f}  "
                      f"[{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]")

    
    GCN_SAVE_PATHS = {
        "STGCN++ Pretrained": 'stgcn_pretrained_finetuned.pt',
        "MS-G3D Pretrained": 'msg3d_pretrained_finetuned.pt',
    }

    if X_gcn is not None and folds_gcn is not None:
        import torch
        for model_name in gcn_models:
            print(f"\n{'=' * 60}")
            print(f"  {model_name} — {N_FOLDS}-Fold CV")
            print(f"{'=' * 60}")

            fold_metrics = []
            best_fold_f1 = -1
            best_fold_state = None
            best_fold_idx = -1

            for fold_i, (train_idx, test_idx) in enumerate(folds_gcn):
                t0 = time.time()
                torch.manual_seed(RANDOM_SEED + fold_i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(RANDOM_SEED + fold_i)

                X_tr, X_te = X_gcn[train_idx], X_gcn[test_idx]
                y_tr, y_te = y_gcn[train_idx], y_gcn[test_idx]

                y_pred, y_prob, model_state = train_gcn_fold(model_name, X_tr, y_tr, X_te, y_te, GCN_CONFIG)

                if y_pred is None:
                    print(f"  Fold {fold_i + 1}: SKIPPED (model import failed)")
                    continue

                metrics = compute_all_metrics(y_te, y_pred, y_prob)
                fold_metrics.append(metrics)

                
                if metrics['f1_risky'] > best_fold_f1:
                    best_fold_f1 = metrics['f1_risky']
                    best_fold_state = model_state
                    best_fold_idx = fold_i

                elapsed = time.time() - t0
                print(f"  Fold {fold_i + 1}/{N_FOLDS}: "
                      f"F1r={metrics['f1_risky']:.3f} BA={metrics['balanced_accuracy']:.3f} "
                      f"MCC={metrics['mcc']:.3f} AUC={metrics.get('roc_auc', 0):.3f} "
                      f"({elapsed:.1f}s)")

            # Save best fold's model
            if best_fold_state is not None:
                save_path = os.path.join(MODELS_DIR, GCN_SAVE_PATHS[model_name])
                torch.save({
                    'model_state_dict': best_fold_state,
                    'config': GCN_CONFIG,
                    'best_fold': best_fold_idx + 1,
                    'best_fold_f1_risky': best_fold_f1,
                }, save_path)
                print(f"  💾 Saved best model (fold {best_fold_idx + 1}, F1r={best_fold_f1:.3f})")

            if fold_metrics:
                agg = aggregate_fold_metrics(fold_metrics)
                all_results[model_name] = {
                    'per_fold': fold_metrics,
                    'aggregated': agg,
                    'n_folds': len(fold_metrics),
                    'type': 'gcn',
                }
                print(f"\n  {model_name} SUMMARY:")
                for k in ['f1_risky', 'balanced_accuracy', 'mcc', 'roc_auc']:
                    if k in agg:
                        a = agg[k]
                        print(f"    {k}: {a['mean']:.4f} ± {a['std']:.4f}  "
                              f"[{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]")

    
    print(f"\n\n{'=' * 75}")
    print(f"  FINAL COMPARISON — {N_FOLDS}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 75}")
    print(f"\n  {'Model':<25s} {'F1(risky)':>12s} {'Bal.Acc':>12s} {'MCC':>12s} {'ROC-AUC':>12s}")
    print("  " + "─" * 73)

    for mn in all_models:
        if mn not in all_results:
            continue
        agg = all_results[mn]['aggregated']
        f1r = agg.get('f1_risky', {})
        ba = agg.get('balanced_accuracy', {})
        mc = agg.get('mcc', {})
        auc = agg.get('roc_auc', {})
        print(f"  {mn:<25s} "
              f"{f1r.get('mean', 0):.3f}±{f1r.get('std', 0):.3f}  "
              f"{ba.get('mean', 0):.3f}±{ba.get('std', 0):.3f}  "
              f"{mc.get('mean', 0):.3f}±{mc.get('std', 0):.3f}  "
              f"{auc.get('mean', 0):.3f}±{auc.get('std', 0):.3f}")

    
    output = {
        'evaluation_method': f'Stratified {N_FOLDS}-Fold Cross-Validation',
        'n_folds': N_FOLDS,
        'random_seed': RANDOM_SEED,
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': time.time() - total_start,
        'models': {},
    }
    for mn, res in all_results.items():
        output['models'][mn] = {
            'aggregated': res['aggregated'],
            'per_fold': res['per_fold'],
            'n_folds': res['n_folds'],
            'type': res['type'],
        }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {OUTPUT_PATH}")

    # Per-model summary files (for Streamlit app)
    for mn, res in all_results.items():
        slug = mn.lower().replace(' ', '_').replace('+', 'p')
        path = os.path.join(MODELS_DIR, f"{slug}_kfold_results.json")
        summary = {
            'model': mn,
            'evaluation_method': f'Stratified {N_FOLDS}-Fold CV',
            'n_folds': res['n_folds'],
            **{k: v for k, v in res['aggregated'].items()},
        }
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    # Also save to results/ dir
    results_path = os.path.join(RESULTS_DIR, "kfold_results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results also saved: {results_path}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  CROSS-VALIDATION COMPLETE")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()