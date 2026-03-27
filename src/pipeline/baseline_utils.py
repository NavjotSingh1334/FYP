#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, matthews_corrcoef,
    cohen_kappa_score, average_precision_score,
    brier_score_loss, log_loss, accuracy_score
)


RANDOM_SEED = 42
TEST_SIZE = 0.2

FRAME_LEVEL_DIR = os.path.join("frame_level")
REP_LEVEL_DIR = os.path.join("rep_level")
MODELS_DIR = os.path.join("models")
RESULTS_DIR = os.path.join("results")

KEYPOINTS_CSV = os.path.join(FRAME_LEVEL_DIR, "pose_keypoints.csv")
LABELS_CSV = os.path.join(REP_LEVEL_DIR, "rep_labels_merged.csv")
SEGMENTS_CSV = os.path.join(REP_LEVEL_DIR, "auto_segments.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_filename(f):
    
    f = os.path.basename(str(f))
    if f.startswith("pose_"):
        f = f[5:]
    return os.path.splitext(f)[0].lower().strip()


def load_data():
    
    print("\n📊 Loading data...")

    df_kp = pd.read_csv(KEYPOINTS_CSV)
    df_kp['file_norm'] = df_kp['file'].apply(normalize_filename)
    print(f"   Keypoints: {len(df_kp):,} rows, {df_kp['file_norm'].nunique()} videos")

    df_lb = pd.read_csv(LABELS_CSV)
    df_lb = df_lb[df_lb['label'].isin(['safe', 'risky'])].copy()
    df_lb['file_norm'] = df_lb['file'].apply(normalize_filename)
    n_safe = (df_lb['label'] == 'safe').sum()
    n_risky = (df_lb['label'] == 'risky').sum()
    print(f"   Labels: {len(df_lb)} (safe={n_safe}, risky={n_risky}, ratio={n_safe/max(n_risky,1):.1f}:1)")

    df_sg = pd.read_csv(SEGMENTS_CSV)
    if 'source_file' in df_sg.columns:
        df_sg['file_norm'] = df_sg['source_file'].apply(normalize_filename)
    elif 'file' in df_sg.columns:
        df_sg['file_norm'] = df_sg['file'].apply(normalize_filename)
    print(f"   Segments: {len(df_sg)}")

    return df_kp, df_lb, df_sg



def extract_rep_features(df_kp_video, start_frame, end_frame):
    
    rep = df_kp_video[(df_kp_video['frame'] >= start_frame) &
                       (df_kp_video['frame'] <= end_frame)]
    if len(rep) < 5:
        return None

    features = {}
    for col in ['hip_y', 'knee_deg', 'hip_deg', 'trunk_deg', 'ankle_deg']:
        if col not in rep.columns:
            continue
        vals = rep[col].astype(float).dropna()
        if len(vals) == 0:
            for stat in ['mean', 'std', 'min', 'max', 'range']:
                features[f'{col}_{stat}'] = 0.0
        else:
            features[f'{col}_mean'] = float(vals.mean())
            features[f'{col}_std'] = float(vals.std())
            features[f'{col}_min'] = float(vals.min())
            features[f'{col}_max'] = float(vals.max())
            features[f'{col}_range'] = float(vals.max() - vals.min())

    features['duration_frames'] = len(rep)
    return features


def build_feature_matrix(df_kp, df_lb, df_sg):
    
    print("\n🔧 Building feature matrix...")
    kp_grouped = {n: g for n, g in df_kp.groupby('file_norm')}

    X_rows, y_rows, metadata_rows = [], [], []
    skipped = 0

    for _, row in df_lb.iterrows():
        fn = row['file_norm']
        rid = row.get('rep_id', 0)
        label = row['label']

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
        feats = extract_rep_features(kp_grouped[fn], sf, ef)

        if feats is None:
            skipped += 1
            continue

        X_rows.append(feats)
        y_rows.append(label)
        metadata_rows.append({'file': fn, 'rep_id': rid, 'start': sf, 'end': ef})

    feature_names = list(X_rows[0].keys()) if X_rows else []
    X = np.array([[row.get(f, 0) for f in feature_names] for row in X_rows], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array(y_rows)

    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X)} (skipped {skipped})")
    print(f"   Class balance: safe={np.sum(y=='safe')}, risky={np.sum(y=='risky')}")

    return X, y, feature_names, metadata_rows



def _specificity(y_true, y_pred):
   
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _g_mean(y_true, y_pred):

    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = _specificity(y_true, y_pred)
    return math.sqrt(sens * spec)


def _bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_boot=2000, ci=0.95, seed=42):
   
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], y_prob[idx] if y_prob is not None else None)
            if np.isfinite(s):
                scores.append(s)
        except:
            pass

    if len(scores) < 100:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return (lo, hi)


def print_evaluation(y_test, y_pred, y_prob, class_names, model_name):
    
    print("\n" + "=" * 70)
    print(f"  {model_name} — EVALUATION RESULTS")
    print("=" * 70)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    {class_names[0]:<10s} {class_names[1]:<10s}")
    print(f"  Actual {class_names[0]:<10s} {tn:<10d} {fp:<10d}")
    print(f"  Actual {class_names[1]:<10s} {fn:<10d} {tp:<10d}")

    # Classification report
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    for line in report.split('\n'):
        if line.strip():
            print(f"   {line}")

    # === Core metrics ===
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_risky = f1_score(y_test, y_pred, pos_label=1)
    f1_safe = f1_score(y_test, y_pred, pos_label=0)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    prec_risky = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_risky = recall_score(y_test, y_pred, pos_label=1)
    spec = _specificity(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    g_mean = _g_mean(y_test, y_pred)

    print(f"\n  --- Threshold-dependent Metrics ---")
    print(f"  {'Accuracy:':<28s} {acc:.4f}")
    print(f"  {'Balanced Accuracy:':<28s} {bal_acc:.4f}")
    print(f"  {'F1 (risky):':<28s} {f1_risky:.4f}")
    print(f"  {'F1 (safe):':<28s} {f1_safe:.4f}")
    print(f"  {'F1 (macro):':<28s} {f1_macro:.4f}")
    print(f"  {'Precision (risky):':<28s} {prec_risky:.4f}")
    print(f"  {'Recall / Sensitivity:':<28s} {rec_risky:.4f}")
    print(f"  {'Specificity:':<28s} {spec:.4f}")
    print(f"  {'MCC:':<28s} {mcc:.4f}")
    print(f"  {'Cohens Kappa:':<28s} {kappa:.4f}")
    print(f"  {'G-Mean:':<28s} {g_mean:.4f}")

    roc_auc, pr_auc, brier, logloss = np.nan, np.nan, np.nan, np.nan
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)

        print(f"\n  --- Probability-based Metrics ---")
        print(f"  {'ROC-AUC:':<28s} {roc_auc:.4f}")
        print(f"  {'PR-AUC (Avg Precision):':<28s} {pr_auc:.4f}")
        print(f"  {'Brier Score:':<28s} {brier:.4f}  (lower is better)")
        print(f"  {'Log Loss:':<28s} {logloss:.4f}  (lower is better)")

    print(f"\n  --- 95% Bootstrap Confidence Intervals (n=2000) ---")

    y_t, y_p = np.array(y_test), np.array(y_pred)
    y_pr = np.array(y_prob) if y_prob is not None else None

    ci_f1r = _bootstrap_ci(y_t, y_p, y_pr, lambda yt, yp, ypr: f1_score(yt, yp, pos_label=1, zero_division=0))
    ci_mcc = _bootstrap_ci(y_t, y_p, y_pr, lambda yt, yp, ypr: matthews_corrcoef(yt, yp))
    ci_bacc = _bootstrap_ci(y_t, y_p, y_pr, lambda yt, yp, ypr: balanced_accuracy_score(yt, yp))
    ci_auc = _bootstrap_ci(y_t, y_p, y_pr, lambda yt, yp, ypr: roc_auc_score(yt, ypr) if ypr is not None else np.nan) if y_prob is not None else (np.nan, np.nan)
    ci_prauc = _bootstrap_ci(y_t, y_p, y_pr, lambda yt, yp, ypr: average_precision_score(yt, ypr) if ypr is not None else np.nan) if y_prob is not None else (np.nan, np.nan)

    print(f"  {'F1 (risky):':<28s} {f1_risky:.4f}  [{ci_f1r[0]:.4f}, {ci_f1r[1]:.4f}]")
    print(f"  {'MCC:':<28s} {mcc:.4f}  [{ci_mcc[0]:.4f}, {ci_mcc[1]:.4f}]")
    print(f"  {'Balanced Accuracy:':<28s} {bal_acc:.4f}  [{ci_bacc[0]:.4f}, {ci_bacc[1]:.4f}]")
    if y_prob is not None:
        print(f"  {'ROC-AUC:':<28s} {roc_auc:.4f}  [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
        print(f"  {'PR-AUC:':<28s} {pr_auc:.4f}  [{ci_prauc[0]:.4f}, {ci_prauc[1]:.4f}]")

    metrics = {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'f1_risky': float(f1_risky),
        'f1_safe': float(f1_safe),
        'f1_macro': float(f1_macro),
        'precision_risky': float(prec_risky),
        'recall_risky': float(rec_risky),
        'specificity': float(spec),
        'mcc': float(mcc),
        'cohens_kappa': float(kappa),
        'g_mean': float(g_mean),
        'roc_auc': float(roc_auc) if np.isfinite(roc_auc) else None,
        'pr_auc': float(pr_auc) if np.isfinite(pr_auc) else None,
        'brier_score': float(brier) if np.isfinite(brier) else None,
        'log_loss': float(logloss) if np.isfinite(logloss) else None,
        'confusion_matrix': cm.tolist(),
        'ci_f1_risky_95': list(ci_f1r),
        'ci_mcc_95': list(ci_mcc),
        'ci_balanced_accuracy_95': list(ci_bacc),
        'ci_roc_auc_95': list(ci_auc) if y_prob is not None else None,
        'ci_pr_auc_95': list(ci_prauc) if y_prob is not None else None,
    }

    return metrics



def mcnemar_test(y_true, y_pred_a, y_pred_b, model_a_name="A", model_b_name="B"):

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right

    if (b + c) == 0:
        return 0.0, 1.0, {'b': int(b), 'c': int(c)}

    if (b + c) < 25:
        
        from scipy.stats import binomtest
        p_val = binomtest(b, b + c, 0.5).pvalue
        chi2 = float((b - c) ** 2 / (b + c))
    else:
        chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
        from scipy.stats import chi2 as chi2_dist
        p_val = float(1 - chi2_dist.cdf(chi2, df=1))

    print(f"\n  McNemar's test: {model_a_name} vs {model_b_name}")
    print(f"    {model_a_name} right, {model_b_name} wrong: {b}")
    print(f"    {model_a_name} wrong, {model_b_name} right: {c}")
    print(f"    Chi2 = {chi2:.4f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print(f"    → Statistically significant difference (p < 0.05)")
    else:
        print(f"    → No significant difference (p >= 0.05)")

    return chi2, p_val, {'b': int(b), 'c': int(c)}


def cochrans_q_test(y_true, predictions_dict):
   
    from scipy.stats import chi2 as chi2_dist

    model_names = list(predictions_dict.keys())
    M = len(model_names)
    if M < 3:
        print("  Cochran's Q requires 3+ models. Use McNemar's test instead.")
        return None, None

    n = len(y_true)
    
    correct_matrix = np.zeros((n, M), dtype=int)
    for j, name in enumerate(model_names):
        correct_matrix[:, j] = (predictions_dict[name] == y_true).astype(int)

  
    R = correct_matrix.sum(axis=1)
    C = correct_matrix.sum(axis=0)
    T = correct_matrix.sum()

 
    numerator = (M - 1) * (M * np.sum(C ** 2) - T ** 2)
    denominator = M * T - np.sum(R ** 2)

    if denominator == 0:
        return 0.0, 1.0

    Q = float(numerator / denominator)
    p_val = float(1 - chi2_dist.cdf(Q, df=M - 1))

    print(f"\n  Cochran's Q Test ({M} models)")
    print(f"    Q = {Q:.4f}, df = {M-1}, p = {p_val:.6f}")
    if p_val < 0.05:
        print(f"    → Significant difference among models (p < 0.05)")
        print(f"    → Proceed with pairwise McNemar tests")
    else:
        print(f"    → No significant difference among models (p >= 0.05)")

    return Q, p_val


def save_results_json(metrics, model_name, extra=None):

    result = {
        'model': model_name,
        **metrics,
    }
    if extra:
        result.update(extra)

    path = os.path.join(RESULTS_DIR, f"{model_name.lower().replace(' ', '_')}_results.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  💾 Results JSON: {path}")

    
    path2 = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_results.json")
    with open(path2, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return path