#!/usr/bin/env python3


import os, sys, json, math
import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

COCO_JOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

CONFIG = {
    'keypoints_raw_path': os.path.join('frame_level', 'pose_keypoints_raw.csv'),
    'keypoints_path': os.path.join('frame_level', 'pose_keypoints.csv'),
    'labels_path': os.path.join('rep_level', 'rep_labels_merged.csv'),
    'segments_path': os.path.join('rep_level', 'auto_segments.csv'),
    'thresholds_save_path': os.path.join('models', 'feedback_thresholds.json'),
    'confidence_min': 0.3,
}

def angle_3pts(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return np.nan
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))

def trunk_lean_from_vertical(shoulder, hip):
    if shoulder is None or hip is None:
        return np.nan
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return np.nan
    return abs(float(np.degrees(np.arctan2(dx, -dy))))

def knee_valgus_angle(hip, knee, ankle):
    if hip is None or knee is None or ankle is None:
        return np.nan
    ha = np.array(ankle) - np.array(hip)
    hk = np.array(knee) - np.array(hip)
    ha_len = np.linalg.norm(ha)
    if ha_len < 1e-6:
        return np.nan
    proj_len = np.dot(hk, ha) / ha_len
    proj_pt = np.array(hip) + (proj_len / ha_len) * ha
    lat = np.linalg.norm(np.array(knee) - proj_pt)
    return float(np.degrees(np.arctan2(lat, proj_len))) if proj_len > 1e-6 else np.nan

def get_joint(kps, idx, ct=0.3):
    if idx >= len(kps):
        return None
    pt = kps[idx]
    if len(pt) >= 3 and pt[2] < ct:
        return None
    if np.isnan(pt[0]) or np.isnan(pt[1]):
        return None
    return (float(pt[0]), float(pt[1]))

def get_confidence(kps, idx):
    if idx >= len(kps):
        return 0.0
    return float(kps[idx][2]) if len(kps[idx]) >= 3 else 1.0

def detect_dominant_side(seq):
    lc = sum(get_confidence(f, i) for f in seq for i in [L_SHOULDER, L_HIP, L_KNEE, L_ANKLE])
    rc = sum(get_confidence(f, i) for f in seq for i in [R_SHOULDER, R_HIP, R_KNEE, R_ANKLE])
    return 'left' if lc >= rc else 'right'

def compute_rep_metrics(keypoints, conf_threshold=0.3):
    T = len(keypoints)
    if T < 5:
        return {}
    side = detect_dominant_side(keypoints)
    SH, HI, KN, AN = (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE) if side == 'left' else (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE)
    ct = conf_threshold

    knee_a, hip_a, trunk_l, ankle_a, valgus_a = [], [], [], [], []
    hip_y, hip_asym, lb_conf = [], [], []

    for t in range(T):
        kps = keypoints[t]
        sh, hi, kn, an = get_joint(kps,SH,ct), get_joint(kps,HI,ct), get_joint(kps,KN,ct), get_joint(kps,AN,ct)
        knee_a.append(angle_3pts(hi, kn, an))
        hip_a.append(angle_3pts(sh, hi, kn))
        trunk_l.append(trunk_lean_from_vertical(sh, hi))
        ankle_a.append(angle_3pts(kn, an, (an[0], an[1]+30)) if kn and an else np.nan)
        lv = knee_valgus_angle(get_joint(kps,L_HIP,ct), get_joint(kps,L_KNEE,ct), get_joint(kps,L_ANKLE,ct))
        rv = knee_valgus_angle(get_joint(kps,R_HIP,ct), get_joint(kps,R_KNEE,ct), get_joint(kps,R_ANKLE,ct))
        vals = [v for v in [lv, rv] if v is not None and np.isfinite(v)]
        valgus_a.append(max(vals) if vals else np.nan)
        hip_y.append(hi[1] if hi else np.nan)
        l_hi, r_hi = get_joint(kps,L_HIP,ct), get_joint(kps,R_HIP,ct)
        if l_hi and r_hi and sh and hi:
            tl = abs(sh[1]-hi[1])
            hip_asym.append(abs(l_hi[1]-r_hi[1])/tl if tl > 10 else np.nan)
        else:
            hip_asym.append(np.nan)
        lb_conf.append(np.mean([get_confidence(kps,i) for i in [HI,KN,AN,SH]]))

    ka, ha, tl_a = np.array(knee_a,dtype=float), np.array(hip_a,dtype=float), np.array(trunk_l,dtype=float)
    aa, va = np.array(ankle_a,dtype=float), np.array(valgus_a,dtype=float)
    hy, asy, conf = np.array(hip_y,dtype=float), np.array(hip_asym,dtype=float), np.array(lb_conf,dtype=float)

    sig = hy.copy()
    v = np.isfinite(sig)
    if np.sum(v) >= 3:
        ix = np.arange(len(sig))
        sig[~v] = np.interp(ix[~v], ix[v], sig[v])
        if len(sig) > 5:
            sig = np.convolve(sig, np.ones(5)/5, mode='same')
        bf = int(np.argmax(sig))
    else:
        bf = T // 2

    def _s(arr, fn):
        f = arr[np.isfinite(arr)]
        return float(fn(f)) if len(f) > 0 else np.nan

    return {
        'knee_min_angle': _s(ka, np.min), 'knee_max_angle': _s(ka, np.max),
        'knee_rom': _s(ka, np.ptp), 'knee_angle_std': _s(ka, np.std),
        'trunk_max_lean': _s(tl_a, np.max), 'trunk_mean_lean': _s(tl_a, np.mean),
        'trunk_lean_at_bottom': float(tl_a[bf]) if bf < len(tl_a) and np.isfinite(tl_a[bf]) else np.nan,
        'trunk_lean_std': _s(tl_a, np.std),
        'hip_min_angle': _s(ha, np.min), 'hip_mean_angle': _s(ha, np.mean), 'hip_rom': _s(ha, np.ptp),
        'ankle_min_angle': _s(aa, np.min), 'ankle_mean_angle': _s(aa, np.mean),
        'max_knee_valgus': _s(va, np.max), 'mean_knee_valgus': _s(va, np.mean),
        'max_hip_asymmetry': _s(asy, np.max), 'mean_hip_asymmetry': _s(asy, np.mean),
        'descent_ratio': float(max(1,bf)/T) if T > 0 else np.nan,
        'total_frames': T, 'mean_confidence': _s(conf, np.mean),
    }

METRIC_DEFINITIONS = {
    'knee_min_angle': {
        'direction': 'higher_is_risky', 'category': 'depth',
        'description': 'Squat depth appears insufficient — knee angle at bottom was {value:.0f}° (threshold: {threshold:.0f}°).',
        'coaching_cue': 'Try sitting back further and pushing your hips down. Imagine sitting into a low chair behind you.',
        'positive_below': 'Good depth — knee angle reached {value:.0f}° at the bottom.',
    },
    'knee_rom': {
        'direction': 'lower_is_risky', 'category': 'range_of_motion',
        'description': 'Limited knee ROM — only {value:.0f}° detected (threshold: {threshold:.0f}°).',
        'coaching_cue': 'Focus on full-range squats. Use a box or bench as a depth target.',
        'positive_above': 'Good range of motion ({value:.0f}° knee ROM).',
    },
    'knee_angle_std': {
        'direction': 'higher_is_risky', 'category': 'stability',
        'description': 'Inconsistent knee angle — variability of {value:.1f}° suggests instability (threshold: {threshold:.1f}°).',
        'coaching_cue': 'Focus on smooth, controlled movement. Pause at the bottom to build positional control.',
        'positive_below': 'Smooth, consistent knee movement pattern.',
    },
    'trunk_max_lean': {
        'direction': 'higher_is_risky', 'category': 'trunk_lean',
        'description': 'Excessive forward lean — trunk tilted {value:.0f}° from vertical (threshold: {threshold:.0f}°). This shifts load onto the lower back.',
        'coaching_cue': 'Keep your chest up. Brace your core and think about driving your elbows forward.',
        'positive_below': 'Upright torso maintained throughout the movement.',
    },
    'trunk_mean_lean': {
        'direction': 'higher_is_risky', 'category': 'trunk_lean',
        'description': 'Average trunk lean of {value:.0f}° is higher than typical safe reps (threshold: {threshold:.0f}°).',
        'coaching_cue': 'Imagine showing the logo on your shirt to someone in front of you.',
    },
    'trunk_lean_at_bottom': {
        'direction': 'higher_is_risky', 'category': 'trunk_lean',
        'description': 'Trunk lean of {value:.0f}° at the bottom (threshold: {threshold:.0f}°). Losing torso position at depth may indicate core weakness or limited ankle mobility.',
        'coaching_cue': 'Spend more time pausing at the bottom with lighter weight to build positional strength.',
    },
    'trunk_lean_std': {
        'direction': 'higher_is_risky', 'category': 'trunk_lean',
        'description': 'Trunk lean variability of {value:.1f}° suggests the torso was not controlled (threshold: {threshold:.1f}°).',
        'coaching_cue': 'Brace your core before descending and maintain that tension throughout.',
    },
    'hip_min_angle': {
        'direction': 'lower_is_risky', 'category': 'hip_hinge',
        'description': 'Hip angle closed to {value:.0f}° — excessive forward fold (threshold: {threshold:.0f}°).',
        'coaching_cue': 'Push your knees forward and out rather than hinging excessively at the hips.',
    },
    'max_knee_valgus': {
        'direction': 'higher_is_risky', 'category': 'knee_tracking',
        'description': 'Knee valgus — knees collapsing inward ({value:.0f}° deviation, threshold: {threshold:.0f}°).',
        'coaching_cue': 'Push your knees out over your toes. A resistance band around the knees can help.',
        'positive_below': 'Knees tracking well — no valgus detected.',
    },
    'max_hip_asymmetry': {
        'direction': 'higher_is_risky', 'category': 'symmetry',
        'description': 'Hip shift detected — one hip drops lower ({value:.3f} normalised, threshold: {threshold:.3f}).',
        'coaching_cue': 'Try single-leg exercises (Bulgarian split squats, lunges) to address imbalance.',
        'positive_below': 'Symmetrical movement — hips level throughout.',
    },
    'descent_ratio': {
        'direction': 'lower_is_risky', 'category': 'tempo',
        'description': 'Quick descent — only {value:.0%} of the rep (threshold: {threshold:.0%}). Controlled eccentrics build more strength.',
        'coaching_cue': 'Try a 2-3 second controlled descent.',
        'positive_above': 'Good tempo — controlled descent and ascent.',
    },
}

def _optimal_threshold_for_metric(safe_vals, risky_vals, direction):
    safe_vals = safe_vals[np.isfinite(safe_vals)]
    risky_vals = risky_vals[np.isfinite(risky_vals)]
    if len(safe_vals) < 10 or len(risky_vals) < 10:
        return None, 0.0, 0.0, 0.5
    pooled_std = np.sqrt((safe_vals.std()**2 + risky_vals.std()**2) / 2)
    if pooled_std < 1e-6:
        return None, 0.0, 0.0, 0.5
    effect_size = abs(risky_vals.mean() - safe_vals.mean()) / pooled_std
    all_vals = np.concatenate([safe_vals, risky_vals])
    all_labels = np.concatenate([np.zeros(len(safe_vals)), np.ones(len(risky_vals))])
    thresholds = np.percentile(all_vals, np.arange(5, 96, 1))
    best_j, best_thresh = -1, None
    tpr_l, fpr_l = [], []
    for th in thresholds:
        preds = (all_vals >= th).astype(int) if direction == 'higher_is_risky' else (all_vals <= th).astype(int)
        tp = np.sum((preds==1) & (all_labels==1))
        fp = np.sum((preds==1) & (all_labels==0))
        fn = np.sum((preds==0) & (all_labels==1))
        tn = np.sum((preds==0) & (all_labels==0))
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0
        j = tpr - fpr
        tpr_l.append(tpr); fpr_l.append(fpr)
        if j > best_j:
            best_j, best_thresh = j, float(th)
    si = np.argsort(fpr_l)
    auc = float(np.trapz(np.array(tpr_l)[si], np.array(fpr_l)[si]))
    return best_thresh, best_j, effect_size, abs(auc)

def calibrate_thresholds(kp_path, labels_path, segments_path, conf_threshold=0.3):
    from tqdm import tqdm
    print("="*60); print("  CALIBRATING FEEDBACK THRESHOLDS FROM TRAINING DATA"); print("="*60)
    kp = pd.read_csv(kp_path)
    lb = pd.read_csv(labels_path)
    lb = lb[lb['label'].isin(['safe','risky'])].copy()
    sg = pd.read_csv(segments_path)
    def nf(f): return os.path.splitext(os.path.basename(str(f)))[0].lower().strip()
    kp['_fn'] = kp['file'].apply(nf)
    lb['_fn'] = lb['file'].apply(nf)
    sg['_fn'] = sg['source_file'].apply(nf) if 'source_file' in sg.columns else sg['file'].apply(nf)
    kp_g = {n: g for n, g in kp.groupby('_fn')}

    all_m, all_l = [], []
    skipped = Counter()
    for _, row in tqdm(lb.iterrows(), total=len(lb), desc="  Analysing"):
        fn, rid, label = row['_fn'], row.get('rep_id', 0), row['label']
        seg = sg[sg['_fn'] == fn]
        if 'rep_id' in sg.columns:
            s2 = seg[seg['rep_id'] == rid]
            if len(s2) > 0: seg = s2
        if len(seg) == 0: skipped['no_segment'] += 1; continue
        if fn not in kp_g: skipped['no_keypoints'] += 1; continue
        sf, ef = int(seg.iloc[0].get('start_frame', 0)), int(seg.iloc[0].get('end_frame', 64))
        rk = kp_g[fn]
        rk = rk[(rk['frame'] >= sf) & (rk['frame'] <= ef)].sort_values('frame')
        if len(rk) < 10: skipped['too_short'] += 1; continue
        fd = []
        for _, fr in rk.iterrows():
            joints = []
            for jn in COCO_JOINTS:
                x = fr.get(f'{jn}_x', np.nan); y = fr.get(f'{jn}_y', np.nan)
                x = np.nan if pd.isna(x) else float(x)
                y = np.nan if pd.isna(y) else float(y)
                c = fr.get(f'{jn}_score', fr.get(f'{jn}_conf', fr.get(f'{jn}_visibility', 1.0)))
                c = 1.0 if pd.isna(c) else float(c)
                joints.append([x, y, c])
            fd.append(joints)
        fd = np.array(fd, dtype=np.float32)
        m = compute_rep_metrics(fd, conf_threshold)
        if m: all_m.append(m); all_l.append(label)

    df = pd.DataFrame(all_m); df['label'] = all_l
    safe_df, risky_df = df[df['label']=='safe'], df[df['label']=='risky']
    calibrated = {}
    for mn, md in METRIC_DEFINITIONS.items():
        if mn not in df.columns: continue
        sv, rv = safe_df[mn].values, risky_df[mn].values
        th, j, es, auc = _optimal_threshold_for_metric(sv, rv, md['direction'])
        if th is not None:
            calibrated[mn] = {
                'threshold': round(th,3), 'direction': md['direction'],
                'j_score': round(j,3), 'effect_size': round(es,3), 'auc': round(auc,3),
                'safe_mean': round(float(np.nanmean(sv)),3), 'safe_std': round(float(np.nanstd(sv)),3),
                'risky_mean': round(float(np.nanmean(rv)),3), 'risky_std': round(float(np.nanstd(rv)),3),
                'category': md['category'], 'importance': round(j * es, 4),
            }
    return {'calibrated_thresholds': calibrated, 'n_safe': len(safe_df), 'n_risky': len(risky_df), 'n_total': len(df)}


@dataclass
class FormIssue:
    category: str; severity: str; metric_name: str
    metric_value: float; threshold: float
    message: str; coaching_cue: str
    importance: float = 0.0; phase: str = "overall"
    is_borderline: bool = False 

@dataclass
class FeedbackReport:
    prediction: str = ""
    confidence: float = 0.0
    model_name: str = ""
    overall_assessment: str = ""
    metrics: dict = field(default_factory=dict)
    issues: List[FormIssue] = field(default_factory=list)
    positive_notes: List[str] = field(default_factory=list)
    coaching_summary: str = ""
    data_quality_note: str = ""
    temporal_pattern_note: str = "" 

class FeedbackGenerator:
    DEFAULTS = {
        'knee_min_angle':       {'threshold': 130.0, 'direction': 'higher_is_risky', 'importance': 0.8},
        'knee_rom':             {'threshold': 40.0,  'direction': 'lower_is_risky',  'importance': 0.6},
        'knee_angle_std':       {'threshold': 25.0,  'direction': 'higher_is_risky', 'importance': 0.5},
        'trunk_max_lean':       {'threshold': 45.0,  'direction': 'higher_is_risky', 'importance': 0.7},
        'trunk_mean_lean':      {'threshold': 30.0,  'direction': 'higher_is_risky', 'importance': 0.5},
        'trunk_lean_at_bottom': {'threshold': 45.0,  'direction': 'higher_is_risky', 'importance': 0.6},
        'trunk_lean_std':       {'threshold': 10.0,  'direction': 'higher_is_risky', 'importance': 0.4},
        'hip_min_angle':        {'threshold': 60.0,  'direction': 'lower_is_risky',  'importance': 0.5},
        'max_knee_valgus':      {'threshold': 15.0,  'direction': 'higher_is_risky', 'importance': 0.6},
        'max_hip_asymmetry':    {'threshold': 0.08,  'direction': 'higher_is_risky', 'importance': 0.4},
        'descent_ratio':        {'threshold': 0.25,  'direction': 'lower_is_risky',  'importance': 0.3},
    }

    def __init__(self, calibration=None):
        self.calibrated = False
        if calibration and 'calibrated_thresholds' in calibration:
            self.thresholds = calibration['calibrated_thresholds']
            self.calibrated = True
        else:
            self.thresholds = {k: dict(v) for k, v in self.DEFAULTS.items()}

    @classmethod
    def load_calibrated(cls, path):
        with open(path) as f:
            cal = json.load(f)
        gen = cls(calibration=cal)
        print(f"  Loaded calibrated thresholds ({len(gen.thresholds)} metrics)")
        return gen

    def analyse_rep(self, keypoints, prediction="", confidence=0.0,
                    model_name="", model_type="baseline"):
        
        T = len(keypoints)
        if T < 5:
            return FeedbackReport(
                prediction=prediction, confidence=confidence, model_name=model_name,
                overall_assessment="Sequence too short for analysis."
            )
        metrics = compute_rep_metrics(keypoints, CONFIG['confidence_min'])
        if not metrics:
            return FeedbackReport(
                prediction=prediction, confidence=confidence, model_name=model_name,
                overall_assessment="Could not compute metrics — joint detection insufficient."
            )

        is_gcn = (model_type == "gcn")
        issues = self._detect_issues(metrics, prediction=prediction, is_gcn=is_gcn)
        positives = self._detect_positives(metrics)
        qn = self._quality_note(metrics)
        overall, coaching, temporal_pattern_note = self._summary(
            prediction, confidence, issues, positives, is_gcn=is_gcn
        )

        return FeedbackReport(
            prediction=prediction, confidence=confidence, model_name=model_name,
            overall_assessment=overall, metrics=metrics, issues=issues,
            positive_notes=positives, coaching_summary=coaching,
            data_quality_note=qn, temporal_pattern_note=temporal_pattern_note
        )

    def _detect_issues(self, metrics, prediction="", is_gcn=False):
        
        issues = []
        seen = set()
        sorted_m = sorted(
            self.thresholds.items(),
            key=lambda x: x[1].get('importance', 0),
            reverse=True
        )

        for mn, tc in sorted_m:
            if mn not in metrics or mn not in METRIC_DEFINITIONS:
                continue
            val = metrics[mn]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue

            th = tc['threshold']
            d = tc['direction']
            imp = tc.get('importance', 0)
            md = METRIC_DEFINITIONS[mn]
            cat = md['category']

            if cat in seen:
                continue

            
            if d == 'higher_is_risky':
                excess = (val - th) / max(abs(th), 1e-6)
                triggered = val > th
                
                borderline = (not triggered) and (val > th * 0.80)
            else:
                excess = (th - val) / max(abs(th), 1e-6)
                triggered = val < th
                
                borderline = (not triggered) and (val < th * 1.20)

            if triggered:
                
                if prediction == "safe":
                    sev = "info"
                    msg = "Advisory: " + md['description'].format(value=val, threshold=th)
                    advisory = True
                else:
                    sev = "concern" if (excess > 0.3 or imp > 0.6) else (
                        "warning" if excess > 0.1 else "info"
                    )
                    msg = md['description'].format(value=val, threshold=th)
                    advisory = False
                issues.append(FormIssue(
                    category=cat, severity=sev, metric_name=mn,
                    metric_value=round(val, 2), threshold=round(th, 2),
                    message=msg,
                    coaching_cue=md['coaching_cue'],
                    importance=imp if not advisory else imp * 0.4,
                    is_borderline=advisory
                ))
                seen.add(cat)

            elif is_gcn and prediction == "risky" and borderline and imp >= 0.4:
                
                borderline_msg = (
                    f"{mn.replace('_', ' ').capitalize()} ({val:.1f}) was near the threshold "
                    f"for concern ({th:.1f}) and may be contributing to the model's assessment."
                )
                issues.append(FormIssue(
                    category=cat, severity="info", metric_name=mn,
                    metric_value=round(val, 2), threshold=round(th, 2),
                    message=borderline_msg,
                    coaching_cue=md.get('coaching_cue', 'Consider reviewing this aspect of your form.'),
                    importance=imp * 0.5,  
                    is_borderline=True
                ))
                seen.add(cat)

        sev_ord = {"concern": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda x: (-x.importance, sev_ord.get(x.severity, 3)))
        return issues

    def _detect_positives(self, metrics):
        pos = []
        for mn, tc in sorted(self.thresholds.items(),
                              key=lambda x: x[1].get('importance', 0), reverse=True):
            if mn not in metrics or mn not in METRIC_DEFINITIONS:
                continue
            val = metrics[mn]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            th, d, md = tc['threshold'], tc['direction'], METRIC_DEFINITIONS[mn]
            tk = 'positive_below' if d == 'higher_is_risky' else 'positive_above'
            ok = (d == 'higher_is_risky' and val <= th) or (d == 'lower_is_risky' and val >= th)
            if ok and tk in md:
                pos.append(md[tk].format(value=val, threshold=th))
        return pos[:4]

    def _quality_note(self, metrics):
        c = metrics.get('mean_confidence', 1.0)
        if isinstance(c, float) and c < 0.5:
            return "Low detection confidence — feedback may be less reliable."
        return ""

    def _summary(self, pred, conf, issues, positives, is_gcn=False):
        
        concerns = [i for i in issues if i.severity == "concern"]
        warnings = [i for i in issues if i.severity == "warning"]
        info_items = [i for i in issues if i.severity == "info"]
        temporal_pattern_note = ""

        if pred == "risky":
            if concerns:
                overall = (
                    f"This rep was classified as risky ({conf:.0%} confidence). "
                    f"Primary concern: {concerns[0].category.replace('_', ' ')}."
                )
            elif warnings:
                overall = (
                    f"This rep was classified as risky ({conf:.0%} confidence). "
                    f"Key area: {warnings[0].category.replace('_', ' ')}."
                )
            elif info_items:
                
                overall = (
                    f"This rep was classified as risky ({conf:.0%} confidence). "
                    f"No metrics clearly exceeded thresholds, but borderline values "
                    f"were detected that may be contributing."
                )
            else:
                
                overall = f"This rep was classified as risky ({conf:.0%} confidence)."
                if is_gcn:
                    temporal_pattern_note = (
                        "The model detected a risky movement pattern in the temporal "
                        "dynamics of this repetition that may not be visible as a "
                        "single-point biomechanical deviation. Consider reviewing the "
                        "skeleton overlay video for any mid-rep instability, "
                        "compensatory movement, or loss of position during the "
                        "descent or ascent."
                    )

        elif pred == "safe":
            overall = (
                f"This rep was classified as safe ({conf:.0%} confidence)."
                + (" Some aspects could still be improved." if concerns else " Form looks solid.")
            )
        else:
            overall = (
                f"Analysis found {len(concerns)} concern(s)." if concerns
                else "Form looks good."
            )

        act = [i for i in issues if i.severity in ("concern", "warning")]
        if act:
            coaching = act[0].coaching_cue
            if len(act) > 1:
                coaching += f" Additionally: {act[1].coaching_cue}"
        elif temporal_pattern_note:
            coaching = "Review the skeleton overlay to identify where the movement broke down."
        elif positives:
            coaching = "Form looks good — maintain this quality as you increase weight."
        else:
            coaching = "No specific corrections identified."

        return overall, coaching, temporal_pattern_note

    def format_report_text(self, report):
        L = ["="*60, "  SQUAT FORM FEEDBACK"]
        if self.calibrated:
            L.append("  (thresholds calibrated from training data)")
        L.append("="*60)
        if report.model_name:
            L.append(f"  Model: {report.model_name}")
        if report.prediction:
            c = f" ({report.confidence:.0%} confidence)" if report.confidence > 0 else ""
            L.append(f"  Classification: {report.prediction.upper()}{c}")
        L.append(f"\n  {report.overall_assessment}\n")

        if report.temporal_pattern_note:
            L.append(f"  [temporal] {report.temporal_pattern_note}\n")

        m = report.metrics
        if m:
            L.append("  --- Key Metrics ---")
            for k, lbl, u, fmt in [
                ('knee_min_angle','Squat depth (knee angle)','°','.1f'),
                ('knee_rom','Knee ROM','°','.1f'),
                ('knee_angle_std','Knee variability','°','.1f'),
                ('trunk_max_lean','Max trunk lean','°','.1f'),
                ('hip_min_angle','Hip angle (bottom)','°','.1f'),
                ('max_knee_valgus','Knee valgus','°','.1f'),
                ('max_hip_asymmetry','Hip asymmetry','','.3f'),
                ('descent_ratio','Descent ratio','','.2f'),
                ('total_frames','Frames','','d'),
            ]:
                v = m.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    ti = ""
                    if k in self.thresholds:
                        t = self.thresholds[k]['threshold']
                        d = self.thresholds[k]['direction']
                        arr = "↑" if d == 'higher_is_risky' else "↓"
                        ti = f"  (threshold: {t:{fmt}}{u} {arr})"
                    L.append(f"  {lbl+':':<30s} {v:{fmt}}{u}{ti}")
            L.append("")

        if report.issues:
            L.append("  --- Areas for Improvement ---")
            for i in report.issues:
                ic = {"concern":"[!]","warning":"[~]","info":"[i]"}[i.severity]
                imp = f" (importance: {i.importance:.2f})" if self.calibrated else ""
                L.append(f"  {ic} {i.message}{imp}")
                L.append(f"      Tip: {i.coaching_cue}")
                L.append("")

        if report.positive_notes:
            L.append("  --- What's Going Well ---")
            for n in report.positive_notes:
                L.append(f"  [+] {n}")
            L.append("")

        if report.coaching_summary:
            L.append("  --- Priority Focus ---")
            L.append(f"  {report.coaching_summary}")
            L.append("")

        if report.data_quality_note:
            L.append(f"  Note: {report.data_quality_note}\n")

        L.append("="*60)
        return "\n".join(L)

    def report_to_dict(self, report):
        def _sr(v, dp=1):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            return round(v, dp)
        return {
            'prediction': report.prediction,
            'confidence': _sr(report.confidence, 4),
            'model_name': report.model_name,
            'overall_assessment': report.overall_assessment,
            'coaching_summary': report.coaching_summary,
            'temporal_pattern_note': report.temporal_pattern_note,
            'positive_notes': report.positive_notes,
            'metrics': {
                k: _sr(v) if isinstance(v, float) else v
                for k, v in report.metrics.items() if k != 'total_frames'
            },
            'issues': [
                {
                    'category': i.category, 'severity': i.severity,
                    'metric_name': i.metric_name,
                    'metric_value': _sr(i.metric_value),
                    'threshold': _sr(i.threshold),
                    'importance': _sr(i.importance, 4),
                    'message': i.message,
                    'coaching_cue': i.coaching_cue
                }
                for i in report.issues
            ],
        }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Squat form feedback generator")
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--keypoints-csv', default=CONFIG['keypoints_raw_path'])
    p.add_argument('--labels-csv', default=CONFIG['labels_path'])
    p.add_argument('--segments-csv', default=CONFIG['segments_path'])
    p.add_argument('--thresholds-path', default=CONFIG['thresholds_save_path'])
    p.add_argument('--demo', action='store_true')
    args = p.parse_args()

    if args.calibrate:
        kp_path = args.keypoints_csv
        if not os.path.exists(kp_path):
            kp_path = CONFIG['keypoints_path']
        if not os.path.exists(kp_path):
            print(f"  ERROR: No keypoints CSV found")
            sys.exit(1)
        result = calibrate_thresholds(kp_path, args.labels_csv, args.segments_csv)
        os.makedirs(os.path.dirname(args.thresholds_path), exist_ok=True)
        with open(args.thresholds_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved to {args.thresholds_path}")

    elif args.demo:
        print("="*60)
        print("  FEEDBACK GENERATOR — DEMO (GCN risky, no clear metrics)")
        print("="*60)
        np.random.seed(42)
        
        fake = np.random.randn(50, 17, 3) * 50 + 300
        fake[:, :, 2] = 0.8
        gen = FeedbackGenerator()
        
        r = gen.analyse_rep(fake, prediction="risky", confidence=0.82,
                            model_name="MS-G3D Pretrained", model_type="gcn")
        print(gen.format_report_text(r))
        print("\n\n")
        print("="*60)
        print("  DEMO (Baseline risky)")
        print("="*60)
        r2 = gen.analyse_rep(fake, prediction="risky", confidence=0.75,
                             model_name="LightGBM", model_type="baseline")
        print(gen.format_report_text(r2))
    else:
        print("  Usage:")
        print("    python generate_feedback.py --calibrate")
        print("    python generate_feedback.py --demo")

if __name__ == "__main__":
    main()