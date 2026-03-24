#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import math
import subprocess
import time
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'keypoints_path': 'frame_level/pose_keypoints_raw.csv',
    'labels_path': 'rep_level/rep_labels_merged.csv',
    'segments_path': 'rep_level/auto_segments.csv',
    'model_save_path': 'models/msg3d_pretrained_finetuned.pt',
    'results_save_path': 'models/msg3d_pretrained_results.json',
    'checkpoint_url': (
        'http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/'
        'msg3d_pyskl_ntu120_xsub_hrnet/j.pth'
    ),
    'checkpoint_local': 'models/msg3d_pyskl_ntu120_xsub_hrnet_j.pth',
    'sequence_length': 64,
    'num_joints': 17,
    'in_channels': 3,
    'num_person': 2,
    'num_stages': 3,
    'base_channels': 96,
    'num_sgcn_subsets': 13,
    'num_g3d_subsets': 6,
    'g3d_windows': [3, 5],
    'mstcn_branches': 6,
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
    'use_mixed_precision': True,
    'weight_decay': 5e-4,
    'label_smoothing': 0.1,
    'gradient_clip_norm': 1.0,
    'test_size': 0.2,
    'threshold_search': True,
    'confidence_analysis': True,
    'error_analysis': True,
    'random_seed': 42,
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

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]



def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def compute_precision(y_true, y_pred, pos_label=1):
    
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    pp = np.sum(y_pred == pos_label)
    return tp / pp if pp > 0 else 0.0


def compute_recall(y_true, y_pred, pos_label=1):
  
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    ap = np.sum(y_true == pos_label)
    return tp / ap if ap > 0 else 0.0


def compute_f1_score(y_true, y_pred, pos_label=1):
    
    p = compute_precision(y_true, y_pred, pos_label)
    r = compute_recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_balanced_accuracy(y_true, y_pred):
   
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = []
    for cls in np.unique(y_true):
        mask = (y_true == cls)
        if mask.sum() > 0:
            recalls.append(np.sum(y_pred[mask] == cls) / mask.sum())
    return np.mean(recalls)


def compute_accuracy(y_true, y_pred):
   
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def compute_matthews_corrcoef(y_true, y_pred):
   
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den > 0 else 0.0


def compute_roc_auc(y_true, y_scores, num_thresholds=1000):
    
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    if total_pos == 0 or total_neg == 0:
        return 0.5
    thresholds = np.linspace(y_scores.min() - 0.001, y_scores.max() + 0.001, num_thresholds)
    tpr_list, fpr_list = [], []
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)
    tpr_arr, fpr_arr = np.array(tpr_list), np.array(fpr_list)
    idx = np.argsort(fpr_arr)
    fpr_s, tpr_s = fpr_arr[idx], tpr_arr[idx]
    auc = 0.0
    for i in range(1, len(fpr_s)):
        auc += (fpr_s[i] - fpr_s[i-1]) * (tpr_s[i] + tpr_s[i-1]) / 2
    return auc


def compute_average_precision(y_true, y_scores, num_thresholds=1000):
    
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    thresholds = np.linspace(y_scores.max() + 0.001, y_scores.min() - 0.001, num_thresholds)
    precs, recs = [], []
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        precs.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    precs, recs = np.array(precs), np.array(recs)
    idx = np.argsort(recs)
    recs_s, precs_s = recs[idx], precs[idx]
    ap = 0.0
    for i in range(1, len(recs_s)):
        ap += (recs_s[i] - recs_s[i-1]) * precs_s[i]
    return ap


def format_classification_report(y_true, y_pred, target_names=None, digits=3):
   
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = sorted(np.unique(y_true))
    if target_names is None:
        target_names = [str(c) for c in classes]
    rows = []
    total = len(y_true)
    w = max(max(len(n) for n in target_names) + 2, 14)
    for cls, name in zip(classes, target_names):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append((name, p, r, f, int(np.sum(y_true == cls))))
    macro_p = np.mean([r[1] for r in rows])
    macro_r = np.mean([r[2] for r in rows])
    macro_f = np.mean([r[3] for r in rows])
    wt_p = sum(r[1]*r[4] for r in rows) / total
    wt_r = sum(r[2]*r[4] for r in rows) / total
    wt_f = sum(r[3]*r[4] for r in rows) / total
    acc = np.mean(y_true == y_pred)
    header = f"{'':>{w}s}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}"
    lines = ["\n" + header + "\n"]
    for name, p, r, f, s in rows:
        lines.append(f"{name:>{w}s}{p:>12.{digits}f}{r:>12.{digits}f}{f:>12.{digits}f}{s:>12d}")
    lines.append("")
    lines.append(f"{'accuracy':>{w}s}{'':>12}{'':>12}{acc:>12.{digits}f}{total:>12d}")
    lines.append(f"{'macro avg':>{w}s}{macro_p:>12.{digits}f}{macro_r:>12.{digits}f}{macro_f:>12.{digits}f}{total:>12d}")
    lines.append(f"{'weighted avg':>{w}s}{wt_p:>12.{digits}f}{wt_r:>12.{digits}f}{wt_f:>12.{digits}f}{total:>12d}")
    return "\n".join(lines)

def compute_specificity(y_true, y_pred):
    
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def compute_g_mean(y_true, y_pred):
    
    sens = compute_recall(y_true, y_pred, pos_label=1)
    spec = compute_specificity(y_true, y_pred)
    return math.sqrt(sens * spec)

def compute_cohens_kappa(y_true, y_pred):
  
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    acc = np.mean(y_true == y_pred)
   
    p_true = np.bincount(y_true, minlength=2) / n
    p_pred = np.bincount(y_pred, minlength=2) / n
    pe = np.sum(p_true * p_pred)
    return (acc - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

def compute_brier_score(y_true, y_prob):
   
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))

def compute_log_loss(y_true, y_prob, eps=1e-15):
    
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob), eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_boot=2000, ci=0.95, seed=42):
    
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
        return (float('nan'), float('nan'))
    alpha = (1 - ci) / 2
    return (float(np.percentile(scores, 100 * alpha)),
            float(np.percentile(scores, 100 * (1 - alpha))))


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        ci = np.where(y == cls)[0]
        rng.shuffle(ci)
        nt = max(1, min(int(round(len(ci) * test_size)), len(ci) - 1))
        test_idx.extend(ci[:nt])
        train_idx.extend(ci[nt:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def apply_smote(sequences, labels, config):
    
    if not config.get('use_smote', False):
        return sequences, labels
    print("\n   Applying SMOTE...")
    rng = np.random.RandomState(config['random_seed'])
    safe_count = int(np.sum(labels == 0))
    risky_count = int(np.sum(labels == 1))
    print(f"      Before: Safe={safe_count}, Risky={risky_count}")
    target_risky = int(safe_count * config.get('smote_sampling_strategy', 0.75))
    n_synthetic = max(0, target_risky - risky_count)
    if n_synthetic == 0:
        return sequences, labels
    print(f"      Generating {n_synthetic} synthetic risky samples")
    orig_shape = sequences.shape[1:]
    flat = sequences.reshape(len(sequences), -1)
    minority_idx = np.where(labels == 1)[0]
    minority_flat = flat[minority_idx]
    n_min = len(minority_idx)
    k = min(config.get('smote_k_neighbors', 5), n_min - 1)
    k = max(1, k)
    
    all_neighbors = []
    for i in range(n_min):
        dists = np.sqrt(np.sum((minority_flat - minority_flat[i:i+1]) ** 2, axis=1))
        dists[i] = np.inf
        all_neighbors.append(np.argpartition(dists, k)[:k].tolist())
    
    synthetic = []
    for _ in range(n_synthetic):
        src = rng.randint(0, n_min)
        nbr = rng.choice(all_neighbors[src])
        lam = rng.uniform(0.0, 1.0)
        synthetic.append(minority_flat[src] + lam * (minority_flat[nbr] - minority_flat[src]))
    synthetic = np.array(synthetic).reshape(-1, *orig_shape)
    syn_labels = np.ones(n_synthetic, dtype=labels.dtype)
    aug_seq = np.concatenate([sequences, synthetic], axis=0)
    aug_lab = np.concatenate([labels, syn_labels], axis=0)
    shuffle = rng.permutation(len(aug_seq))
    print(f"      After: Safe={int(np.sum(aug_lab==0))}, Risky={int(np.sum(aug_lab==1))}")
    return aug_seq[shuffle], aug_lab[shuffle]


class CosineAnnealingScheduler:
   
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.step_count = 0
        self.initial_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        for group, init_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            group['lr'] = (self.eta_min + 0.5 * (init_lr - self.eta_min)
                           * (1 + math.cos(math.pi * self.step_count / self.T_max)))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def build_coco_adjacency():
    
    num_joints = 17
    adj = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in COCO_EDGES:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def build_multiscale_adjacency(num_joints, edges, max_hop):
   
    
    A1 = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A1[i, j] = 1
        A1[j, i] = 1
    A0 = np.eye(num_joints, dtype=np.float32)

    
    hop_dist = np.full((num_joints, num_joints), np.inf)
    for start in range(num_joints):
        hop_dist[start, start] = 0
        queue = [start]
        visited = {start}
        while queue:
            cur = queue.pop(0)
            for nbr in range(num_joints):
                if A1[cur, nbr] > 0 and nbr not in visited:
                    hop_dist[start, nbr] = hop_dist[start, cur] + 1
                    visited.add(nbr)
                    queue.append(nbr)

    
    adjacency_list = []
    for hop in range(max_hop + 1):
        Ah = np.zeros((num_joints, num_joints), dtype=np.float32)
        for i in range(num_joints):
            for j in range(num_joints):
                if hop_dist[i, j] == hop:
                    Ah[i, j] = 1
        adjacency_list.append(Ah)

    
    normed = []
    for A in adjacency_list:
        A_hat = A + np.eye(num_joints) * 1e-6
        D = A_hat.sum(axis=1)
        D_inv_sqrt = np.power(D + 1e-6, -0.5)
        D_mat = np.diag(D_inv_sqrt)
        normed.append((D_mat @ A_hat @ D_mat).astype(np.float32))

    return normed


def build_g3d_adjacency(num_joints, edges, window_size, num_subsets=6):
  
    V = num_joints
    VW = V * window_size

   
    A_spatial = np.zeros((V, V), dtype=np.float32)
    for i, j in edges:
        A_spatial[i, j] = 1
        A_spatial[j, i] = 1

   
    hop_dist = np.full((V, V), np.inf)
    for start in range(V):
        hop_dist[start, start] = 0
        queue = [start]
        visited = {start}
        while queue:
            cur = queue.pop(0)
            for nbr in range(V):
                if A_spatial[cur, nbr] > 0 and nbr not in visited:
                    hop_dist[start, nbr] = hop_dist[start, cur] + 1
                    visited.add(nbr)
                    queue.append(nbr)

    
    A_list = []
    for s in range(num_subsets):
        A_list.append(np.zeros((VW, VW), dtype=np.float32))

    for t1 in range(window_size):
        for v1 in range(V):
            idx1 = t1 * V + v1
            for t2 in range(window_size):
                for v2 in range(V):
                    idx2 = t2 * V + v2
                    temporal_dist = abs(t1 - t2)
                    spatial_dist = hop_dist[v1, v2]

                    if temporal_dist == 0 and spatial_dist == 0:
                        A_list[0][idx1, idx2] = 1  
                    elif temporal_dist == 0 and spatial_dist == 1:
                        A_list[1][idx1, idx2] = 1  
                    elif temporal_dist == 1 and spatial_dist == 0:
                        A_list[2][idx1, idx2] = 1 
                    elif temporal_dist == 1 and spatial_dist == 1:
                        A_list[3][idx1, idx2] = 1 
                    elif temporal_dist > 1 and spatial_dist == 0:
                        A_list[4][idx1, idx2] = 1  
                    else:
                        A_list[5][idx1, idx2] = 1 

    
    normed = []
    for A in A_list:
        col_sum = A.sum(axis=0)
        col_sum[col_sum == 0] = 1
        normed.append((A / col_sum[np.newaxis, :]).astype(np.float32))

    return np.stack(normed, axis=0)  

class TemporalConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (kernel_size, 1),
                              (stride, 1), (padding, 0), (dilation, 1))
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class Branch4(nn.Module):
   
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.stride = stride
     
        self._modules['0'] = nn.Conv2d(in_ch, out_ch, 1)
        self._modules['1'] = nn.BatchNorm2d(out_ch)
        self._modules['2'] = nn.Identity()  # ReLU placeholder (no params)
        self._modules['3'] = nn.Identity()  # Stride placeholder (no params)
        self._modules['4'] = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self._modules['0'](x)
        x = self._modules['1'](x)
        x = F.relu(x, inplace=True)
        if self.stride > 1:
            x = x[:, :, ::self.stride, :]
        x = self._modules['4'](x)
        return x


class MSTCN(nn.Module):
   
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        br_ch = out_ch // 6
        self.branches = nn.ModuleList()

       
        for i in range(4):
            dilation = i + 1
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, br_ch, 1),                       # .branches.{i}.0
                nn.BatchNorm2d(br_ch),                             # .branches.{i}.1
                nn.ReLU(inplace=True),                             # .branches.{i}.2
                TemporalConv(br_ch, br_ch, 3, stride, dilation),   # .branches.{i}.3
            ))

        
        self.branches.append(Branch4(in_ch, br_ch, stride))

        
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, br_ch, 1),        # .branches.5.0
            nn.BatchNorm2d(br_ch),             # .branches.5.1
        ))

        self._stride = stride
        self._out_ch = out_ch

        
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Module()
            self.residual.conv = nn.Conv2d(in_ch, out_ch, (1, 1), (stride, 1))
            self.residual.bn = nn.BatchNorm2d(out_ch)
            self._has_residual = True
        else:
            self._has_residual = False

    def forward(self, x):
        residual = x
        branch_outs = []

        
        for i in range(4):
            branch_outs.append(self.branches[i](x))

        
        branch_outs.append(self.branches[4](x))

       
        pooled = F.max_pool2d(x, (3, 1), (self._stride, 1), (1, 0))
        b5 = self.branches[5](pooled)
        branch_outs.append(b5)

        out = torch.cat(branch_outs, dim=1)

      
        if self._has_residual:
            residual = self.residual.bn(self.residual.conv(residual))
        elif residual.shape == out.shape:
            pass  # Identity residual
        else:
            residual = 0

        return F.relu(out + residual, inplace=True)


class MultiScaleSGCN(nn.Module):
  
    def __init__(self, in_ch, out_ch, num_subsets, num_joints):
        super().__init__()
        self.num_subsets = num_subsets
        
        self.register_buffer('A', torch.zeros(num_subsets, num_joints, num_joints))
        self.PA = nn.Parameter(torch.zeros(num_subsets, num_joints, num_joints))

        
        self.mlp = nn.Module()
        self.mlp.layers = nn.Sequential(
            nn.Conv2d(in_ch * num_subsets, out_ch, 1),  # .mlp.layers.0
            nn.BatchNorm2d(out_ch),                      # .mlp.layers.1
        )

    def forward(self, x):
        N, C, T, V = x.shape
        A = self.A + self.PA  

        x_flat = x.reshape(N * C * T, V)  # (NCT, V)
        subset_features = []
        for k in range(self.num_subsets):
            feat = torch.matmul(x_flat, A[k])  # (NCT, V)
            subset_features.append(feat.reshape(N, C, T, V))

        
        out = torch.cat(subset_features, dim=1)

        # MLP reduction
        out = self.mlp.layers(out)
        out = F.relu(out, inplace=True)

        return out


class G3DInnerGCN(nn.Module):
    
    def __init__(self, in_ch, mid_ch, num_subsets, graph_size):
        super().__init__()
        self.num_subsets = num_subsets
        self.register_buffer('A', torch.zeros(num_subsets, graph_size, graph_size))
        self.PA = nn.Parameter(torch.zeros(num_subsets, graph_size, graph_size))
        self.mlp = nn.Module()
        self.mlp.layers = nn.Sequential(
            nn.Conv2d(in_ch * num_subsets, mid_ch, 1),
            nn.BatchNorm2d(mid_ch),
        )

    def forward(self, x):
        A = self.A + self.PA  # (K, VW, VW)
        N, C, T, VW = x.shape

        x_flat = x.reshape(N * C * T, VW)  # (NCT, VW)
        subset_feats = []
        for k in range(self.num_subsets):
            feat = torch.matmul(x_flat, A[k])  # (NCT, VW)
            subset_feats.append(feat.reshape(N, C, T, VW))

        out = torch.cat(subset_feats, dim=1)  # (N, K*C, T, VW)
        out = self.mlp.layers(out)
        out = F.relu(out, inplace=True)
        return out


class G3DModule(nn.Module):
   
    def __init__(self, in_ch, mid_ch, out_ch, num_joints, window_size, num_subsets=6):
        super().__init__()
        self.window_size = window_size
        self.num_joints = num_joints
        self.num_subsets = num_subsets
        VW = num_joints * window_size

        self.gcn3d = nn.ModuleList([
            nn.Identity(), 
            G3DInnerGCN(in_ch, mid_ch, num_subsets, VW),  
        ])

        
        self.out_conv = nn.Conv3d(mid_ch, out_ch, (1, window_size, 1),
                                  padding=0, bias=True)
        self.out_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        N, C, T, V = x.shape
        W = self.window_size

        
        pad = W // 2
        x_padded = F.pad(x, (0, 0, pad, pad))  

        
        x_win = x_padded.unfold(2, W, 1) 
        x_win = x_win.permute(0, 1, 2, 4, 3).contiguous()  
        x_win = x_win.reshape(N, C, T, W * V)  

        
        out = self.gcn3d[1](x_win)  

        
        mid_ch = out.shape[1]
        out = out.reshape(N, mid_ch, T, W, V)

        
        out = self.out_conv(out)  # (N, out_ch, T, 1, V)
        out = out.squeeze(3)  # (N, out_ch, T, V)
        out = self.out_bn(out)

        return out


class G3DBlock(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch, num_joints, windows=(3, 5), num_subsets=6):
        super().__init__()
        self.gcn3d = nn.ModuleList()
        for w in windows:
            self.gcn3d.append(G3DModule(in_ch, mid_ch, out_ch, num_joints, w, num_subsets))

    def forward(self, x):
        out = 0
        for module in self.gcn3d:
            out = out + module(x)
        return out


class PYSKL_MSG3D(nn.Module):
   
    def __init__(self, in_ch=3, num_joints=17, num_person=2, num_classes=2,
                 num_subsets=13, num_g3d_subsets=6, windows=(3, 5)):
        super().__init__()
        self.in_ch = in_ch
        self.num_joints = num_joints
        self.num_person = num_person

       
        A_base = build_coco_adjacency()
        self.register_buffer('A', torch.tensor(A_base))

        self.data_bn = nn.BatchNorm1d(in_ch * num_joints * num_person)

        
        ch1 = 96
        self.sgcn1 = nn.ModuleList([
            MultiScaleSGCN(in_ch, ch1, num_subsets, num_joints),
            MSTCN(ch1, ch1, stride=1),
            MSTCN(ch1, ch1, stride=1),
        ])
        self.gcn3d1 = G3DBlock(in_ch, ch1, ch1, num_joints, windows, num_g3d_subsets)
        self.tcn1 = MSTCN(ch1, ch1, stride=1)

        
        ch2 = 192
        self.sgcn2 = nn.ModuleList([
            MultiScaleSGCN(ch1, ch1, num_subsets, num_joints),
            MSTCN(ch1, ch2, stride=1),  
            MSTCN(ch2, ch2, stride=1),
        ])
        self.gcn3d2 = G3DBlock(ch1, ch1, ch2, num_joints, windows, num_g3d_subsets)
        self.tcn2 = MSTCN(ch2, ch2, stride=1)

  
        ch3 = 384
        self.sgcn3 = nn.ModuleList([
            MultiScaleSGCN(ch2, ch2, num_subsets, num_joints),
            MSTCN(ch2, ch3, stride=1),  
            MSTCN(ch3, ch3, stride=1),
        ])
        self.gcn3d3 = G3DBlock(ch2, ch2, ch3, num_joints, windows, num_g3d_subsets)
        self.tcn3 = MSTCN(ch3, ch3, stride=1)

        
        self.cls_head = nn.Module()
        self.cls_head.fc_cls = nn.Linear(ch3, num_classes)

    def forward(self, x):
        
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add person dim: (N, 1, T, V, C)

        N, M, T, V, C = x.shape

        
        if M < self.num_person:
            pad = torch.zeros(N, self.num_person - M, T, V, C,
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            M = self.num_person

       
        x = x.permute(0, 4, 1, 3, 2).contiguous()  # (N, C, M, V, T)
        x = x.reshape(N, C * M * V, T)
        x = self.data_bn(x)

   
        x = x.reshape(N, C, M, V, T).permute(0, 2, 1, 4, 3).contiguous()
        x = x.reshape(N * M, C, T, V)

        
        x_sgcn = self.sgcn1[0](x)     # Spatial GCN: (N*M, 3, T, V) -> (N*M, 96, T, V)
        x_sgcn = self.sgcn1[1](x_sgcn)  # MSTCN
        x_sgcn = self.sgcn1[2](x_sgcn)  # MSTCN
        x_g3d = self.gcn3d1(x)         # G3D: (N*M, 3, T, V) -> (N*M, 96, T, V)
        x = x_sgcn + x_g3d            # Combine
        x = self.tcn1(x)              # TCN

        
        x_sgcn = self.sgcn2[0](x)
        x_sgcn = self.sgcn2[1](x_sgcn)
        x_sgcn = self.sgcn2[2](x_sgcn)
        x_g3d = self.gcn3d2(x)
        x = x_sgcn + x_g3d
        x = self.tcn2(x)

        
        x_sgcn = self.sgcn3[0](x)
        x_sgcn = self.sgcn3[1](x_sgcn)
        x_sgcn = self.sgcn3[2](x_sgcn)
        x_g3d = self.gcn3d3(x)
        x = x_sgcn + x_g3d
        x = self.tcn3(x)

        
        x = x.mean(dim=-1).mean(dim=-1)  # (N*M, 384)
        x = x.reshape(N, M, -1).mean(dim=1)  # (N, 384) average over persons

        return self.cls_head.fc_cls(x)



def download_checkpoint(config):
    
    path = config['checkpoint_local']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   Checkpoint exists: {path} ({size_mb:.1f} MB)")
        return path
    print(f"   Downloading pretrained checkpoint...")
    try:
        import urllib.request
        urllib.request.urlretrieve(config['checkpoint_url'], path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   Downloaded: {size_mb:.1f} MB")
        return path
    except Exception:
        try:
            subprocess.check_call(['wget', '-q', '-O', path, config['checkpoint_url']])
            return path
        except Exception:
            print(f"   Download failed. Place checkpoint at: {path}")
            return None


def load_pretrained_weights(model, checkpoint_path):
    
    print(f"\n   Loading pretrained weights...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif 'model' in checkpoint:
        checkpoint = checkpoint['model']

    model_sd = model.state_dict()
    loaded, skipped_head, skipped_shape, skipped_missing = 0, 0, 0, 0

    for ck, cv in checkpoint.items():
        if 'cls_head' in ck:
            skipped_head += 1
            continue
        
        mk = ck if ck in model_sd else ck.replace('backbone.', '', 1)
        if mk in model_sd:
            if model_sd[mk].shape == cv.shape:
                model_sd[mk] = cv
                loaded += 1
            else:
                skipped_shape += 1
                if skipped_shape <= 5:
                    print(f"   SHAPE: {mk} ckpt={list(cv.shape)} model={list(model_sd[mk].shape)}")
        else:
            skipped_missing += 1
            if skipped_missing <= 5:
                print(f"   MISS: {ck}")

    model.load_state_dict(model_sd)
    total_backbone = len(checkpoint) - skipped_head
    pct = loaded / max(total_backbone, 1) * 100
    print(f"\n   Loaded {loaded}/{total_backbone} backbone layers ({pct:.1f}%)")
    if skipped_shape:
        print(f"   Shape mismatches: {skipped_shape}")
    if skipped_missing:
        print(f"   Missing in model: {skipped_missing}")
    return loaded

def normalize_filename(filename):
   
    if pd.isna(filename):
        return None
    name = str(filename)
    if name.lower().startswith('pose_'):
        name = name[5:]
    return os.path.splitext(name)[0].lower()


def load_data(config):
    
    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)
    kp = pd.read_csv(config['keypoints_path'])
    kp['file_norm'] = kp['file'].apply(normalize_filename)
    print(f"   Keypoints: {len(kp):,} rows")
    lb = pd.read_csv(config['labels_path'])
    lb = lb[lb['label'].isin(['safe', 'risky'])].copy()
    lb['file_norm'] = lb['file'].apply(normalize_filename)
    print(f"   Labels: {len(lb)}")
    sg = pd.read_csv(config['segments_path'])
    sg['source_norm'] = sg['source_file'].apply(normalize_filename)
    print(f"   Segments: {len(sg)}")
    for label, count in lb['label'].value_counts().items():
        print(f"     {label}: {count} ({count/len(lb)*100:.1f}%)")
    return kp, lb, sg


def extract_sequences(kp_df, lb_df, sg_df, config):
    
    print("\n  EXTRACTING SEQUENCES")
    seq_len = config['sequence_length']
    kp_grouped = {n: g for n, g in kp_df.groupby('file_norm')}
    sequences, labels = [], []
    skip_counts = Counter()

    for _, row in tqdm(lb_df.iterrows(), total=len(lb_df), desc="   Extract"):
        fn, rid = row['file_norm'], row['rep_id']
        label = 0 if row['label'] == 'safe' else 1
        seg = sg_df[(sg_df['source_norm'] == fn) & (sg_df['rep_id'] == rid)]
        if len(seg) == 0:
            skip_counts['no_segment'] += 1
            continue
        if fn not in kp_grouped:
            skip_counts['no_keypoints'] += 1
            continue
        sf, ef = int(seg['start_frame'].values[0]), int(seg['end_frame'].values[0])
        vk = kp_grouped[fn]
        rep_kp = vk[(vk['frame'] >= sf) & (vk['frame'] <= ef)].sort_values('frame')
        if len(rep_kp) < 10:
            skip_counts['too_short'] += 1
            continue

        
        frames_data = []
        for _, fr in rep_kp.iterrows():
            joints = []
            for kn in KEYPOINT_NAMES:
                x = fr.get(f'{kn}_x', 0)
                y = fr.get(f'{kn}_y', 0)
                x = 0 if pd.isna(x) else x
                y = 0 if pd.isna(y) else y
                conf = fr.get(f'{kn}_score', np.nan)
                if pd.isna(conf):
                    conf = fr.get(f'{kn}_conf', np.nan)
                if pd.isna(conf):
                    conf = fr.get(f'{kn}_visibility', np.nan)
                if pd.isna(conf):
                    conf = 1.0
                joints.append([x, y, float(conf)])
            frames_data.append(joints)
        frames_data = np.array(frames_data, dtype=np.float32)

       
        hip_center = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
        frames_data[:, :, :2] -= hip_center[:, np.newaxis, :]
       
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

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(f"   Extracted: {len(sequences)}, Skipped: {sum(skip_counts.values())}")
    if skip_counts:
        for reason, count in skip_counts.most_common():
            print(f"     {reason}: {count}")
    print(f"   Shape: {sequences.shape}")
    return sequences, labels


class SquatDataset(Dataset):
    """PyTorch Dataset for squat skeleton sequences."""
    def __init__(self, sequences, labels, augment=False, config=None):
        self.sequences = sequences.copy()
        self.labels = labels.copy()
        self.augment = augment
        self.config = config or {}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()  # (T, V, C)
        label = self.labels[idx]

        if self.augment:
            seq = self._apply_augmentation(seq)

       
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return seq_tensor, label_tensor

    def _apply_augmentation(self, seq):
        """Apply random augmentations during training."""
        aug = self.config.get('augmentation', {})
        rng = np.random

      
        if rng.random() < aug.get('noise_prob', 0.5):
            std = aug.get('gaussian_noise_std', 0.01)
            noise = rng.normal(0, std, seq[:, :, :2].shape).astype(np.float32)
            seq[:, :, :2] += noise

       
        if rng.random() < aug.get('scale_prob', 0.5):
            lo, hi = aug.get('scale_range', (0.9, 1.1))
            scale = rng.uniform(lo, hi)
            seq[:, :, :2] *= scale

        
        if rng.random() < aug.get('shift_prob', 0.3):
            lo, hi = aug.get('temporal_shift_range', (-3, 4))
            shift = rng.randint(lo, hi)
            if shift != 0:
                seq = np.roll(seq, shift, axis=0)

        
        if rng.random() < aug.get('horizontal_flip_prob', 0.3):
            seq[:, :, 0] = -seq[:, :, 0]
            for l, r in FLIP_PAIRS:
                seq[:, l, :], seq[:, r, :] = seq[:, r, :].copy(), seq[:, l, :].copy()

        return seq



def train_one_epoch(model, loader, criterion, optimizer, device, clip_norm=1.0,
                    scaler=None):
    """Train for one epoch with optional mixed precision (AMP)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == 'cuda'

    for sequences, labels in loader:
        sequences = sequences.to(device)  # (N, T, V, C)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(sequences)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
   
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(sequences)
                    loss = criterion(logits, labels)
            else:
                logits = model(sequences)
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = F.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return (total_loss / max(total, 1), correct / max(total, 1),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def find_optimal_threshold(y_true, y_probs, metric='f1'):
   
    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_probs >= t).astype(int)
        if metric == 'f1':
            score = compute_f1_score(y_true, preds, pos_label=1)
        elif metric == 'balanced_accuracy':
            score = compute_balanced_accuracy(y_true, preds)
        else:
            score = compute_f1_score(y_true, preds, pos_label=1)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score



def run_training_phase(model, train_loader, val_loader, config, device,
                       phase_name, num_epochs, lr, patience, freeze_backbone=False):
    
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    if freeze_backbone:
        print("   Freezing backbone layers...")
        frozen = 0
        for name, param in model.named_parameters():
            if 'cls_head' not in name:
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Frozen: {frozen} params, Trainable: {trainable:,}")
    else:
        print("   Unfreezing all layers...")
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingScheduler(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    ls = config.get('label_smoothing', 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    # Mixed precision scaler for GPU memory efficiency
    use_amp = config.get('use_mixed_precision', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_f1, best_epoch, wait = 0.0, 0, 0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_f1r': [], 'val_bacc': [], 'lr': []}

    print(f"   LR: {lr}, Epochs: {num_epochs}, Patience: {patience}")
    print(f"   Label smoothing: {ls}")
    if use_amp:
        print(f"   Mixed precision (AMP): enabled")
    print()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            clip_norm=config.get('gradient_clip_norm', 1.0),
            scaler=scaler,
        )
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        f1r = compute_f1_score(val_labels, val_preds, pos_label=1)
        bacc = compute_balanced_accuracy(val_labels, val_preds)
        cur_lr = scheduler.get_lr()
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1r'].append(f1r)
        history['val_bacc'].append(bacc)
        history['lr'].append(cur_lr)

        elapsed = time.time() - t0
        marker = ""
        if f1r > best_f1:
            best_f1 = f1r
            best_epoch = epoch
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            wait += 1

        if epoch <= 5 or epoch % 5 == 0 or marker or epoch == num_epochs:
            print(f"   Ep {epoch:3d}/{num_epochs} | "
                  f"TrL={train_loss:.4f} TrA={train_acc:.3f} | "
                  f"VL={val_loss:.4f} VA={val_acc:.3f} | "
                  f"F1r={f1r:.3f} BA={bacc:.3f} | "
                  f"LR={cur_lr:.6f} | {elapsed:.1f}s{marker}")

        if wait >= patience:
            print(f"\n   Early stopping at epoch {epoch} (best F1r={best_f1:.3f} @ ep {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"   Restored best weights from epoch {best_epoch}")

    return history, best_f1



def comprehensive_evaluation(model, test_loader, device, config):
    """Full evaluation with metrics, threshold search, confidence analysis."""
    print(f"\n{'='*60}")
    print("  COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    _, test_acc, preds, labels, probs = evaluate(model, test_loader, criterion, device)

    # Basic metrics at threshold 0.5
    print("\n   --- Default Threshold (0.5) ---")
    f1r = compute_f1_score(labels, preds, pos_label=1)
    f1s = compute_f1_score(labels, preds, pos_label=0)
    pr = compute_precision(labels, preds, pos_label=1)
    rr = compute_recall(labels, preds, pos_label=1)
    bacc = compute_balanced_accuracy(labels, preds)
    mcc = compute_matthews_corrcoef(labels, preds)
    auc = compute_roc_auc(labels, probs)
    ap = compute_average_precision(labels, probs)
    spec = compute_specificity(labels, preds)
    kappa = compute_cohens_kappa(labels, preds)
    g_mean = compute_g_mean(labels, preds)
    f1_macro = (f1r + f1s) / 2
    brier = compute_brier_score(labels, probs) if probs is not None else None
    logloss = compute_log_loss(labels, probs) if probs is not None else None

    print(f"   Accuracy:          {test_acc:.4f}")
    print(f"   Balanced Accuracy: {bacc:.4f}")
    print(f"   F1 (risky):        {f1r:.4f}")
    print(f"   F1 (safe):         {f1s:.4f}")
    print(f"   Precision (risky): {pr:.4f}")
    print(f"   Recall (risky):    {rr:.4f}")
    print(f"   MCC:               {mcc:.4f}")
    print(f"   ROC-AUC:           {auc:.4f}")
    print(f"   Average Precision: {ap:.4f}")

    cm = compute_confusion_matrix(labels, preds)
    print(f"\n   Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Safe  Risky")
    print(f"   Actual Safe  [{cm[0,0]:5d} {cm[0,1]:5d}]")
    print(f"   Actual Risky [{cm[1,0]:5d} {cm[1,1]:5d}]")

    report = format_classification_report(labels, preds, target_names=['safe', 'risky'])
    print(report)
    print(f"   Specificity:       {spec:.4f}")
    print(f"   Cohen's Kappa:     {kappa:.4f}")
    print(f"   G-Mean:            {g_mean:.4f}")
    print(f"   F1 (macro):        {f1_macro:.4f}")
    if brier is not None:
        print(f"   Brier Score:       {brier:.4f}  (lower is better)")
        print(f"   Log Loss:          {logloss:.4f}  (lower is better)")

    # Bootstrap CIs
    print(f"\n   --- 95% Bootstrap Confidence Intervals (n=2000) ---")
    ci_f1r = bootstrap_ci(labels, preds, probs,
                          lambda yt, yp, ypr: compute_f1_score(yt, yp, pos_label=1))
    ci_mcc = bootstrap_ci(labels, preds, probs,
                          lambda yt, yp, ypr: compute_matthews_corrcoef(yt, yp))
    ci_bacc = bootstrap_ci(labels, preds, probs,
                           lambda yt, yp, ypr: compute_balanced_accuracy(yt, yp))
    ci_auc = bootstrap_ci(labels, preds, probs,
                          lambda yt, yp, ypr: compute_roc_auc(yt, ypr)) if probs is not None else (None, None)

    print(f"   F1 (risky):        {f1r:.4f}  [{ci_f1r[0]:.4f}, {ci_f1r[1]:.4f}]")
    print(f"   MCC:               {mcc:.4f}  [{ci_mcc[0]:.4f}, {ci_mcc[1]:.4f}]")
    print(f"   Balanced Accuracy: {bacc:.4f}  [{ci_bacc[0]:.4f}, {ci_bacc[1]:.4f}]")
    if ci_auc[0] is not None:
        print(f"   ROC-AUC:           {auc:.4f}  [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
    results = {
        'accuracy': float(test_acc),
        'balanced_accuracy': float(bacc),
        'f1_risky': float(f1r),
        'f1_safe': float(f1s),
        'precision_risky': float(pr),
        'recall_risky': float(rr),
        'mcc': float(mcc),
        'roc_auc': float(auc),
        'pr_auc': float(ap),
        'confusion_matrix': cm.tolist(),
        'specificity': float(spec),
        'cohens_kappa': float(kappa),
        'g_mean': float(g_mean),
        'f1_macro': float(f1_macro),
        'brier_score': float(brier) if brier is not None else None,
        'log_loss': float(logloss) if logloss is not None else None,
        'ci_f1_risky_95': list(ci_f1r),
        'ci_mcc_95': list(ci_mcc),
        'ci_balanced_accuracy_95': list(ci_bacc),
        'ci_roc_auc_95': list(ci_auc) if ci_auc[0] is not None else None,
    }

    # Threshold search
    if config.get('threshold_search', True):
        print("\n   --- Optimal Threshold Search ---")
        opt_thresh, opt_f1 = find_optimal_threshold(labels, probs, metric='f1')
        opt_preds = (probs >= opt_thresh).astype(int)
        opt_pr = compute_precision(labels, opt_preds, pos_label=1)
        opt_rr = compute_recall(labels, opt_preds, pos_label=1)
        opt_bacc = compute_balanced_accuracy(labels, opt_preds)
        opt_mcc = compute_matthews_corrcoef(labels, opt_preds)
        print(f"   Optimal threshold:   {opt_thresh:.2f}")
        print(f"   F1 (risky) @ opt:    {opt_f1:.4f}")
        print(f"   Precision @ opt:     {opt_pr:.4f}")
        print(f"   Recall @ opt:        {opt_rr:.4f}")
        print(f"   Balanced Acc @ opt:  {opt_bacc:.4f}")
        print(f"   MCC @ opt:           {opt_mcc:.4f}")

        opt_cm = compute_confusion_matrix(labels, opt_preds)
        print(f"\n   Confusion Matrix @ optimal threshold:")
        print(f"                  Predicted")
        print(f"                  Safe  Risky")
        print(f"   Actual Safe  [{opt_cm[0,0]:5d} {opt_cm[0,1]:5d}]")
        print(f"   Actual Risky [{opt_cm[1,0]:5d} {opt_cm[1,1]:5d}]")

        results['optimal_threshold'] = float(opt_thresh)
        results['optimal_f1_risky'] = float(opt_f1)
        results['optimal_precision'] = float(opt_pr)
        results['optimal_recall'] = float(opt_rr)
        results['optimal_balanced_accuracy'] = float(opt_bacc)
        results['optimal_mcc'] = float(opt_mcc)

    # Confidence analysis
    if config.get('confidence_analysis', True):
        print("\n   --- Confidence Analysis ---")
        safe_probs = probs[labels == 0]
        risky_probs = probs[labels == 1]
        print(f"   Safe samples prob(risky):  mean={safe_probs.mean():.3f}, "
              f"std={safe_probs.std():.3f}, median={np.median(safe_probs):.3f}")
        print(f"   Risky samples prob(risky): mean={risky_probs.mean():.3f}, "
              f"std={risky_probs.std():.3f}, median={np.median(risky_probs):.3f}")

        # Confidence buckets
        for lo, hi, desc in [(0.0, 0.3, "Low"), (0.3, 0.5, "Med-Low"),
                              (0.5, 0.7, "Med-High"), (0.7, 1.0, "High")]:
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bucket_acc = np.mean(((probs[mask] >= 0.5).astype(int)) == labels[mask])
                print(f"   {desc:>8s} ({lo:.1f}-{hi:.1f}): "
                      f"n={mask.sum():4d}, acc={bucket_acc:.3f}")

    # Error analysis
    if config.get('error_analysis', True):
        print("\n   --- Error Analysis ---")
        fp_mask = (preds == 1) & (labels == 0)
        fn_mask = (preds == 0) & (labels == 1)
        print(f"   False Positives (safe predicted risky): {fp_mask.sum()}")
        if fp_mask.sum() > 0:
            print(f"     Avg prob(risky): {probs[fp_mask].mean():.3f}")
        print(f"   False Negatives (risky predicted safe): {fn_mask.sum()}")
        if fn_mask.sum() > 0:
            print(f"     Avg prob(risky): {probs[fn_mask].mean():.3f}")

    return results


def main():
    start_time = time.time()
    print("=" * 60)
    print("  PRETRAINED MS-G3D FOR SQUAT FORM ASSESSMENT")
    print("  Transfer Learning from NTU RGB+D 120 (HRNet COCO 17-Joint)")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Seed
    seed = CONFIG['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    kp_df, lb_df, sg_df = load_data(CONFIG)
    sequences, labels = extract_sequences(kp_df, lb_df, sg_df, CONFIG)

    # Split
    print(f"\n  STRATIFIED SPLIT (test_size={CONFIG['test_size']})")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        sequences, labels, test_size=CONFIG['test_size'], random_state=seed
    )
    print(f"   Train: {len(X_train)} (Safe={int(np.sum(y_train==0))}, "
          f"Risky={int(np.sum(y_train==1))})")
    print(f"   Test:  {len(X_test)} (Safe={int(np.sum(y_test==0))}, "
          f"Risky={int(np.sum(y_test==1))})")

    # SMOTE on training set
    X_train, y_train = apply_smote(X_train, y_train, CONFIG)

    # Datasets and loaders
    train_ds = SquatDataset(X_train, y_train, augment=True, config=CONFIG)
    test_ds = SquatDataset(X_test, y_test, augment=False, config=CONFIG)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'],
                             shuffle=False, num_workers=0)

    # Build model
    print(f"\n{'='*60}")
    print("  BUILDING MS-G3D MODEL")
    print(f"{'='*60}")
    model = PYSKL_MSG3D(
        in_ch=CONFIG['in_channels'],
        num_joints=CONFIG['num_joints'],
        num_person=CONFIG['num_person'],
        num_classes=2,
        num_subsets=CONFIG['num_sgcn_subsets'],
        num_g3d_subsets=CONFIG['num_g3d_subsets'],
        windows=tuple(CONFIG['g3d_windows']),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Print model key count for verification
    model_keys = list(model.state_dict().keys())
    print(f"   Model state_dict keys: {len(model_keys)}")

    # Download and load pretrained weights
    ckpt_path = download_checkpoint(CONFIG)
    if ckpt_path:
        loaded = load_pretrained_weights(model, ckpt_path)
        if loaded == 0:
            print("\n   WARNING: No pretrained weights loaded!")
            print("   Falling back to training from scratch.")
    else:
        print("   No checkpoint available. Training from scratch.")

    # Phase 1: Freeze backbone, train head
    h1, best_f1_p1 = run_training_phase(
        model, train_loader, test_loader, CONFIG, device,
        phase_name="PHASE 1: CLASSIFIER HEAD TRAINING (backbone frozen)",
        num_epochs=CONFIG['phase1_epochs'],
        lr=CONFIG['phase1_lr'],
        patience=CONFIG['phase1_patience'],
        freeze_backbone=True,
    )

    # Clear GPU memory before Phase 2
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Rebuild loaders with smaller batch size for Phase 2 (full backprop needs more VRAM)
    p2_bs = CONFIG.get('phase2_batch_size', CONFIG['batch_size'])
    if p2_bs != CONFIG['batch_size']:
        print(f"\n   Reducing batch size for Phase 2: {CONFIG['batch_size']} -> {p2_bs}")
    train_loader_p2 = DataLoader(train_ds, batch_size=p2_bs,
                                  shuffle=True, num_workers=0, drop_last=False)
    test_loader_p2 = DataLoader(test_ds, batch_size=p2_bs,
                                 shuffle=False, num_workers=0)

    # Phase 2: Fine-tune all
    h2, best_f1_p2 = run_training_phase(
        model, train_loader_p2, test_loader_p2, CONFIG, device,
        phase_name="PHASE 2: END-TO-END FINE-TUNING",
        num_epochs=CONFIG['phase2_epochs'],
        lr=CONFIG['phase2_lr'],
        patience=CONFIG['phase2_patience'],
        freeze_backbone=False,
    )

    # Comprehensive evaluation
    results = comprehensive_evaluation(model, test_loader, device, CONFIG)

    # Save model
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'results': results,
    }, CONFIG['model_save_path'])
    print(f"\n   Model saved: {CONFIG['model_save_path']}")

  
    results['phase1_best_f1r'] = float(best_f1_p1)
    results['phase2_best_f1r'] = float(best_f1_p2)
    results['total_time_seconds'] = time.time() - start_time
    results['model_params'] = int(total_params)
    results['architecture'] = 'MS-G3D (pretrained PYSKL NTU120 XSub HRNet)'
    results['checkpoint_url'] = CONFIG['checkpoint_url']
    with open(CONFIG['results_save_path'], 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved: {CONFIG['results_save_path']}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best F1(risky) Phase 1: {best_f1_p1:.4f}")
    print(f"  Best F1(risky) Phase 2: {best_f1_p2:.4f}")
    print(f"  Final F1(risky):        {results['f1_risky']:.4f}")
    if 'optimal_f1_risky' in results:
        print(f"  Optimal F1(risky):      {results['optimal_f1_risky']:.4f} "
              f"(threshold={results['optimal_threshold']:.2f})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()