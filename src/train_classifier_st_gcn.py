#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import subprocess
import time
import json
from datetime import datetime
from collections import Counter, defaultdict

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
    'model_save_path': 'models/stgcn_pretrained_finetuned.pt',
    'results_save_path': 'models/stgcn_pretrained_results.json',
    'checkpoint_url': (
        'http://download.openmmlab.com/mmaction/pyskl/ckpt/'
        'stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth'
    ),
    'checkpoint_local': 'models/stgcnpp_ntu120_xsub_hrnet_j.pth',
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
    'phase1_epochs': 40,
    'phase1_lr': 0.001,
    'phase1_patience': 15,
    'phase2_epochs': 30,
    'phase2_lr': 0.0001,
    'phase2_patience': 12,
    'batch_size': 32,
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
   
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    pp = np.sum(y_pred == pos_label)
    if pp == 0:
        return 0.0
    return tp / pp


def compute_recall(y_true, y_pred, pos_label=1):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    ap = np.sum(y_true == pos_label)
    if ap == 0:
        return 0.0
    return tp / ap


def compute_f1_score(y_true, y_pred, pos_label=1):
    
    p = compute_precision(y_true, y_pred, pos_label)
    r = compute_recall(y_true, y_pred, pos_label)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_balanced_accuracy(y_true, y_pred):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for cls in np.unique(y_true):
        mask = (y_true == cls)
        if mask.sum() == 0:
            continue
        recalls.append(np.sum(y_pred[mask] == cls) / mask.sum())
    return np.mean(recalls)


def compute_accuracy(y_true, y_pred):
  
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def compute_matthews_corrcoef(y_true, y_pred):
    
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return num / den


def compute_roc_auc(y_true, y_scores, num_thresholds=1000):
   
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)
    if total_pos == 0 or total_neg == 0:
        return 0.5

    thresholds = np.linspace(y_scores.min() - 0.001, y_scores.max() + 0.001, num_thresholds)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

   
    idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[idx]
    tpr_sorted = tpr_arr[idx]

 
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        dx = fpr_sorted[i] - fpr_sorted[i - 1]
        avg_y = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        auc += dx * avg_y

    return auc


def compute_average_precision(y_true, y_scores, num_thresholds=1000):
    
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    thresholds = np.linspace(y_scores.max() + 0.001, y_scores.min() - 0.001, num_thresholds)
    precisions = []
    recalls = []

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    idx = np.argsort(recalls)
    rec_sorted = recalls[idx]
    prec_sorted = precisions[idx]

    ap = 0.0
    for i in range(1, len(rec_sorted)):
        dr = rec_sorted[i] - rec_sorted[i - 1]
        ap += dr * prec_sorted[i]

    return ap


def format_classification_report(y_true, y_pred, target_names=None, digits=3):
   
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(np.unique(y_true))
    if target_names is None:
        target_names = [str(c) for c in classes]

    rows = []
    total = len(y_true)
    max_name_len = max(len(n) for n in target_names)
    w = max(max_name_len + 2, 14)

    for cls, name in zip(classes, target_names):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sup = int(np.sum(y_true == cls))
        rows.append((name, prec, rec, f1, sup))

    macro_p = np.mean([r[1] for r in rows])
    macro_r = np.mean([r[2] for r in rows])
    macro_f1 = np.mean([r[3] for r in rows])
    wt_p = sum(r[1] * r[4] for r in rows) / total
    wt_r = sum(r[2] * r[4] for r in rows) / total
    wt_f1 = sum(r[3] * r[4] for r in rows) / total
    acc = np.mean(y_true == y_pred)

    header = (f"{'':>{w}s}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}")
    lines = ["\n" + header + "\n"]
    for name, p, r, f, s in rows:
        lines.append(f"{name:>{w}s}{p:>12.{digits}f}{r:>12.{digits}f}{f:>12.{digits}f}{s:>12d}")
    lines.append("")
    lines.append(f"{'accuracy':>{w}s}{'':>12}{'':>12}{acc:>12.{digits}f}{total:>12d}")
    lines.append(f"{'macro avg':>{w}s}{macro_p:>12.{digits}f}{macro_r:>12.{digits}f}{macro_f1:>12.{digits}f}{total:>12d}")
    lines.append(f"{'weighted avg':>{w}s}{wt_p:>12.{digits}f}{wt_r:>12.{digits}f}{wt_f1:>12.{digits}f}{total:>12d}")

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
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        n_test = min(n_test, len(cls_idx) - 1)
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_euclidean_distances(point, dataset):

    diff = dataset - point[np.newaxis, :]
    return np.sqrt(np.sum(diff ** 2, axis=1))


def find_k_nearest_neighbors(point_idx, dataset, k):

    distances = compute_euclidean_distances(dataset[point_idx], dataset)
    distances[point_idx] = np.inf  # exclude self
    return np.argpartition(distances, k)[:k].tolist()


def generate_synthetic_sample(sample, neighbor, rng):
    
    lam = rng.uniform(0.0, 1.0)
    return sample + lam * (neighbor - sample)


def apply_smote(sequences, labels, config):
   
    if not config.get('use_smote', False):
        print("   SMOTE: Disabled")
        return sequences, labels

    print("\n   Applying SMOTE Oversampling...")

    rng = np.random.RandomState(config['random_seed'])

    safe_count = int(np.sum(labels == 0))
    risky_count = int(np.sum(labels == 1))
    print(f"      Before: Safe={safe_count}, Risky={risky_count}")
    print(f"      Ratio: {safe_count / max(risky_count, 1):.1f}:1")

 
    sampling_strategy = config.get('smote_sampling_strategy', 0.75)
    target_risky = int(safe_count * sampling_strategy)
    n_synthetic = max(0, target_risky - risky_count)

    if n_synthetic == 0:
        print("      No synthetic samples needed")
        return sequences, labels

    print(f"      Target risky count: {target_risky}")
    print(f"      Synthetic samples to generate: {n_synthetic}")

 
    original_shape = sequences.shape[1:]  # (64, 17, 3)
    flat_all = sequences.reshape(len(sequences), -1)

   
    minority_indices = np.where(labels == 1)[0]
    minority_flat = flat_all[minority_indices]
    n_minority = len(minority_indices)

 
    k = min(config.get('smote_k_neighbors', 5), n_minority - 1)
    k = max(1, k)
    print(f"      k-NN neighbors: {k}")


    print(f"      Computing {n_minority} x {n_minority} distance matrix...")
    all_neighbors = []
    for i in range(n_minority):
        neighbors = find_k_nearest_neighbors(i, minority_flat, k)
        all_neighbors.append(neighbors)

   
    synthetic_samples = []
    for i in range(n_synthetic):
       
        source_idx = rng.randint(0, n_minority)
        source_sample = minority_flat[source_idx]

      
        neighbor_local_idx = rng.choice(all_neighbors[source_idx])
        neighbor_sample = minority_flat[neighbor_local_idx]

        # Generate synthetic sample by interpolation
        synthetic = generate_synthetic_sample(source_sample, neighbor_sample, rng)
        synthetic_samples.append(synthetic)

    synthetic_samples = np.array(synthetic_samples)

    # Reshape synthetic samples back to sequence format
    synthetic_sequences = synthetic_samples.reshape(-1, *original_shape)
    synthetic_labels = np.ones(n_synthetic, dtype=labels.dtype)

    # Concatenate originals with synthetic samples
    augmented_sequences = np.concatenate([sequences, synthetic_sequences], axis=0)
    augmented_labels = np.concatenate([labels, synthetic_labels], axis=0)

    # Shuffle the augmented dataset
    shuffle_idx = rng.permutation(len(augmented_sequences))
    augmented_sequences = augmented_sequences[shuffle_idx]
    augmented_labels = augmented_labels[shuffle_idx]

    new_safe = int(np.sum(augmented_labels == 0))
    new_risky = int(np.sum(augmented_labels == 1))
    print(f"      Synthetic samples generated: {n_synthetic}")
    print(f"      After: Safe={new_safe}, Risky={new_risky}")
    print(f"      New ratio: {new_safe / max(new_risky, 1):.1f}:1")

    return augmented_sequences, augmented_labels



class CosineAnnealingScheduler:
    
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.current_step = 0

        # Store initial learning rates for each parameter group
        self.initial_lrs = [
            group['lr'] for group in optimizer.param_groups
        ]

    def step(self):
        
        self.current_step += 1

        for param_group, initial_lr in zip(
            self.optimizer.param_groups, self.initial_lrs
        ):
            # Cosine annealing formula
            cosine_factor = math.cos(
                math.pi * self.current_step / self.T_max
            )
            new_lr = (
                self.eta_min
                + 0.5 * (initial_lr - self.eta_min) * (1 + cosine_factor)
            )
            param_group['lr'] = new_lr

    def get_lr(self):
        
        return self.optimizer.param_groups[0]['lr']



def build_coco_adjacency():
   
    num_joints = 17
    center_joint = 0  # Nose as root (PYSKL default for COCO)

    # Build binary adjacency matrix from skeleton edges
    adjacency = np.zeros((num_joints, num_joints))
    for i, j in COCO_EDGES:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    # Compute shortest-path hop distances via BFS from each joint
    hop_distance = np.full((num_joints, num_joints), np.inf)
    for start in range(num_joints):
        hop_distance[start, start] = 0
        queue = [start]
        visited = {start}
        while queue:
            current = queue.pop(0)
            for neighbor in range(num_joints):
                if adjacency[current, neighbor] > 0 and neighbor not in visited:
                    hop_distance[start, neighbor] = hop_distance[start, current] + 1
                    visited.add(neighbor)
                    queue.append(neighbor)

    # Partition into 3 subsets based on hop distance from center
    A = np.zeros((3, num_joints, num_joints), dtype=np.float32)
    for i in range(num_joints):
        for j in range(num_joints):
            if hop_distance[i, j] == 0:
                A[0, i, j] = 1  # Self-connection
            elif hop_distance[i, j] == 1:
                if hop_distance[j, center_joint] >= hop_distance[i, center_joint]:
                    A[1, i, j] = 1  # Centripetal
                else:
                    A[2, i, j] = 1  # Centrifugal

    # Column normalization for each subset
    for k in range(3):
        col_sum = A[k].sum(axis=0)
        col_sum[col_sum == 0] = 1
        A[k] = A[k] / col_sum[np.newaxis, :]

    return A


def get_branch_channels(out_channels):
   
    base = out_channels // 6
    remainder = out_channels % 6
    return [base + remainder] + [base] * 5


class TemporalConv(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=1,
        )

    def forward(self, x):
        return self.conv(x)


class PyskBlock(nn.Module):
   
    def __init__(self, in_ch, out_ch, A, stride=1):
        super().__init__()
        num_subsets = A.shape[0]
        self._in_ch = in_ch
        self._out_ch = out_ch
        self._stride = stride
        self._num_subsets = num_subsets

        # GCN
        self.gcn = nn.Module()
        self.gcn.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.gcn.conv = nn.Conv2d(in_ch, out_ch * num_subsets, 1)
        self.gcn.bn = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.gcn.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
            )

        # TCN
        self.tcn = nn.Module()
        self.tcn.transform = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
        )

        br_chs = get_branch_channels(out_ch)
        branches = nn.ModuleList()

        # Branches 0-3: bottleneck + temporal conv with dilations [1,2,3,4]
        for b in range(4):
            mid = br_chs[b]
            d = b + 1
            pad = d
            branches.append(nn.Sequential(
                nn.Conv2d(out_ch, mid, 1),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                TemporalConv(mid, mid, 3, stride, pad, d),
            ))

        # Branch 4: bottleneck only (no temporal conv)
        branches.append(nn.Sequential(
            nn.Conv2d(out_ch, br_chs[4], 1),
            nn.BatchNorm2d(br_chs[4]),
        ))

        # Branch 5: maxpool path (simple conv)
        branches.append(nn.Conv2d(out_ch, br_chs[5], 1))

        self.tcn.branches = branches
        self.tcn.bn = nn.BatchNorm2d(out_ch)

        # Block residual
        if in_ch != out_ch:
            self.residual = nn.Module()
            self.residual.conv = nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1))
            self.residual.bn = nn.BatchNorm2d(out_ch)
        self._has_residual = (in_ch != out_ch)

    def forward(self, x):
        residual = x
        N, C, T, V = x.shape

        # Spatial graph convolution
        y = self.gcn.conv(x)
        oc = y.shape[1] // self._num_subsets
        graph_out = 0
        for k in range(self._num_subsets):
            subset = y[:, k * oc:(k + 1) * oc, :, :]
            graph_out = graph_out + torch.einsum('nctv,vw->nctw', subset, self.gcn.A[k])
        graph_out = self.gcn.bn(graph_out)

        if hasattr(self.gcn, 'down'):
            graph_out = graph_out + self.gcn.down(x)
        elif C == oc:
            graph_out = graph_out + x

        x = F.relu(graph_out, inplace=True)

        # Multi-scale temporal convolution
        z = self.tcn.transform(x)
        branch_outs = []
        for i, branch in enumerate(self.tcn.branches):
            if i < 4:
                branch_outs.append(branch(z))
            elif i == 4:
                b4 = branch(z)
                if self._stride > 1:
                    b4 = b4[:, :, ::self._stride, :]
                branch_outs.append(b4)
            else:
                pooled = F.max_pool2d(z, (3, 1), (self._stride, 1), (1, 0))
                branch_outs.append(branch(pooled))

        x = torch.cat(branch_outs, dim=1)
        x = self.tcn.bn(x)

        # Block residual connection
        if self._has_residual:
            residual = self.residual.bn(self.residual.conv(residual))
            x = x + residual
        elif residual.shape == x.shape:
            x = x + residual

        return F.relu(x, inplace=True)


class PYSKL_STGCNPP(nn.Module):
    
    def __init__(self, in_ch=3, nj=17, nc=2):
        super().__init__()
        A = build_coco_adjacency()
        self.data_bn = nn.BatchNorm1d(in_ch * nj)
        channels = [in_ch, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        self.gcn = nn.ModuleList()
        for i in range(10):
            stride = 2 if i in [4, 7] else 1
            self.gcn.append(PyskBlock(channels[i], channels[i + 1], A, stride))
        self.cls_head = nn.Module()
        self.cls_head.fc_cls = nn.Linear(256, nc)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        N, M, T, V, C = x.shape

        # Data batch normalization
        x = x.permute(0, 1, 3, 4, 2).reshape(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.reshape(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # ST-GCN blocks
        for block in self.gcn:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=-1).mean(dim=-1)
        x = x.reshape(N, M, -1).mean(dim=1)

        return self.cls_head.fc_cls(x)



def download_checkpoint(config):
    """Download pretrained checkpoint if not present."""
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

    loaded = 0
    skipped_head = 0
    skipped_shape = 0
    skipped_missing = 0

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
                if skipped_shape <= 3:
                    print(f"   SHAPE: {mk} ckpt={cv.shape} model={model_sd[mk].shape}")
        else:
            skipped_missing += 1
            if skipped_missing <= 3:
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
    """Normalize filenames for cross-file matching."""
    if pd.isna(filename):
        return None
    name = str(filename)
    if name.lower().startswith('pose_'):
        name = name[5:]
    return os.path.splitext(name)[0].lower()


def load_data(config):
    """Load keypoints, labels, and segment boundaries."""
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
        pct = count / len(lb) * 100
        print(f"     {label}: {count} ({pct:.1f}%)")

    return kp, lb, sg


def extract_sequences(kp_df, lb_df, sg_df, config):
    
    print("\n  EXTRACTING SEQUENCES")
    seq_len = config['sequence_length']
    kp_grouped = {name: group for name, group in kp_df.groupby('file_norm')}

    sequences = []
    labels = []
    skip_counts = Counter()

    for _, row in tqdm(lb_df.iterrows(), total=len(lb_df), desc="   Extract"):
        fn = row['file_norm']
        rid = row['rep_id']
        label = 0 if row['label'] == 'safe' else 1

        # Find segment
        seg = sg_df[(sg_df['source_norm'] == fn) & (sg_df['rep_id'] == rid)]
        if len(seg) == 0:
            skip_counts['no_segment'] += 1
            continue
        if fn not in kp_grouped:
            skip_counts['no_keypoints'] += 1
            continue

        sf = int(seg['start_frame'].values[0])
        ef = int(seg['end_frame'].values[0])

        # Extract frame keypoints
        vk = kp_grouped[fn]
        rep_kp = vk[(vk['frame'] >= sf) & (vk['frame'] <= ef)].sort_values('frame')
        if len(rep_kp) < 10:
            skip_counts['too_short'] += 1
            continue

        # Build (frames, 17, 3) array
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

        # Normalize: center on hip midpoint
        hip_center = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
        frames_data[:, :, :2] -= hip_center[:, np.newaxis, :]

        # Normalize: scale by torso length
        shoulder_center = (frames_data[:, 5, :2] + frames_data[:, 6, :2]) / 2
        hip_center_2 = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
        torso_len = np.linalg.norm(shoulder_center - hip_center_2, axis=1).mean()
        if torso_len > 1e-6:
            frames_data[:, :, :2] /= torso_len

        # Resample to fixed length via linear interpolation
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
    print(f"   Safe={np.sum(labels == 0)} Risky={np.sum(labels == 1)}")

    return sequences, labels



class SquatDataset(Dataset):
 
    def __init__(self, sequences, labels, augment=False, config=None):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.aug_cfg = config.get('augmentation', {}) if config else {}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()

        if self.augment:
            # Gaussian noise on coordinates
            if np.random.rand() < self.aug_cfg.get('noise_prob', 0.5):
                std = self.aug_cfg.get('gaussian_noise_std', 0.01)
                noise = np.random.randn(seq.shape[0], seq.shape[1], 2).astype(np.float32) * std
                seq[:, :, :2] += noise

            # Random scaling
            if np.random.rand() < self.aug_cfg.get('scale_prob', 0.5):
                lo, hi = self.aug_cfg.get('scale_range', (0.9, 1.1))
                seq[:, :, :2] *= np.random.uniform(lo, hi)

            # Temporal shift
            if np.random.rand() < self.aug_cfg.get('shift_prob', 0.3):
                lo, hi = self.aug_cfg.get('temporal_shift_range', (-3, 4))
                seq = np.roll(seq, np.random.randint(lo, hi), axis=0)

            # Horizontal flip
            if np.random.rand() < self.aug_cfg.get('horizontal_flip_prob', 0.3):
                seq = seq.copy()
                seq[:, :, 0] = -seq[:, :, 0]
                for l, r in FLIP_PAIRS:
                    seq[:, l, :], seq[:, r, :] = seq[:, r, :].copy(), seq[:, l, :].copy()

        # Add person dimension: (64, 17, 3) -> (1, 64, 17, 3)
        return torch.FloatTensor(seq[np.newaxis, ...]), self.labels[idx]



class LabelSmoothingLoss(nn.Module):
    
    def __init__(self, num_classes=2, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.class_weight = weight

    def forward(self, predictions, targets):
        log_probs = F.log_softmax(predictions, dim=1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=1)

        if self.class_weight is not None:
            loss = loss * self.class_weight[targets]

        return loss.mean()



def freeze_backbone(model):
    
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if 'cls_head' in name:
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1
    print(f"   Frozen={frozen} Trainable={trainable}")
    return frozen, trainable


def unfreeze_all(model):
   
    total = 0
    for param in model.parameters():
        param.requires_grad = True
        total += param.numel()
    print(f"   All unfrozen: {total:,} params")
    return total


def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_norm=1.0):
 
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seqs, labs in dataloader:
        seqs, labs = seqs.to(device), labs.to(device)

        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        correct += outputs.argmax(1).eq(labs).sum().item()
        total += labs.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate_model(model, dataloader, device):
   
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for seqs, labs in dataloader:
            outputs = model(seqs.to(device))
            probs = F.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labs.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def print_detailed_results(preds, probs, labels, name=""):
    """Print comprehensive evaluation using custom metrics."""
    cm = compute_confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    f1s = compute_f1_score(labels, preds, pos_label=0)
    f1r = compute_f1_score(labels, preds, pos_label=1)
    prec_r = compute_precision(labels, preds, pos_label=1)
    rec_r = compute_recall(labels, preds, pos_label=1)
    ba = compute_balanced_accuracy(labels, preds)
    auc = compute_roc_auc(labels, probs)
    ap = compute_average_precision(labels, probs)
    mcc = compute_matthews_corrcoef(labels, preds)
    acc = compute_accuracy(labels, preds)
    spec = compute_specificity(labels, preds)
    kappa = compute_cohens_kappa(labels, preds)
    g_mean = compute_g_mean(labels, preds)
    f1_macro = (f1r + f1s) / 2
    brier = compute_brier_score(labels, probs) if probs is not None else None
    logloss = compute_log_loss(labels, probs) if probs is not None else None

    print(f"\n   {name}")
    print(f"   CM: safe[{tn},{fp}] risky[{fn},{tp}]")
    print(format_classification_report(labels, preds, target_names=['safe', 'risky'], digits=3))
    print(f"   F1s={f1s:.3f} F1r={f1r:.3f} BA={ba:.3f} AUC={auc:.3f}")
    print(f"   Precision(risky)={prec_r:.3f} Recall(risky)={rec_r:.3f}")
    print(f"   AP={ap:.3f} MCC={mcc:.3f} Acc={acc:.3f}")
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
    print(f"   Balanced Accuracy: {ba:.4f}  [{ci_bacc[0]:.4f}, {ci_bacc[1]:.4f}]")
    if ci_auc[0] is not None:
        print(f"   ROC-AUC:           {auc:.4f}  [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
    return {
    'f1_safe': f1s, 'f1_risky': f1r,
    'precision_risky': prec_r, 'recall_risky': rec_r,
    'balanced_accuracy': ba, 'roc_auc': auc,
    'pr_auc': ap, 'mcc': mcc, 'accuracy': acc,
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


def analyze_confidence_distribution(probs, labels):
    
    print(f"\n   Confidence Distribution Analysis")
    print(f"   {'─' * 50}")

    preds = (probs >= 0.5).astype(int)
    correct_mask = preds == labels

    safe_probs = probs[labels == 0]
    risky_probs = probs[labels == 1]

    print(f"\n   True Safe (should be LOW prob):")
    print(f"     Mean={safe_probs.mean():.3f} Median={np.median(safe_probs):.3f} Std={safe_probs.std():.3f}")
    print(f"\n   True Risky (should be HIGH prob):")
    print(f"     Mean={risky_probs.mean():.3f} Median={np.median(risky_probs):.3f} Std={risky_probs.std():.3f}")

    if correct_mask.any():
        correct_conf = np.abs(probs[correct_mask] - 0.5) + 0.5
        print(f"\n   Correct ({correct_mask.sum()}): mean conf={correct_conf.mean():.3f}")
    if (~correct_mask).any():
        wrong_conf = np.abs(probs[~correct_mask] - 0.5) + 0.5
        print(f"   Incorrect ({(~correct_mask).sum()}): mean conf={wrong_conf.mean():.3f}")

    # Bucket distribution
    print(f"\n   Probability buckets:")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (probs >= lo) & (probs < hi)
        cnt = mask.sum()
        s = (labels[mask] == 0).sum() if mask.any() else 0
        r = (labels[mask] == 1).sum() if mask.any() else 0
        bar = '█' * (cnt // 2)
        print(f"     [{lo:.1f}-{hi:.1f}): {cnt:4d} (safe={s}, risky={r}) {bar}")


def search_optimal_threshold(probs, labels):
    
    print(f"\n   Threshold Optimization")
    print(f"   {'Thresh':>8} {'F1r':>8} {'F1s':>8} {'Prec':>8} {'Rec':>8} {'BA':>8}")

    best_thresh, best_f1r = 0.5, 0

    for thresh in np.arange(0.25, 0.80, 0.05):
        p = (probs >= thresh).astype(int)
        if len(np.unique(p)) < 2:
            continue
        f1r = compute_f1_score(labels, p, pos_label=1)
        f1s = compute_f1_score(labels, p, pos_label=0)
        prec = compute_precision(labels, p, pos_label=1)
        rec = compute_recall(labels, p, pos_label=1)
        ba = compute_balanced_accuracy(labels, p)

        marker = " <-- best" if f1r > best_f1r else ""
        print(f"   {thresh:>8.2f} {f1r:>8.3f} {f1s:>8.3f} {prec:>8.3f} {rec:>8.3f} {ba:>8.3f}{marker}")

        if f1r > best_f1r:
            best_f1r = f1r
            best_thresh = thresh

    print(f"\n   Optimal: threshold={best_thresh:.2f}, F1(risky)={best_f1r:.3f}")
    return {'optimal_threshold': best_thresh, 'f1_risky_at_optimal': best_f1r}


def analyze_errors(preds, probs, labels):
    
    print(f"\n   Error Analysis")
    print(f"   {'─' * 50}")

    # False negatives (risky missed as safe)
    fn_mask = (labels == 1) & (preds == 0)
    total_risky = (labels == 1).sum()
    print(f"\n   False Negatives (risky missed): {fn_mask.sum()}/{total_risky}")
    if fn_mask.any():
        fn_probs = probs[fn_mask]
        print(f"     Mean risky prob: {fn_probs.mean():.3f}")
        print(f"     Range: [{fn_probs.min():.3f}, {fn_probs.max():.3f}]")
        if fn_probs.mean() > 0.35:
            print(f"     Near threshold - consider lowering it")

    # False positives (safe flagged as risky)
    fp_mask = (labels == 0) & (preds == 1)
    total_safe = (labels == 0).sum()
    print(f"\n   False Positives (safe flagged): {fp_mask.sum()}/{total_safe}")
    if fp_mask.any():
        fp_probs = probs[fp_mask]
        print(f"     Mean risky prob: {fp_probs.mean():.3f}")
        print(f"     Range: [{fp_probs.min():.3f}, {fp_probs.max():.3f}]")

    total_err = fn_mask.sum() + fp_mask.sum()
    print(f"\n   Total errors: {total_err}/{len(labels)} ({total_err/len(labels)*100:.1f}%)")


def print_comparison_table(metrics):
    
    print("\n" + "=" * 75)
    prev = [
        ("LightGBM (baseline)",       0.680, 0.900, 0.775, 0.901),
        ("MLP (baseline)",            0.670, 0.890, 0.776, 0.840),
        ("Random Forest (baseline)",  0.620, 0.890, 0.729, 0.898),
        ("SVM (baseline)",            0.490, 0.870, 0.661, 0.828),
        ("ST-GCN (from scratch)",     0.300, 0.850, 0.578, 0.740),
        ("MS-G3D (from scratch)",     0.440, 0.820, 0.623, 0.707),
    ]
    print(f"   {'Model':<35} {'F1r':>6} {'F1s':>6} {'BA':>6} {'AUC':>6}")
    print("   " + "─" * 60)
    for n, r, s, b, a in prev:
        print(f"   {n:<35} {r:>6.3f} {s:>6.3f} {b:>6.3f} {a:>6.3f}")
    print("   " + "─" * 60)
    print(f"   {'STGCN++ (pretrained)':<35} {metrics['f1_risky']:>6.3f} "
          f"{metrics['f1_safe']:>6.3f} {metrics['balanced_accuracy']:>6.3f} "
          f"{metrics['roc_auc']:>6.3f}")

    if metrics['f1_risky'] >= 0.68:
        imp = metrics['f1_risky'] - 0.68
        print(f"   Exceeds LightGBM baseline by +{imp:.3f} F1(risky)")


def train_model(config):
    
    total_start = time.time()

    # Setup
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    training_history = {
        'phase1': {'loss': [], 'accuracy': [], 'f1_risky': [], 'lr': []},
        'phase2': {'loss': [], 'accuracy': [], 'f1_risky': [], 'lr': []},
    }

    # Load data
    kp, lb, sg = load_data(config)
    seqs, labs = extract_sequences(kp, lb, sg, config)

    # Stratified split (custom - no sklearn)
    print("\n  DATA SPLITTING (stratified)")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        seqs, labs,
        test_size=config.get('test_size', 0.2),
        random_state=config['random_seed'],
    )
    print(f"   Train={len(X_train)} (safe={sum(y_train==0)}, risky={sum(y_train==1)})")
    print(f"   Test={len(X_test)} (safe={sum(y_test==0)}, risky={sum(y_test==1)})")

    # SMOTE (custom - no imblearn)
    X_train_aug, y_train_aug = apply_smote(X_train, y_train, config)

    # DataLoaders
    train_dl = DataLoader(
        SquatDataset(X_train_aug, y_train_aug, augment=True, config=config),
        batch_size=config['batch_size'], shuffle=True,
    )
    test_dl = DataLoader(
        SquatDataset(X_test, y_test, augment=False, config=config),
        batch_size=config['batch_size'],
    )

    # Build model
    print("\n  BUILDING MODEL")
    model = PYSKL_STGCNPP(in_ch=config['in_channels'], nj=config['num_joints'], nc=2)
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights
    print("\n  LOADING PRETRAINED WEIGHTS")
    ckpt = download_checkpoint(config)
    n_loaded = load_pretrained_weights(model, ckpt) if ckpt else 0
    model = model.to(device)

    # Loss with class weights
    ws = len(y_train_aug) / (2 * sum(y_train_aug == 0))
    wr = len(y_train_aug) / (2 * sum(y_train_aug == 1))
    criterion = LabelSmoothingLoss(2, config['label_smoothing'],
                                   torch.FloatTensor([ws, wr]).to(device))

    # ---- Phase 1: Frozen backbone ----
    print("\n  PHASE 1: HEAD ONLY")
    phase1_start = time.time()

    if n_loaded > 0:
        freeze_backbone(model)
    else:
        unfreeze_all(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['phase1_lr'], weight_decay=config['weight_decay'],
    )
    scheduler = CosineAnnealingScheduler(optimizer, config['phase1_epochs'], eta_min=1e-6)

    best_f1r = 0
    best_state = None
    patience = 0

    for ep in range(config['phase1_epochs']):
        loss, acc = train_one_epoch(model, train_dl, criterion, optimizer, device,
                                    config['gradient_clip_norm'])
        preds, probs, labs = evaluate_model(model, test_dl, device)
        f1r = compute_f1_score(labs, preds, pos_label=1)
        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        training_history['phase1']['loss'].append(loss)
        training_history['phase1']['accuracy'].append(acc)
        training_history['phase1']['f1_risky'].append(f1r)
        training_history['phase1']['lr'].append(cur_lr)

        if (ep + 1) % 5 == 0 or f1r > best_f1r:
            print(f"   Ep{ep+1:3d} L={loss:.4f} A={acc:.3f} F1r={f1r:.3f}")

        if f1r > best_f1r:
            best_f1r = f1r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= config['phase1_patience']:
            print(f"   Early stop {ep + 1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    phase1_time = time.time() - phase1_start
    print(f"   Phase 1: {phase1_time/60:.1f} min, best F1r={best_f1r:.3f}")

    preds, probs, labs = evaluate_model(model, test_dl, device)
    p1_metrics = print_detailed_results(preds, probs, labs, "PHASE 1")


    if n_loaded > 0:
        print("\n  PHASE 2: FULL FINETUNE")
        phase2_start = time.time()
        unfreeze_all(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['phase2_lr'], weight_decay=config['weight_decay'],
        )
        scheduler = CosineAnnealingScheduler(optimizer, config['phase2_epochs'], eta_min=1e-7)
        patience = 0

        for ep in range(config['phase2_epochs']):
            loss, acc = train_one_epoch(model, train_dl, criterion, optimizer, device,
                                        config['gradient_clip_norm'])
            preds, probs, labs = evaluate_model(model, test_dl, device)
            f1r = compute_f1_score(labs, preds, pos_label=1)
            cur_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            training_history['phase2']['loss'].append(loss)
            training_history['phase2']['accuracy'].append(acc)
            training_history['phase2']['f1_risky'].append(f1r)
            training_history['phase2']['lr'].append(cur_lr)

            if (ep + 1) % 5 == 0 or f1r > best_f1r:
                print(f"   Ep{ep+1:3d} L={loss:.4f} A={acc:.3f} F1r={f1r:.3f}")

            if f1r > best_f1r:
                best_f1r = f1r
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience >= config['phase2_patience']:
                print(f"   Early stop {ep + 1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        phase2_time = time.time() - phase2_start
        print(f"   Phase 2: {phase2_time/60:.1f} min, best F1r={best_f1r:.3f}")

    # ---- Final Evaluation ----
    print("\n  FINAL EVALUATION")
    preds, probs, labs = evaluate_model(model, test_dl, device)
    final_metrics = print_detailed_results(preds, probs, labs, "STGCN++ Pretrained+Finetuned")

    if config.get('confidence_analysis', True):
        analyze_confidence_distribution(probs, labs)

    if config.get('threshold_search', True):
        thresh_results = search_optimal_threshold(probs, labs)
        final_metrics.update(thresh_results)

    if config.get('error_analysis', True):
        analyze_errors(preds, probs, labs)

    print_comparison_table(final_metrics)

    total_time = time.time() - total_start
    print(f"\n   Total time: {total_time/60:.1f} minutes")

    # Save
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'metrics': final_metrics,
        'loaded': n_loaded,
        'history': training_history,
        'timestamp': datetime.now().isoformat(),
    }, config['model_save_path'])
    print(f"   Model saved: {config['model_save_path']}")

    results_path = config.get('results_save_path', 'models/stgcn_results.json')
    serializable = {
        k: (v if not isinstance(v, np.ndarray) else v.tolist())
        for k, v in final_metrics.items()
    }
    with open(results_path, 'w') as f:
        json.dump({'metrics': serializable, 'timestamp': datetime.now().isoformat()}, f, indent=2)
    print(f"   Results saved: {results_path}")

    return model, final_metrics


if __name__ == '__main__':
    print("=" * 60)
    print("  PRETRAINED ST-GCN++ FOR SQUAT FORM ASSESSMENT")
    print("  Architecture verified from checkpoint tensor shapes")
    print("=" * 60)
    print(f"  Custom implementations (no sklearn/imblearn):")
    print(f"    - Stratified train/test split")
    print(f"    - SMOTE oversampling with k-NN")
    print(f"    - F1, Precision, Recall, Balanced Accuracy")
    print(f"    - ROC-AUC (trapezoidal integration)")
    print(f"    - Average Precision, MCC")
    print(f"    - Classification report formatting")
    print(f"    - Cosine annealing LR scheduler")
    print("=" * 60)

    model, metrics = train_model(CONFIG)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)