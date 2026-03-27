#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm

# ===================== CONFIG =====================
CONFIG = {
    'keypoints_path': 'frame_level/pose_keypoints_raw.csv',
    'labels_path': 'rep_level/rep_labels_merged.csv',
    'segments_path': 'rep_level/auto_segments.csv',
    'model_save_path': 'models/skeleton_heatmap_3dcnn.pt',
    'results_save_path': 'models/skeleton_heatmap_3dcnn_results.json',
    'sequence_length': 32,   # reduced from 64 — memory constraint with image volumes
    'num_joints': 17,
    'heatmap_size': 64,      # H x W of each rendered skeleton frame
    'heatmap_sigma': 3,      # gaussian spread radius for joint rendering
    'epochs': 40,
    'lr': 0.0005,
    'batch_size': 8,         # small batch — (N, 1, 32, 64, 64) tensors are large
    'patience': 10,
    'test_size': 0.2,
    'random_seed': 42,
}

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]



def normalize_filename(f):
    f = os.path.basename(str(f))
    if f.startswith('pose_'):
        f = f[5:]
    return os.path.splitext(f)[0].lower().strip()


def render_skeleton_heatmap(joints_xy, heatmap_size, sigma):
    
    H = W = heatmap_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    
    scale = H / 4.0  # maps roughly -2..2 range to 0..H

    for v in range(joints_xy.shape[0]):
        cx = int(W / 2 + joints_xy[v, 0] * scale)
        cy = int(H / 2 + joints_xy[v, 1] * scale)

        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue

        # Draw Gaussian centred at (cx, cy)
        for i in range(max(0, cy - sigma * 3), min(H, cy + sigma * 3 + 1)):
            for j in range(max(0, cx - sigma * 3), min(W, cx + sigma * 3 + 1)):
                d2 = (i - cy) ** 2 + (j - cx) ** 2
                heatmap[i, j] += np.exp(-d2 / (2 * sigma ** 2))

    # Normalise to [0, 1]
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx

    return heatmap


def load_and_extract(config):
    
    kp = pd.read_csv(config['keypoints_path'])
    kp['file_norm'] = kp['file'].apply(normalize_filename)
    lb = pd.read_csv(config['labels_path'])
    lb = lb[lb['label'].isin(['safe', 'risky'])].copy()
    lb['file_norm'] = lb['file'].apply(normalize_filename)
    sg = pd.read_csv(config['segments_path'])
    if 'source_file' in sg.columns:
        sg['file_norm'] = sg['source_file'].apply(normalize_filename)
    else:
        sg['file_norm'] = sg['file'].apply(normalize_filename)

    print(f"Labels: {len(lb)} (safe={sum(lb['label']=='safe')}, risky={sum(lb['label']=='risky')})")

    seq_len = config['sequence_length']
    H = W = config['heatmap_size']
    sigma = config['heatmap_sigma']
    kp_grouped = {n: g for n, g in kp.groupby('file_norm')}

    volumes, labels = [], []
    skipped = Counter()

    for _, row in tqdm(lb.iterrows(), total=len(lb), desc="Rendering heatmaps"):
        fn, rid = row['file_norm'], row.get('rep_id', 0)
        label = 0 if row['label'] == 'safe' else 1
        seg = sg[sg['file_norm'] == fn]
        if 'rep_id' in sg.columns:
            s2 = seg[seg['rep_id'] == rid]
            if len(s2) > 0:
                seg = s2
        if len(seg) == 0 or fn not in kp_grouped:
            skipped['no_data'] += 1
            continue

        sf = int(seg.iloc[0].get('start_frame', 0))
        ef = int(seg.iloc[0].get('end_frame', sf + 64))
        vk = kp_grouped[fn]
        rep_kp = vk[(vk['frame'] >= sf) & (vk['frame'] <= ef)].sort_values('frame')
        if len(rep_kp) < 10:
            skipped['too_short'] += 1
            continue

        # Extract joint coordinates per frame
        frames_xy = []
        for _, fr in rep_kp.iterrows():
            joints = []
            for kn in KEYPOINT_NAMES:
                x = float(fr.get(f'{kn}_x', 0) or 0)
                y = float(fr.get(f'{kn}_y', 0) or 0)
                joints.append([x, y])
            frames_xy.append(joints)
        frames_xy = np.array(frames_xy, dtype=np.float32)  # (T, V, 2)

        # Normalise: subtract hip midpoint, divide by torso length
        hip_c = (frames_xy[:, 11, :] + frames_xy[:, 12, :]) / 2
        frames_xy -= hip_c[:, np.newaxis, :]
        sc = (frames_xy[:, 5, :] + frames_xy[:, 6, :]) / 2
        hc = (frames_xy[:, 11, :] + frames_xy[:, 12, :]) / 2
        torso = np.linalg.norm(sc - hc, axis=1).mean()
        if torso > 1e-6:
            frames_xy /= torso

        # Resample to fixed temporal length
        T = len(frames_xy)
        if T != seq_len:
            old_i = np.linspace(0, 1, T)
            new_i = np.linspace(0, 1, seq_len)
            resampled = np.zeros((seq_len, 17, 2), dtype=np.float32)
            for v in range(17):
                for c in range(2):
                    resampled[:, v, c] = np.interp(new_i, old_i, frames_xy[:, v, c])
            frames_xy = resampled

        # Render each frame as a heatmap image → volume of shape (T, H, W)
        volume = np.zeros((seq_len, H, W), dtype=np.float32)
        for t in range(seq_len):
            volume[t] = render_skeleton_heatmap(frames_xy[t], H, sigma)

        volumes.append(volume)
        labels.append(label)

    volumes = np.array(volumes)[:, np.newaxis, :, :, :]  # (N, 1, T, H, W)
    labels = np.array(labels)
    print(f"Extracted: {len(volumes)}, Shape: {volumes.shape}, Skipped: {sum(skipped.values())}")
    print(f"Memory per sample: {volumes[0].nbytes / 1024:.1f} KB  "
          f"(vs ~{64*17*3*4/1024:.1f} KB for GCN tensor)")
    return volumes, labels


class SkeletonHeatmapDataset(Dataset):
    def __init__(self, volumes, labels):
        self.volumes = torch.tensor(volumes, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.volumes[idx], self.labels[idx]


class SkeletonHeatmap3DCNN(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: (N, 1, T, 64, 64) -> (N, 32, T/2, 32, 32)
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            # Block 2: (N, 32, T/2, 32, 32) -> (N, 64, T/2, 16, 16)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            # Block 3: (N, 64, T/2, 16, 16) -> (N, 128, T/4, 8, 8)
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

    
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (N, 1, T, H, W)
        x = self.conv_blocks(x)        # (N, 128, T', H', W')
        x = self.global_pool(x)        # (N, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)     # (N, 128)
        return self.classifier(x)      # (N, 2)


def f1_risky(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for vols, labels in loader:
        vols, labels = vols.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(vols)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for vols, labels in loader:
            vols, labels = vols.to(device), labels.to(device)
            logits = model(vols)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (total_loss / max(total, 1), correct / max(total, 1),
            np.array(all_preds), np.array(all_labels))


def main():
    print("=" * 60)
    print("  3D CNN ON SKELETON HEATMAP IMAGES")
    print("  (Development script — not used in final pipeline)")
    print("=" * 60)

    seed = CONFIG['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")

    volumes, labels = load_and_extract(CONFIG)

    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * CONFIG['test_size']))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    train_ds = SkeletonHeatmapDataset(volumes[train_idx], labels[train_idx])
    test_ds = SkeletonHeatmapDataset(volumes[test_idx], labels[test_idx])
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = SkeletonHeatmap3DCNN(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}\n")

    safe_n = int(np.sum(labels[train_idx] == 0))
    risky_n = int(np.sum(labels[train_idx] == 1))
    weight = torch.tensor([1.0, safe_n / max(risky_n, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)

    best_f1, best_state, wait = 0.0, None, 0
    print("  Training...\n")

    for epoch in range(1, CONFIG['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc, preds, true = evaluate(model, test_dl, criterion, device)
        f1 = f1_risky(true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1
            marker = ""

        if epoch <= 5 or epoch % 5 == 0 or marker:
            print(f"  Ep {epoch:3d}: TrL={tr_loss:.4f} VL={val_loss:.4f} "
                  f"F1(risky)={f1:.3f}{marker}")

        if wait >= CONFIG['patience']:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best F1(risky): {best_f1:.4f}")
    

    if best_state:
        os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
        torch.save({'model_state_dict': best_state, 'best_f1_risky': best_f1},
                   CONFIG['model_save_path'])
        print(f"\n  Model saved: {CONFIG['model_save_path']}")

    results = {
        'best_f1_risky': float(best_f1),
        'architecture': 'SkeletonHeatmap3DCNN',
        'outcome': 'abandoned — poor risky-class discrimination; '
                   'image rendering discards skeleton graph structure',
    }
    with open(CONFIG['results_save_path'], 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()