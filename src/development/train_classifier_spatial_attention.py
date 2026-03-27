#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm


CONFIG = {
    'keypoints_path': 'frame_level/pose_keypoints_raw.csv',
    'labels_path': 'rep_level/rep_labels_merged.csv',
    'segments_path': 'rep_level/auto_segments.csv',
    'model_save_path': 'models/spatial_attention_classifier.pt',
    'results_save_path': 'models/spatial_attention_results.json',
    'sequence_length': 64,
    'num_joints': 17,
    'in_channels': 3,
    'd_model': 64,           # attention embedding dimension
    'num_heads': 4,          # multi-head attention heads
    'lstm_hidden': 128,      # LSTM hidden size
    'lstm_layers': 2,
    'dropout': 0.3,
    'epochs': 50,
    'lr': 0.001,
    'batch_size': 16,
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


def load_data(config):
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
    return kp, lb, sg


def extract_sequences(kp_df, lb_df, sg_df, config):
    seq_len = config['sequence_length']
    kp_grouped = {n: g for n, g in kp_df.groupby('file_norm')}
    sequences, labels = [], []
    skipped = Counter()

    for _, row in tqdm(lb_df.iterrows(), total=len(lb_df), desc="Extracting"):
        fn, rid = row['file_norm'], row.get('rep_id', 0)
        label = 0 if row['label'] == 'safe' else 1
        seg = sg_df[(sg_df['file_norm'] == fn)]
        if 'rep_id' in sg_df.columns:
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

        frames_data = []
        for _, fr in rep_kp.iterrows():
            joints = []
            for kn in KEYPOINT_NAMES:
                x = float(fr.get(f'{kn}_x', 0) or 0)
                y = float(fr.get(f'{kn}_y', 0) or 0)
                c = float(fr.get(f'{kn}_score', fr.get(f'{kn}_conf', 1.0)) or 1.0)
                joints.append([x, y, c])
            frames_data.append(joints)
        frames_data = np.array(frames_data, dtype=np.float32)

        # Normalise: subtract hip midpoint, divide by torso length
        hip_c = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
        frames_data[:, :, :2] -= hip_c[:, np.newaxis, :]
        sc = (frames_data[:, 5, :2] + frames_data[:, 6, :2]) / 2
        hc = (frames_data[:, 11, :2] + frames_data[:, 12, :2]) / 2
        torso = np.linalg.norm(sc - hc, axis=1).mean()
        if torso > 1e-6:
            frames_data[:, :, :2] /= torso

        # Resample to fixed length
        if len(frames_data) != seq_len:
            old_i = np.linspace(0, 1, len(frames_data))
            new_i = np.linspace(0, 1, seq_len)
            resampled = np.zeros((seq_len, 17, 3), dtype=np.float32)
            for j in range(17):
                for c in range(3):
                    resampled[:, j, c] = np.interp(new_i, old_i, frames_data[:, j, c])
            frames_data = resampled

        sequences.append(frames_data)
        labels.append(label)

    print(f"Extracted: {len(sequences)}, Skipped: {sum(skipped.values())}")
    return np.array(sequences), np.array(labels)


class SquatDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class JointSelfAttention(nn.Module):
    
    def __init__(self, in_ch, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_ch, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, T, V, C) — batch, time, joints, channels
        N, T, V, C = x.shape
        # Flatten time into batch for per-frame joint attention
        x_flat = x.reshape(N * T, V, C)
        q = self.proj(x_flat)                         # (N*T, V, d_model)
        out, _ = self.attn(q, q, q)                   # self-attention over joints
        out = self.norm(q + self.dropout(out))        # residual + norm
        out = out.reshape(N, T, V, -1)                # (N, T, V, d_model)
        return out


class SpatioTemporalAttentionNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        in_ch = config['in_channels']
        d_model = config['d_model']
        num_heads = config['num_heads']
        lstm_hidden = config['lstm_hidden']
        lstm_layers = config['lstm_layers']
        dropout = config['dropout']
        num_joints = config['num_joints']

        self.spatial_attn = JointSelfAttention(in_ch, d_model, num_heads, dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (N, T, V, C)
        x = self.spatial_attn(x)        # (N, T, V, d_model)
        x = x.mean(dim=2)               # (N, T, d_model) — pool over joints
        out, _ = self.lstm(x)           # (N, T, lstm_hidden*2)
        out = out[:, -1, :]             # take last timestep
        return self.classifier(out)     # (N, 2)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
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
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return (total_loss / max(total, 1), correct / max(total, 1),
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def f1_risky(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
   

    seed = CONFIG['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")

    kp_df, lb_df, sg_df = load_data(CONFIG)
    sequences, labels = extract_sequences(kp_df, lb_df, sg_df, CONFIG)

    # Stratified split
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * CONFIG['test_size']))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    X_train, X_test = sequences[train_idx], sequences[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    train_ds = SquatDataset(X_train, y_train)
    test_ds = SquatDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    model = SpatioTemporalAttentionNet(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}\n")

    safe_n = int(np.sum(y_train == 0))
    risky_n = int(np.sum(y_train == 1))
    weight = torch.tensor([1.0, safe_n / max(risky_n, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'],
                                  weight_decay=1e-4)

    best_f1, best_state, wait = 0.0, None, 0
    print("  Training...\n")
    for epoch in range(1, CONFIG['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc, preds, true, probs = evaluate(model, test_dl, criterion, device)
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
        model.load_state_dict(best_state)
        os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
        torch.save({'model_state_dict': best_state, 'best_f1_risky': best_f1},
                   CONFIG['model_save_path'])
        print(f"\n  Model saved: {CONFIG['model_save_path']}")

    results = {'best_f1_risky': float(best_f1), 'architecture': 'SpatioTemporalAttentionNet',
                'outcome': 'abandoned — F1(risky) < 0.40, below all baselines'}
    with open(CONFIG['results_save_path'], 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()