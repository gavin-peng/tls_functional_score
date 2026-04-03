"""
Train TLS functional state classifier (GNN).

Replaces notebook 04_gnn_training.ipynb as the primary training entrypoint.
Saves best_model.pt, tls_predictions.csv, tls_embeddings.pt, and training
curves to checkpoints/.

Required environment variable:
    TLS_DATA_ROOT  -- directory containing:
                        processed/tls_graphs.pt      (built by nb03)
                        splits/arch_config.json      (built by nb03)

Optional:
    TLS_PROJECT_ROOT  -- project root directory (default: cwd)

Usage:
    TLS_DATA_ROOT=/path/to/data python scripts/train.py

Data download:
    tls_graphs.pt and arch_config.json are deposited on Zenodo (see README).
    To regenerate from raw GEO data (GSE175540), run notebooks 01-03.
"""

import os
import sys
import json
import random
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('TLS_PROJECT_ROOT', Path.cwd())).resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score,
)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from src.models.gnn import TLSFunctionalGNN
from src.training.losses import TLSTrainingLoss

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if 'TLS_DATA_ROOT' not in os.environ:
    raise EnvironmentError(
        "TLS_DATA_ROOT is not set.\n"
        "Point it to the directory containing processed/tls_graphs.pt\n"
        "and splits/arch_config.json (download from Zenodo or run nb01-nb03).\n"
        "Example: TLS_DATA_ROOT=/path/to/data python scripts/train.py"
    )

DATA_ROOT  = Path(os.environ['TLS_DATA_ROOT'])
GRAPHS_PT  = DATA_ROOT / 'processed' / 'tls_graphs.pt'
ARCH_CFG   = DATA_ROOT / 'splits' / 'arch_config.json'
CKPT_DIR   = PROJECT_ROOT / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

for p in [GRAPHS_PT, ARCH_CFG]:
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.\n"
            "Download processed data from Zenodo (see README) or run nb01-nb03."
        )

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ---------------------------------------------------------------------------
# Load graphs (single file, split by g.split attribute)
# ---------------------------------------------------------------------------
print('Loading graphs ...')
t0 = time.time()
graphs = torch.load(GRAPHS_PT, weights_only=False)
print(f'Loaded {len(graphs)} graphs in {time.time() - t0:.1f}s')

train_graphs = [g for g in graphs if g.split == 'train']
val_graphs   = [g for g in graphs if g.split == 'val']
test_graphs  = [g for g in graphs if g.split == 'test']
print(f'Split: train={len(train_graphs)}  val={len(val_graphs)}  test={len(test_graphs)}')

# ---------------------------------------------------------------------------
# Arch config (written by nb03)
# ---------------------------------------------------------------------------
with open(ARCH_CFG) as f:
    arch = json.load(f)
IN_DIM   = arch['in_dim']
K_NICHE  = arch.get('k_niche_clusters', arch.get('k_niche', 5))
K_REGION = arch.get('k_region_clusters', arch.get('k_region', 2))
print(f'Arch: in_dim={IN_DIM}  k_niche={K_NICHE}  k_region={K_REGION}')

# ---------------------------------------------------------------------------
# Class weights (inverse-frequency on training set)
# ---------------------------------------------------------------------------
from collections import Counter
train_counts = Counter(int(g.y) for g in train_graphs)
n_immuno = train_counts.get(0, 1)
n_tolero = train_counts.get(1, 1)
n_labeled = n_immuno + n_tolero
w_immuno = n_labeled / (2 * n_immuno)
w_tolero = n_labeled / (2 * n_tolero)
class_weights = torch.tensor([w_immuno, w_tolero], dtype=torch.float32).to(DEVICE)
print(f'Class weights: immunogenic={w_immuno:.3f}  tolerogenic={w_tolero:.3f}')

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_loader   = DataLoader(graphs,       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
HIDDEN  = 128
HEADS   = 4
DROPOUT = 0.2

model = TLSFunctionalGNN(
    in_dim            = IN_DIM,
    hidden            = HIDDEN,
    n_classes         = 2,
    n_niche_clusters  = K_NICHE,
    n_region_clusters = K_REGION,
    heads             = HEADS,
    dropout           = DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {n_params:,}')

# ---------------------------------------------------------------------------
# Optimizer, scheduler, loss
# ---------------------------------------------------------------------------
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 100
PATIENCE     = 15
WARMUP_FRAC  = 0.05

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)

loss_fn = TLSTrainingLoss(
    task_weight        = 1.0,
    domain_weight      = 0.0,
    contrastive_weight = 0.1,
    aux_weight         = 0.01,
    n_classes          = 2,
    class_weights      = class_weights,
    use_focal          = True,
    focal_gamma        = 2.0,
)

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def warmup_lr(optimizer, step, warmup_steps, base_lr):
    if step < warmup_steps:
        scale = (step + 1) / max(warmup_steps, 1)
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * scale


def train_one_epoch(model, loader, optimizer, loss_fn, device, step, warmup_steps):
    model.train()
    totals = {k: 0.0 for k in ['total', 'task', 'contrastive', 'aux']}
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        warmup_lr(optimizer, step, warmup_steps, LR)
        optimizer.zero_grad()
        logits, aux_loss = model(batch.x, batch.edge_index, batch.batch)
        embeddings = model.get_embeddings(batch.x, batch.edge_index, batch.batch)
        losses = loss_fn(
            task_logits  = logits,
            task_labels  = batch.y,
            embeddings   = embeddings,
            aux_loss     = aux_loss,
        )
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k in ['total', 'task', 'contrastive', 'aux']:
            totals[k] += losses[k].item()
        n_batches += 1
        step += 1
    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return avg, step


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels = batch.y.cpu().numpy()
        mask   = labels >= 0
        if mask.sum() > 0:
            all_probs.append(probs[mask])
            all_labels.append(labels[mask])
    if not all_probs:
        return {'auc_roc': 0.0, 'auc_pr': 0.0, 'f1_macro': 0.0, 'accuracy': 0.0}
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}
    try:    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    except: metrics['auc_roc'] = 0.0
    try:    metrics['auc_pr']  = average_precision_score(y_true, y_prob)
    except: metrics['auc_pr']  = 0.0
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    return metrics


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    records = []
    all_embeddings = []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        emb   = model.get_embeddings(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(logits, dim=1)
        all_embeddings.append(emb.cpu())
        cluster_ids = batch.cluster_id if hasattr(batch, 'cluster_id') else [-1] * len(batch)
        sample_ids  = batch.sample_id  if hasattr(batch, 'sample_id')  else [''] * len(batch)
        splits      = batch.split      if hasattr(batch, 'split')      else [''] * len(batch)
        label_names = batch.label_name if hasattr(batch, 'label_name') else [''] * len(batch)
        for i in range(len(batch)):
            cid = cluster_ids[i]
            if hasattr(cid, 'item'):
                cid = cid.item()
            records.append({
                'cluster_id'          : cid,
                'sample_id'           : sample_ids[i],
                'split'               : splits[i],
                'label_name'          : label_names[i],
                'y_true'              : int(batch.y[i]),
                'p_immunogenic'       : float(probs[i, 0]),
                'p_tolerogenic'       : float(probs[i, 1]),
                'tls_functional_score': float(probs[i, 1]),
                'pred_label'          : int(probs[i].argmax()),
            })
    return pd.DataFrame(records), torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
WARMUP_STEPS = int(WARMUP_FRAC * EPOCHS * len(train_loader))
print(f'\nWarmup steps: {WARMUP_STEPS}  |  Max epochs: {EPOCHS}  |  Patience: {PATIENCE}')
print('Starting training ...\n')

CKPT_BEST    = CKPT_DIR / 'best_model.pt'
best_val_auc = 0.0
best_epoch   = 0
patience_ctr = 0
global_step  = 0
history      = []
t_start      = time.time()

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()
    train_losses, global_step = train_one_epoch(
        model, train_loader, optimizer, loss_fn, DEVICE, global_step, WARMUP_STEPS
    )
    scheduler.step(epoch - 1)
    val_metrics = evaluate_loader(model, val_loader, DEVICE)
    lr_now = optimizer.param_groups[0]['lr']
    history.append({
        'epoch'      : epoch,
        'train_loss' : train_losses['total'],
        'train_task' : train_losses['task'],
        **{f'val_{k}': v for k, v in val_metrics.items()},
        'lr'         : lr_now,
    })
    print(
        f'Epoch {epoch:03d}/{EPOCHS} | '
        f'loss={train_losses["total"]:.4f} (task={train_losses["task"]:.4f}) | '
        f'val_auc={val_metrics["auc_roc"]:.4f} val_ap={val_metrics["auc_pr"]:.4f} '
        f'val_f1={val_metrics["f1_macro"]:.4f} | '
        f'lr={lr_now:.6f} | {time.time() - t_ep:.1f}s',
        flush=True,
    )
    if val_metrics['auc_roc'] > best_val_auc:
        best_val_auc = val_metrics['auc_roc']
        best_epoch   = epoch
        patience_ctr = 0
        torch.save({
            'epoch'      : epoch,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'val_auc'    : best_val_auc,
            'arch'       : dict(in_dim=IN_DIM, hidden=HIDDEN, k_niche=K_NICHE, k_region=K_REGION),
        }, CKPT_BEST)
        print(f'  >> Best model saved (val_auc={best_val_auc:.4f})', flush=True)
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f'Early stopping at epoch {epoch} (best was epoch {best_epoch})', flush=True)
            break

print(f'\nTraining complete in {(time.time() - t_start) / 60:.1f} min')
print(f'Best val AUC-ROC: {best_val_auc:.4f} at epoch {best_epoch}')

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
df_hist = pd.DataFrame(history)
df_hist.to_csv(CKPT_DIR / 'training_history.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(df_hist['epoch'], df_hist['train_loss'], label='Total', lw=2)
ax.plot(df_hist['epoch'], df_hist['train_task'], label='Task',  lw=2, linestyle='--')
ax.axvline(best_epoch, color='red', lw=1, linestyle=':')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss'); ax.legend()

ax = axes[1]
ax.plot(df_hist['epoch'], df_hist['val_auc_roc'], label='AUC-ROC', lw=2)
ax.plot(df_hist['epoch'], df_hist['val_auc_pr'],  label='AUC-PR',  lw=2)
ax.axvline(best_epoch, color='red', lw=1, linestyle=':')
ax.set_xlabel('Epoch'); ax.set_ylabel('AUC'); ax.set_title('Validation AUC'); ax.legend()

ax = axes[2]
ax.plot(df_hist['epoch'], df_hist['val_f1_macro'], label='F1 macro', lw=2)
ax.plot(df_hist['epoch'], df_hist['val_accuracy'], label='Accuracy', lw=2)
ax.axvline(best_epoch, color='red', lw=1, linestyle=':')
ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.set_title('Validation Metrics'); ax.legend()

plt.tight_layout()
fig.savefig(CKPT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
print(f'Saved: {CKPT_DIR}/training_curves.png')

# ---------------------------------------------------------------------------
# Evaluation on best checkpoint
# ---------------------------------------------------------------------------
ckpt = torch.load(CKPT_BEST, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['state_dict'])

val_metrics  = evaluate_loader(model, val_loader,  DEVICE)
test_metrics = evaluate_loader(model, test_loader, DEVICE)

print('\n=== Val set ===')
for k, v in val_metrics.items():
    print(f'  {k}: {v:.4f}')
print('\n=== Test set ===')
for k, v in test_metrics.items():
    print(f'  {k}: {v:.4f}')

# ---------------------------------------------------------------------------
# Inference: predictions + embeddings on all graphs
# ---------------------------------------------------------------------------
print(f'\nRunning inference on all {len(graphs)} TLS graphs ...')
pred_df, embeddings = run_inference(model, all_loader, DEVICE)

pred_csv = DATA_ROOT / 'processed' / 'tls_predictions.csv'
emb_pt   = DATA_ROOT / 'processed' / 'tls_embeddings.pt'
pred_df.to_csv(pred_csv, index=False)
torch.save(embeddings, emb_pt)

# Local copies for notebooks
pred_df.to_csv(CKPT_DIR / 'tls_predictions.csv', index=False)
torch.save(embeddings, CKPT_DIR / 'tls_embeddings.pt')

print(f'Saved predictions -> {pred_csv}')
print(f'Saved embeddings  -> {emb_pt}')
print(f'\n=== Score distribution across all {len(pred_df)} TLS regions ===')
print(pred_df['tls_functional_score'].describe().round(4))
