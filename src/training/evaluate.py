"""Evaluation metrics for the TLS functional state classifier."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate model on a DataLoader.

    Returns dict with: auc_roc, auc_pr, f1_macro, accuracy.
    """
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, return_domain_logits=False)
        probs = torch.softmax(out["task_logits"], dim=1)[:, 1].cpu().numpy()
        labels = batch.y.cpu().numpy()

        # Only evaluate on labeled samples
        labeled = labels >= 0
        if labeled.sum() > 0:
            all_probs.append(probs[labeled])
            all_labels.append(labels[labeled])

    if not all_probs:
        return {"auc_roc": 0.0, "auc_pr": 0.0, "f1_macro": 0.0, "accuracy": 0.0}

    y_prob  = np.concatenate(all_probs)
    y_true  = np.concatenate(all_labels)
    y_pred  = (y_prob >= 0.5).astype(int)

    metrics = {}
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.0
    try:
        metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auc_pr"] = 0.0

    metrics["f1_macro"]  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["accuracy"]  = accuracy_score(y_true, y_pred)

    return metrics
