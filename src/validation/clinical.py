"""
Clinical validation: link TLS functional state scores to ICB response and survival.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("WARNING: lifelines not installed. Install: pip install lifelines")

from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu


def compute_patient_tls_score(
    tls_scores_per_region: dict[str, np.ndarray],
    aggregation: str = "max",
) -> pd.Series:
    """
    Aggregate per-TLS-region functional state scores into a single patient score.

    Args:
        tls_scores_per_region: {patient_id: array of per-TLS immunogenic probabilities}
        aggregation: 'max' | 'mean' | 'fraction_immunogenic'

    Returns:
        pd.Series indexed by patient_id.
    """
    scores = {}
    for pid, region_scores in tls_scores_per_region.items():
        if len(region_scores) == 0:
            scores[pid] = 0.0
            continue
        if aggregation == "max":
            scores[pid] = float(region_scores.max())
        elif aggregation == "mean":
            scores[pid] = float(region_scores.mean())
        elif aggregation == "fraction_immunogenic":
            scores[pid] = float((region_scores >= 0.5).mean())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return pd.Series(scores, name="tls_functional_score")


def icb_response_auc(
    patient_scores: pd.Series,
    response_labels: pd.Series,
) -> dict[str, float]:
    """
    Compute AUC-ROC for ICB response prediction from TLS functional state score.

    Args:
        patient_scores: Patient-level TLS functional state score.
        response_labels: Binary ICB response (1=responder, 0=non-responder).

    Returns:
        Dict with 'auc', 'optimal_threshold', 'sensitivity', 'specificity'.
    """
    shared = patient_scores.index.intersection(response_labels.index)
    scores = patient_scores[shared].values
    labels = response_labels[shared].values

    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Youden's J statistic for optimal threshold
    j = tpr - fpr
    opt_idx = j.argmax()

    # Comparison: Mann-Whitney U between responders vs non-responders
    resp  = scores[labels == 1]
    nresp = scores[labels == 0]
    stat, p_val = mannwhitneyu(resp, nresp, alternative="greater")

    return {
        "auc": auc,
        "optimal_threshold": thresholds[opt_idx],
        "sensitivity": tpr[opt_idx],
        "specificity": 1 - fpr[opt_idx],
        "mannwhitney_p": p_val,
        "n_responders": int(labels.sum()),
        "n_non_responders": int((1 - labels).sum()),
    }


def survival_analysis(
    patient_scores: pd.Series,
    survival_time: pd.Series,
    event_observed: pd.Series,
    score_threshold: float | None = None,
    output_dir: str | None = None,
) -> dict:
    """
    Kaplan-Meier survival analysis stratified by TLS functional state score.

    Args:
        patient_scores: Patient-level TLS functional state score.
        survival_time: Time-to-event (e.g., months).
        event_observed: Binary event indicator (1=event, 0=censored).
        score_threshold: Cut-off for high/low groups; uses median if None.
        output_dir: If provided, save KM plot to this directory.

    Returns:
        Dict with log-rank test p-value, median survival times.
    """
    if not HAS_LIFELINES:
        raise ImportError("Install lifelines: pip install lifelines")

    shared = patient_scores.index.intersection(survival_time.index)
    scores  = patient_scores[shared]
    times   = survival_time[shared]
    events  = event_observed[shared]

    threshold = score_threshold if score_threshold is not None else scores.median()
    high_group = scores >= threshold
    low_group  = ~high_group

    # Log-rank test
    lr = logrank_test(
        times[high_group], times[low_group],
        event_observed_A=events[high_group],
        event_observed_B=events[low_group],
    )

    # KM curves
    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()
    kmf_high.fit(times[high_group], events[high_group], label="High TLS score")
    kmf_low.fit(times[low_group],  events[low_group],  label="Low TLS score")

    if output_dir:
        fig, ax = plt.subplots(figsize=(8, 6))
        kmf_high.plot_survival_function(ax=ax, ci_show=True)
        kmf_low.plot_survival_function(ax=ax, ci_show=True)
        ax.set_xlabel("Time (months)")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"TLS Functional State Score — KM Curve\n(log-rank p={lr.p_value:.4f})")
        ax.text(0.6, 0.8, f"p = {lr.p_value:.4f}", transform=ax.transAxes, fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/km_curve.pdf", dpi=300)
        plt.close(fig)
        print(f"KM curve saved to {output_dir}/km_curve.pdf")

    return {
        "logrank_p": lr.p_value,
        "logrank_stat": lr.test_statistic,
        "median_survival_high": kmf_high.median_survival_time_,
        "median_survival_low":  kmf_low.median_survival_time_,
        "threshold_used": threshold,
        "n_high": int(high_group.sum()),
        "n_low":  int(low_group.sum()),
    }


def benchmark_vs_baselines(
    patient_scores: pd.Series,
    baseline_scores: dict[str, pd.Series],
    response_labels: pd.Series,
) -> pd.DataFrame:
    """
    Compare TLS functional state score against simpler baselines.

    Baselines typically include:
      - 'tls_presence': binary TLS detected (yes/no)
      - 'tls_count': number of TLS per sample
      - 'tls_cxcl13': mean CXCL13 expression in TLS regions
      - 'tls_maturation': maturation stage score

    Returns DataFrame with AUC-ROC for each score.
    """
    results = []
    all_scores = {"TLS_functional_state": patient_scores, **baseline_scores}

    for name, scores in all_scores.items():
        shared = scores.index.intersection(response_labels.index)
        if len(shared) < 5:
            continue
        try:
            auc = roc_auc_score(response_labels[shared].values, scores[shared].values)
        except ValueError:
            auc = np.nan
        results.append({"method": name, "auc_roc": auc, "n_patients": len(shared)})

    df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    print(df.to_string(index=False))
    return df
