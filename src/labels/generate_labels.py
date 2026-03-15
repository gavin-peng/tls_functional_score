"""
Pseudo-label generation for TLS functional state.

Labels are generated from high-resolution CosMx/Xenium data, where single-cell
resolution allows direct assessment of TLS internal composition.

Labels:
  0 = immunogenic TLS  (active germinal center, Tfh cells, IgG+ plasma output, low Tregs)
  1 = tolerogenic TLS  (high Tregs, suppressive myeloid, weak GC activity)
 -1 = uncertain         (excluded from supervised training)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad


# Thresholds loaded from configs/data.yaml by default
DEFAULT_THRESHOLDS = {
    "gc_score_threshold": 0.3,
    "treg_fraction_threshold": 0.15,
    "plasma_score_threshold": 0.2,
    "min_tls_cells": 50,
    "uncertain_buffer": 0.05,
}

LABEL_MAP = {0: "immunogenic", 1: "tolerogenic", -1: "uncertain"}


def _cell_fraction(adata_tls: ad.AnnData, cell_type: str, cell_type_col: str = "cell_type") -> float:
    """Fraction of cells in a TLS region matching a given cell type label."""
    if cell_type_col not in adata_tls.obs.columns:
        return 0.0
    return (adata_tls.obs[cell_type_col].str.lower() == cell_type.lower()).mean()


def _score_mean(adata_tls: ad.AnnData, score_col: str) -> float:
    """Mean value of a score column over all cells in the TLS region."""
    if score_col not in adata_tls.obs.columns:
        return 0.0
    return adata_tls.obs[score_col].mean()


def label_single_tls(
    adata_tls: ad.AnnData,
    thresholds: dict | None = None,
    cell_type_col: str = "cell_type",
) -> int:
    """
    Assign a functional state label to a single TLS region (subset AnnData).

    Requires that signature scoring has already been run (score_* columns present).

    Args:
        adata_tls: AnnData subset containing only cells within one TLS region.
        thresholds: Override default thresholds (from DEFAULT_THRESHOLDS).
        cell_type_col: Column in .obs with cell type annotations.

    Returns:
        0 (immunogenic), 1 (tolerogenic), or -1 (uncertain).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    if adata_tls.n_obs < t["min_tls_cells"]:
        return -1  # too few cells to reliably assess

    gc_score     = _score_mean(adata_tls, "score_germinal_center")
    treg_frac    = _cell_fraction(adata_tls, "treg", cell_type_col)
    if treg_frac == 0.0:
        # Fall back to FOXP3 expression score if cell type labels absent
        treg_frac = min(_score_mean(adata_tls, "score_tregs"), 1.0)
    plasma_score = _score_mean(adata_tls, "score_plasma_output")

    is_immunogenic = (
        gc_score     >= t["gc_score_threshold"]
        and treg_frac <= t["treg_fraction_threshold"]
        and plasma_score >= t["plasma_score_threshold"]
    )
    is_tolerogenic = (
        gc_score     <= t["gc_score_threshold"] - t["uncertain_buffer"]
        and treg_frac >= t["treg_fraction_threshold"] + t["uncertain_buffer"]
    )

    if is_immunogenic:
        return 0
    elif is_tolerogenic:
        return 1
    else:
        return -1  # ambiguous — excluded from training


def generate_labels_for_sample(
    adata: ad.AnnData,
    tls_cluster_col: str = "tls_cluster_id",
    thresholds: dict | None = None,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    """
    Generate functional state labels for all TLS regions in a sample.

    Args:
        adata: AnnData with TLS cluster assignments in .obs[tls_cluster_col].
               Must have signature scores already computed.
        tls_cluster_col: Column identifying TLS cluster membership (-1 = not TLS).
        thresholds: Override labeling thresholds.
        cell_type_col: Column with cell type annotations.

    Returns:
        DataFrame indexed by cluster ID with columns:
          'label' (int), 'label_name' (str), 'n_cells', 'gc_score', 'treg_fraction'
    """
    records = []
    cluster_ids = adata.obs[tls_cluster_col].unique()
    cluster_ids = [c for c in cluster_ids if c >= 0]

    for cid in sorted(cluster_ids):
        mask = adata.obs[tls_cluster_col] == cid
        region = adata[mask]

        label = label_single_tls(region, thresholds=thresholds, cell_type_col=cell_type_col)
        records.append({
            "cluster_id": cid,
            "n_cells": region.n_obs,
            "label": label,
            "label_name": LABEL_MAP[label],
            "gc_score": _score_mean(region, "score_germinal_center"),
            "treg_fraction": _cell_fraction(region, "treg", cell_type_col),
            "plasma_score": _score_mean(region, "score_plasma_output"),
        })

    df = pd.DataFrame(records).set_index("cluster_id")
    counts = df["label_name"].value_counts()
    print(f"Labels: {counts.to_dict()}")
    return df


def attach_labels_to_adata(
    adata: ad.AnnData,
    label_df: pd.DataFrame,
    tls_cluster_col: str = "tls_cluster_id",
) -> ad.AnnData:
    """
    Write per-cell/spot functional state labels back into adata.obs.

    Adds columns: 'functional_state' (int), 'functional_state_name' (str).
    Non-TLS spots get label = -1.
    """
    adata = adata.copy()
    adata.obs["functional_state"] = -1
    adata.obs["functional_state_name"] = "non_tls"

    for cid, row in label_df.iterrows():
        mask = adata.obs[tls_cluster_col] == cid
        adata.obs.loc[mask, "functional_state"] = row["label"]
        adata.obs.loc[mask, "functional_state_name"] = row["label_name"]

    return adata
