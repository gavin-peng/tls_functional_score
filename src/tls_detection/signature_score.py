"""
Gene signature scoring for TLS detection.

Scores each spot/cell using a panel of TLS-associated gene sets,
then combines them into a composite TLS likelihood score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import yaml
from pathlib import Path


# Default TLS gene signatures (can be overridden by configs/data.yaml)
TLS_SIGNATURES = {
    "cxcl13_anchor": ["CXCL13"],
    "b_cell_core":   ["MS4A1", "CD19", "CD79A", "CD79B"],
    "germinal_center": ["BCL6", "AICDA", "MYBL1", "LMO2", "MEF2B"],
    "t_cell_zone":   ["CD3D", "CD3E", "CCR7", "SELL"],
    "tfh":           ["CXCR5", "PDCD1", "ICOS", "BTLA"],
    "plasma_output": ["MZB1", "SDC1", "IGHG1", "IGHG2", "IGHA1", "JCHAIN"],
    "hev_markers":   ["TNFRSF9", "GLYCAM1", "MADCAM1"],
    "tls_chemokines": ["CCL19", "CCL21", "LTB"],
}

TOLEROGENIC_SIGNATURES = {
    "tregs":       ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "myeloid_sup": ["IL10", "TGFB1", "CD163", "MRC1", "ARG1"],
    "exhaustion":  ["HAVCR2", "TIGIT", "LAG3", "PDCD1", "TOX"],
}


def score_tls_signatures(
    adata: ad.AnnData,
    signatures: dict[str, list[str]] | None = None,
    layer: str | None = "log_norm",
) -> ad.AnnData:
    """
    Add per-spot TLS module scores to adata.obs using scanpy's score_genes.

    Args:
        adata: AnnData (Visium spots or CosMx cells).
        signatures: Dict of {score_name: [gene_list]}. Uses defaults if None.
        layer: Which layer to score from ('log_norm' recommended).

    Returns:
        adata with new columns in .obs: one per signature + 'tls_composite_score'.
    """
    adata = adata.copy()
    if signatures is None:
        signatures = TLS_SIGNATURES

    # Temporarily set .X to the target layer for scoring
    if layer and layer in adata.layers:
        X_orig = adata.X.copy()
        adata.X = adata.layers[layer]

    for name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) == 0:
            print(f"  WARNING: no genes from '{name}' found in dataset -- skipping")
            adata.obs[f"score_{name}"] = 0.0
            continue
        if len(present) < len(genes):
            missing = set(genes) - set(present)
            print(f"  NOTE: '{name}' missing {len(missing)} genes: {missing}")
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}", use_raw=False)

    if layer and layer in adata.layers:
        adata.X = X_orig

    # Composite TLS score: weighted average of best-performing signatures
    # (germinal_center excluded -- AUROC ~0.50 at Visium resolution, drags composite down)
    core_scores = ["score_b_cell_core", "score_plasma_output", "score_tls_chemokines", "score_t_cell_zone"]
    available = [s for s in core_scores if s in adata.obs.columns]
    if available:
        raw = adata.obs[available].values
        # Min-max normalize each component then average
        raw_norm = (raw - raw.min(axis=0)) / ((raw.max(axis=0) - raw.min(axis=0)) + 1e-8)
        adata.obs["tls_composite_score"] = raw_norm.mean(axis=1)
    else:
        adata.obs["tls_composite_score"] = 0.0

    # CXCL13 raw expression as a quick anchor
    if "CXCL13" in adata.var_names:
        layer_key = layer if layer in adata.layers else None
        X = adata.layers[layer_key] if layer_key else adata.X
        idx = list(adata.var_names).index("CXCL13")
        cxcl13 = np.asarray(X[:, idx].todense()).flatten() if hasattr(X, "todense") else X[:, idx]
        adata.obs["cxcl13_expression"] = cxcl13

    return adata


def score_tolerogenic_signatures(adata: ad.AnnData, **kwargs) -> ad.AnnData:
    """Add tolerogenic/immunosuppressive signature scores to adata.obs."""
    return score_tls_signatures(adata, signatures=TOLEROGENIC_SIGNATURES, **kwargs)


def load_signatures_from_config(config_path: str | Path) -> dict[str, list[str]]:
    """Load gene signatures from configs/data.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tls_signatures", TLS_SIGNATURES)
