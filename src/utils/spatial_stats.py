"""Utility functions for spatial statistics and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree


def plot_spatial_score(
    adata: ad.AnnData,
    score_col: str,
    title: str | None = None,
    spot_size: float = 20.0,
    cmap: str = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    highlight_tls: bool = True,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    """
    Plot a score (e.g., tls_composite_score) overlaid on tissue coordinates.

    Args:
        adata: AnnData with .obsm['spatial'] and score in .obs.
        score_col: Column name in adata.obs.
        highlight_tls: If True, outline TLS candidate spots in black.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    coords = adata.obsm["spatial"]
    scores = adata.obs[score_col].values

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=scores, cmap=cmap, s=spot_size,
        vmin=vmin, vmax=vmax, alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label=score_col, shrink=0.6)

    if highlight_tls and "tls_candidate" in adata.obs.columns:
        tls_mask = adata.obs["tls_candidate"].values
        ax.scatter(
            coords[tls_mask, 0], coords[tls_mask, 1],
            facecolors="none", edgecolors="black",
            s=spot_size * 2, linewidths=0.5, label="TLS candidate",
        )
        ax.legend(loc="upper right")

    ax.set_aspect("equal")
    ax.set_title(title or score_col)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return ax


def plot_tls_functional_score(
    adata: ad.AnnData,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    """Plot the TLS functional state score overlaid on tissue."""
    return plot_spatial_score(
        adata,
        score_col="tls_functional_score",
        title="TLS Functional State Score\n(immunogenic → 1.0)",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        ax=ax,
        save_path=save_path,
    )


def compute_tls_neighbor_composition(
    adata: ad.AnnData,
    radius_um: float = 200.0,
    score_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each TLS region, compute the mean signature scores of its microenvironmental
    neighborhood (spots within radius_um that are NOT part of the TLS itself).

    This captures the "context" around each TLS — e.g., whether it is
    surrounded by suppressive myeloid cells or stromal fibroblasts.

    Returns DataFrame indexed by tls_cluster_id.
    """
    if score_cols is None:
        score_cols = [c for c in adata.obs.columns if c.startswith("score_")]

    coords = adata.obsm["spatial"]
    tree = cKDTree(coords)
    records = []

    for cid in adata.obs["tls_cluster_id"].unique():
        if cid < 0:
            continue
        tls_mask = (adata.obs["tls_cluster_id"] == cid).values
        tls_coords = coords[tls_mask]
        tls_centre = tls_coords.mean(axis=0)

        # Find non-TLS spots within radius
        neighbor_ids = tree.query_ball_point(tls_centre, r=radius_um)
        neighbor_ids = [i for i in neighbor_ids if not tls_mask[i]]

        if not neighbor_ids:
            continue

        row = {"cluster_id": cid}
        for col in score_cols:
            if col in adata.obs.columns:
                row[f"neighbor_{col}"] = adata.obs[col].iloc[neighbor_ids].mean()
        records.append(row)

    return pd.DataFrame(records).set_index("cluster_id")
