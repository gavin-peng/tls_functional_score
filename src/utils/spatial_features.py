"""
Spot-level spatial feature computation for TLS analysis.

All functions expect to be called on a single-sample AnnData.
Visium slides share array-grid coordinates across samples, so building
a cross-sample KNN graph creates spurious edges between different slides.
"""

from __future__ import annotations

import numpy as np
import anndata as ad
from scipy.spatial import cKDTree


def _build_knn(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (nbrs_all, nbrs_only) for k-NN on coords.

    nbrs_all:  (n, k+1) — self in col 0, then k neighbors
    nbrs_only: (n, k)   — neighbors only, for adjacency / gradient
    """
    tree = cKDTree(coords)
    _, nbrs_all = tree.query(coords, k=k + 1)
    return nbrs_all, nbrs_all[:, 1:]


def compute_spatial_features(
    adata: ad.AnnData,
    score_cols: list[str] | None = None,
    k: int = 6,
) -> None:
    """
    Add spot-level spatial features to adata.obs in-place.

    For each column in score_cols, three features are added:

    {col}_sm          — KNN-smoothed score: mean(self + k neighbors).
                        Reduces measurement noise; used as smoothed input feature.

    {col}_local_grad  — Local gradient: score_i - mean(k neighbors).
                        Positive at spatial peaks (TLS core candidates).
                        Negative at periphery. Stable regardless of cluster size.

    {col}_local_moran — Local Moran's I (LISA): z_i × mean(z_j for j in neighbors),
                        where z = (score - sample_mean) / sample_std.
                        Positive = local cluster of similar values (hot/cold spot).
                        Negative = spatial outlier.

    Args:
        adata:      Single-sample AnnData with obsm['spatial'].
        score_cols: Columns in adata.obs to process. Defaults to
                    tls_composite_score + cxcl13_expression + all score_* columns.
        k:          Number of nearest neighbors.
    """
    if score_cols is None:
        score_cols = (
            [c for c in ['tls_composite_score', 'cxcl13_expression']
             if c in adata.obs.columns]
            + [c for c in adata.obs.columns if c.startswith('score_')]
        )

    coords = adata.obsm['spatial'].astype(float)
    nbrs_all, nbrs_only = _build_knn(coords, k)

    for col in score_cols:
        if col not in adata.obs.columns:
            continue
        vals = adata.obs[col].values.astype(float)

        # Smoothed score (self-inclusive mean). Used to stabilize gradient and
        # Moran's I — not included as a standalone model feature.
        sm = vals[nbrs_all].mean(axis=1)
        adata.obs[f'{col}_sm'] = sm

        # Local gradient on smoothed values: sm_i - mean(sm_j for j in neighbors).
        # Using sm rather than raw vals suppresses single-spot noise spikes.
        adata.obs[f'{col}_local_grad'] = sm - sm[nbrs_only].mean(axis=1)

        # Local Moran's I on smoothed values: z(sm_i) × mean(z(sm_j)).
        # Captures spatial clustering of the smoothed signal.
        mu, sigma = sm.mean(), sm.std()
        if sigma > 1e-8:
            z = (sm - mu) / sigma
            adata.obs[f'{col}_local_moran'] = z * z[nbrs_only].mean(axis=1)
        else:
            adata.obs[f'{col}_local_moran'] = 0.0
