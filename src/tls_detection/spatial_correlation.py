"""
Spatial autocorrelation and co-localization statistics for TLS detection.

Uses Moran's I to test whether TLS signature scores are spatially clustered,
and spatial cross-correlation to detect co-localization of B and T cell signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree

try:
    import esda
    import libpysal
    HAS_ESDA = True
except ImportError:
    HAS_ESDA = False
    print("WARNING: esda/libpysal not installed. Install via: pip install esda libpysal")


def build_spatial_weights(
    coords: np.ndarray,
    k: int = 6,
    row_standardize: bool = True,
):
    """
    Build a k-nearest-neighbor spatial weights matrix from (x, y) coordinates.

    Args:
        coords: (N, 2) array of spatial coordinates.
        k: Number of nearest neighbors.
        row_standardize: Whether to row-standardize weights.

    Returns:
        libpysal.weights.W object.
    """
    if not HAS_ESDA:
        raise ImportError("Install esda and libpysal: pip install esda libpysal")

    w = libpysal.weights.KNN(coords, k=k)
    if row_standardize:
        w.transform = "R"
    return w


def morans_i(
    values: np.ndarray,
    w,
    permutations: int = 999,
) -> dict[str, float]:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Args:
        values: 1D array of per-spot values.
        w: libpysal spatial weights matrix.
        permutations: Number of random permutations for significance test.

    Returns:
        Dict with keys: 'I', 'p_value', 'z_score'.
    """
    mi = esda.Moran(values, w, permutations=permutations)
    return {"I": mi.I, "p_value": mi.p_sim, "z_score": mi.z_sim}


def spatial_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    w,
    permutations: int = 999,
) -> dict[str, float]:
    """
    Compute bivariate Moran's I (spatial cross-correlation) between two variables.
    Tests whether high x values are surrounded by high y values spatially.

    Used to test B cell x T cell co-localization (hallmark of TLS).
    """
    mi_bv = esda.Moran_BV(x, y, w, permutations=permutations)
    return {"I": mi_bv.I, "p_value": mi_bv.p_sim, "z_score": mi_bv.z_sim}


def compute_tls_spatial_stats(
    adata: ad.AnnData,
    k: int = 6,
    permutations: int = 499,
) -> pd.DataFrame:
    """
    Run spatial autocorrelation tests on all TLS-relevant signature scores.

    Args:
        adata: AnnData with .obsm['spatial'] and score columns in .obs.
        k: k-NN for spatial weights.
        permutations: Permutations for significance testing.

    Returns:
        DataFrame with Moran's I stats per signature + BxT cross-correlation.
    """
    coords = adata.obsm["spatial"]
    w = build_spatial_weights(coords, k=k)

    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    results = []

    for col in score_cols:
        vals = adata.obs[col].values.astype(float)
        if vals.std() < 1e-8:
            results.append({"variable": col, "I": np.nan, "p_value": np.nan, "z_score": np.nan})
            continue
        stats = morans_i(vals, w, permutations=permutations)
        stats["variable"] = col
        results.append(stats)

    # B x T cross-correlation
    if "score_b_cell_core" in adata.obs and "score_t_cell_zone" in adata.obs:
        b = adata.obs["score_b_cell_core"].values.astype(float)
        t = adata.obs["score_t_cell_zone"].values.astype(float)
        if b.std() > 1e-8 and t.std() > 1e-8:
            bv = spatial_cross_correlation(b, t, w, permutations=permutations)
            bv["variable"] = "B_T_crosscorr"
            results.append(bv)

    df = pd.DataFrame(results).set_index("variable")
    df["significant"] = df["p_value"] < 0.05
    return df


def flag_tls_hotspots(
    adata: ad.AnnData,
    composite_score_col: str = "tls_composite_score",
    cxcl13_col: str = "cxcl13_expression",
    k: int = 6,
    score_threshold: float = 0.25,
    cxcl13_threshold: float = 0.5,
    min_cluster_size: int = 3,
    min_compactness: float = 0.0,
) -> ad.AnnData:
    """
    Mark spots as TLS candidates using spatial clustering of high-scoring regions.

    A spot is a TLS candidate if:
      1. Its composite TLS score >= score_threshold AND its spatially-smoothed
         CXCL13 >= cxcl13_threshold (AND logic requires co-occurrence of B cell
         signal + CXCL13, avoiding diffuse immune infiltrate flagged by OR logic).
      2. It belongs to a spatially contiguous cluster of >= min_cluster_size spots.
      3. The cluster compactness >= min_compactness, filtering elongated diffuse
         infiltrate. Compactness = (n_spots x nn_dist^2) / convex_hull_area.
         Real TLS (focal, roughly circular) score ~0.3-0.8; diffuse infiltrate ~0.05.
         Set to 0.0 to disable. Recommended: 0.3.

    Adds 'tls_candidate' (bool) and 'tls_cluster_id' (int, -1 = not TLS) to adata.obs.
    Modifies adata in-place and also returns it.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull

    coords = adata.obsm["spatial"].astype(float)

    score = adata.obs[composite_score_col].values
    cxcl13 = adata.obs.get(cxcl13_col, pd.Series(np.zeros(adata.n_obs))).values

    # Smooth CXCL13 over spatial neighbors
    tree = cKDTree(coords)
    _, nbrs = tree.query(coords, k=k + 1)
    cxcl13_smooth = cxcl13[nbrs].mean(axis=1)

    # AND logic: require both high composite score AND high smoothed CXCL13.
    # OR logic fired on any immune-infiltrated region (CXCL13 from exhausted T cells)
    # producing high recall but very low precision (~0.06). AND logic requires
    # co-occurrence of B cell / plasma / chemokine signal with CXCL13.
    candidate_mask = (score >= score_threshold) & (cxcl13_smooth >= cxcl13_threshold)

    # Connected components among candidate spots
    n = adata.n_obs
    rows, cols = [], []
    for i, neighbors in enumerate(nbrs):
        if candidate_mask[i]:
            for j in neighbors[1:]:  # skip self
                if candidate_mask[j]:
                    rows.append(i)
                    cols.append(j)

    if rows:
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.full(n, -1)

    # Estimate spot area from median nearest-neighbor distance (for compactness)
    if min_compactness > 0:
        nn_dists = nbrs  # already have neighbor indices; re-query for distances
        dists, _ = tree.query(coords, k=2)
        median_nn = float(np.median(dists[:, 1]))
        spot_area = median_nn ** 2

    # Filter by min_cluster_size and optional compactness
    cluster_ids = np.full(n, -1)
    for cid in np.unique(labels):
        members = np.where(labels == cid)[0]
        if not (candidate_mask[members].all() and len(members) >= min_cluster_size):
            continue

        if min_compactness > 0 and len(members) >= 3:
            cluster_coords = coords[members]
            try:
                hull_area = ConvexHull(cluster_coords).volume  # .volume = area in 2D
                compactness = (len(members) * spot_area) / hull_area
                if compactness < min_compactness:
                    continue
            except Exception:
                pass  # degenerate hull (collinear) -- keep the cluster

        cluster_ids[members] = cid

    adata.obs["tls_candidate"] = cluster_ids >= 0
    adata.obs["tls_cluster_id"] = cluster_ids

    n_tls = (adata.obs["tls_cluster_id"] >= 0).sum()
    n_clusters = len(np.unique(cluster_ids[cluster_ids >= 0]))
    print(f"Identified {n_clusters} TLS clusters spanning {n_tls} spots")
    return adata
