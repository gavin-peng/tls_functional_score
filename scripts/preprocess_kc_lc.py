"""
Preprocess KC/LC Visium data from zenodo_14620362.

Per sample:
  1. Fix coordinate system → obsm['spatial']
  2. Normalize (raw counts → log1p CPM)
  3. Score TLS signatures
  4. Compute spatial features (local gradient, local Moran's I on smoothed values)
  5. Project into RCC PCA reference space → obsm['X_pca']
  6. Standardize metadata columns
  7. Strip X — downstream analysis uses obs features and obsm only

Output: data/processed/kc_lc_visium.h5ad
        Obs + obsm only (no expression matrix).
        Labels kept as-is for post-hoc validation — not used in training.
"""

import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tls_detection.signature_score import score_tls_signatures, score_tolerogenic_signatures
from src.utils.spatial_features import compute_spatial_features

ZENODO   = ROOT / 'data/raw/zenodo_14620362/TLS_VISIUM_USZ/h5ad_preprocessed'
PCA_REF  = ROOT / 'data/processed/rcc_pca_reference.npz'
OUT_PATH = ROOT / 'data/processed/kc_lc_visium.h5ad'

# KC2 excluded: low RNA quality. LC4 excluded: slide misalignment.
# Per Dawo et al. 2025, sec 4.5.3.
SAMPLES = {
    'KC1': 'kc', 'KC3': 'kc',
    'LC1': 'lc', 'LC2': 'lc', 'LC3': 'lc', 'LC5': 'lc',
}

MIN_BLUR_SCORE = 10.0

sc.settings.verbosity = 1

# Load PCA reference once at module level — shared across all samples
_ref = np.load(PCA_REF)
HVG_GENES      = _ref['hvg_genes'].tolist()
SCALE_MEAN     = _ref['scale_mean']
SCALE_STD      = _ref['scale_std']
PCA_COMPONENTS = _ref['pca_components']   # (3000, 50)
HVG_SET        = set(HVG_GENES)


def _project_pca(adata: ad.AnnData) -> np.ndarray:
    """Project adata.X (log-normalized) into RCC PCA space. Returns (n_obs, 50)."""
    gene_map = {g: i for i, g in enumerate(adata.var_names)}
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    X_hvg = np.zeros((adata.n_obs, len(HVG_GENES)), dtype=np.float32)
    for j, g in enumerate(HVG_GENES):
        if g in gene_map:
            X_hvg[:, j] = X[:, gene_map[g]]
    X_scaled = (X_hvg - SCALE_MEAN) / (SCALE_STD + 1e-8)
    return (X_scaled @ PCA_COMPONENTS).astype(np.float32)


def preprocess_sample(sid: str, cancer_type: str) -> ad.AnnData:
    print(f'\n── {sid} ({cancer_type.upper()}) ──────────────────────────')
    adata = ad.read_h5ad(ZENODO / f'{sid}.h5ad')
    print(f'  Loaded: {adata.n_obs} spots × {adata.n_vars} genes')

    # ── 1. Fix coordinates ────────────────────────────────────────────────────
    adata.obsm['spatial'] = adata.obs[['x_pixel', 'y_pixel']].values.astype(float)
    adata.obs.drop(columns=['x_pixel', 'y_pixel', 'x_array', 'y_array'],
                   inplace=True, errors='ignore')

    # ── 2. QC filter ──────────────────────────────────────────────────────────
    if 'blur_score' in adata.obs.columns:
        n_before = adata.n_obs
        adata = adata[adata.obs['blur_score'] >= MIN_BLUR_SCORE].copy()
        if adata.n_obs < n_before:
            print(f'  blur_score filter: removed {n_before - adata.n_obs} spots')

    sc.pp.filter_genes(adata, min_cells=3)
    print(f'  After QC: {adata.n_obs} spots × {adata.n_vars} genes')

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    x_max = float(adata.X.toarray().max() if sp.issparse(adata.X) else adata.X.max())
    if x_max > 20:
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f'  Normalized from raw counts (X.max was {x_max:.0f})')
    else:
        print(f'  Already log-normalized (X.max = {x_max:.2f})')

    # ── 4. TLS signature scoring ───────────────────────────────────────────────
    adata = score_tls_signatures(adata, layer=None)
    adata = score_tolerogenic_signatures(adata, layer=None)
    print(f'  TLS scores computed. composite range: '
          f'[{adata.obs["tls_composite_score"].min():.3f}, '
          f'{adata.obs["tls_composite_score"].max():.3f}]')

    # ── 5. Spatial features ───────────────────────────────────────────────────
    compute_spatial_features(adata)
    feat_cols = [c for c in adata.obs.columns
                 if c.endswith(('_sm', '_local_grad', '_local_moran'))]
    print(f'  Spatial features added: {len(feat_cols)} columns')

    # ── 6. PCA projection ─────────────────────────────────────────────────────
    adata.obsm['X_pca'] = _project_pca(adata)
    n_missing = sum(1 for g in HVG_GENES if g not in set(adata.var_names))
    print(f'  PCA projected: {n_missing} HVGs zero-filled. '
          f'Range [{adata.obsm["X_pca"].min():.2f}, {adata.obsm["X_pca"].max():.2f}]')

    # ── 7. Standardize metadata ───────────────────────────────────────────────
    adata.obs['sample_id']   = sid
    adata.obs['cancer_type'] = cancer_type
    adata.obs['dataset']     = 'kclc'

    if 'manual_anno_tls' in adata.obs.columns:
        adata.obs['tls_label_fine']   = adata.obs['manual_anno_tls']
    if 'ground_truth' in adata.obs.columns:
        adata.obs['tls_label_coarse'] = adata.obs['ground_truth']

    # Exclude lymph node contamination spots (LN label in LC3)
    if 'tls_label_fine' in adata.obs.columns:
        n_before = adata.n_obs
        adata = adata[adata.obs['tls_label_fine'] != 'LN'].copy()
        if adata.n_obs < n_before:
            print(f'  LN exclusion: removed {n_before - adata.n_obs} spots')

    drop_cols = ['manual_anno', 'manual_anno_tls',
                 'aestetik_manual_anno', 'aestetik_manual_anno_tls',
                 'ground_truth', 'in_tissue']
    adata.obs.drop(columns=drop_cols, inplace=True, errors='ignore')

    adata.obs_names = [f'{sid}_{bc}' for bc in adata.obs_names]

    # ── 8. Strip expression matrix ────────────────────────────────────────────
    # Downstream analysis uses obs features and obsm['X_pca'] only.
    # Dropping X here means concat never holds multiple full matrices in RAM.
    obs_out  = adata.obs.copy()
    obsm_out = {k: v.copy() for k, v in adata.obsm.items()}
    return ad.AnnData(obs=obs_out, obsm=obsm_out)


def main():
    print(f'PCA reference: {len(HVG_GENES)} HVGs, {PCA_COMPONENTS.shape[1]} components')

    tmp_dir = Path(tempfile.mkdtemp(prefix='kclc_preprocess_'))
    tmp_paths = []

    for sid, cancer_type in SAMPLES.items():
        adata = preprocess_sample(sid, cancer_type)
        tmp_path = tmp_dir / f'{sid}.h5ad'
        adata.write_h5ad(tmp_path)
        tmp_paths.append(tmp_path)
        del adata

    # ── Combine ── obs+obsm only; no expression matrix to hold in RAM ─────────
    print('\n── Combining all samples ────────────────────────────────────')
    adatas = [ad.read_h5ad(p) for p in tmp_paths]
    combined = ad.concat(adatas, join='outer', merge='same')
    del adatas
    for p in tmp_paths:
        p.unlink()
    tmp_dir.rmdir()

    print(f'Combined: {combined.n_obs} spots')
    print(f'Samples:  {combined.obs["sample_id"].nunique()}')
    print(f'Cancer types: {dict(combined.obs["cancer_type"].value_counts())}')

    for col in ['tls_label_fine', 'tls_label_coarse']:
        if col in combined.obs.columns:
            print(f'\n{col}:')
            print(combined.obs.groupby('cancer_type', observed=True)[col]
                  .value_counts().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(OUT_PATH)
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()
