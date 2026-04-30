"""
Rebuild rcc_pca_reference.npz from rcc_visium_labeled.h5ad.
Output: data/processed/rcc_pca_reference.npz
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

ROOT   = Path(__file__).resolve().parents[1]
H5AD   = ROOT / 'data/processed/rcc_visium_labeled.h5ad'
OUT    = ROOT / 'data/processed/rcc_pca_reference.npz'
N_PCA  = 50

def main():
    print(f'Loading {H5AD} ...')
    t0 = time.time()
    adata = sc.read_h5ad(H5AD)
    print(f'  {adata.n_obs} spots x {adata.n_vars} genes  ({time.time()-t0:.0f}s)')

    if 'log_norm' in adata.layers:
        adata.X = adata.layers['log_norm']

    hvg_mask = adata.var.get('highly_variable', pd.Series(False, index=adata.var_names))
    hvg_genes = adata.var_names[hvg_mask.values.astype(bool)].tolist()
    print(f'  HVG genes: {len(hvg_genes)}')

    adata_hvg = adata[:, hvg_mask.values.astype(bool)].copy()
    del adata; gc.collect()

    X = adata_hvg.X
    scale_mean = np.asarray(X.mean(axis=0)).flatten().astype(np.float32)
    X_c = X - scale_mean
    if issparse(X_c):
        scale_std = np.sqrt(np.asarray(X_c.power(2).mean(axis=0))).flatten().astype(np.float32)
    else:
        scale_std = np.std(np.asarray(X_c), axis=0).astype(np.float32)
    scale_std = np.where(scale_std > 1e-6, scale_std, 1.0)
    del X_c; gc.collect()

    if 'PCs' in adata_hvg.varm and adata_hvg.varm['PCs'].shape == (len(hvg_genes), N_PCA):
        pca_comps = np.array(adata_hvg.varm['PCs'], dtype=np.float32)
        print(f'  Using stored varm["PCs"] {pca_comps.shape}')
    else:
        print(f'  Fitting PCA (n_comps={N_PCA}) ...')
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, n_comps=N_PCA)
        pca_comps = adata_hvg.varm['PCs'].astype(np.float32)
        print(f'  PCA components: {pca_comps.shape}')

    del adata_hvg; gc.collect()

    np.savez_compressed(
        OUT,
        hvg_genes=np.array(hvg_genes),
        scale_mean=scale_mean,
        scale_std=scale_std,
        pca_components=pca_comps,
    )
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
