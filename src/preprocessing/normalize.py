"""Normalization, HVG selection, and dimensionality reduction."""

import gc
import numpy as np
import anndata as ad
import scanpy as sc


def normalize_and_reduce(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    n_hvg: int = 3000,
    n_pca: int = 50,
    batch_key: str | None = "sample_id",
) -> ad.AnnData:
    """
    Standard preprocessing: normalize -> log1p -> HVG -> scale -> PCA.

    Assumes .layers['counts'] holds the raw counts.

    Args:
        adata: AnnData object with raw counts in .layers['counts'].
        target_sum: Library size normalization target.
        n_hvg: Number of highly variable genes to select.
        n_pca: Number of PCA components.
        batch_key: Column in .obs for batch-aware HVG selection.

    Returns:
        adata with .obsm['X_pca'] added; .X set to log-normalized values.
    """
    if "log_norm" in adata.layers:
        # Data was pre-normalized during loading (memory-efficient path for large datasets).
        # X is already log-normalized; use seurat HVG which works on log-norm data.
        adata.X = adata.layers["log_norm"]
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            batch_key=batch_key,
            flavor="seurat",
        )
    else:
        # Standard path: normalize from raw counts.
        # .X is raw counts; layers["counts"] is a separate backup copy.
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        adata.layers["log_norm"] = adata.X  # share ref -- X not modified after this
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_hvg,
            batch_key=batch_key,
            flavor="seurat_v3",
            layer="counts",
        )
        del adata.layers["counts"]
        gc.collect()

    # Scale + PCA on the HVG subset only -- avoids densifying the full 33k-gene matrix
    # (70k spots x 33k genes x float32 ~= 9 GB; HVG subset is ~11x smaller)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=n_pca)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    del adata_hvg
    gc.collect()

    print(
        f"Normalized {adata.n_obs} cells/spots | "
        f"{adata.var['highly_variable'].sum()} HVGs | "
        f"PCA: {adata.obsm['X_pca'].shape}"
    )
    return adata


def ensure_gene_overlap(
    adata_source: ad.AnnData,
    adata_target: ad.AnnData,
    min_overlap: int = 200,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Subset both AnnData objects to their shared genes.
    Used to align CosMx (960 genes) with Visium (33k genes) for transfer learning.
    """
    shared = adata_source.var_names.intersection(adata_target.var_names)
    if len(shared) < min_overlap:
        raise ValueError(
            f"Only {len(shared)} shared genes between datasets "
            f"(minimum required: {min_overlap}). "
            "Check that gene symbols use the same naming convention."
        )
    print(f"Shared genes between platforms: {len(shared)}")
    return adata_source[:, shared].copy(), adata_target[:, shared].copy()
