"""Load and validate 10x Visium spatial transcriptomics data."""

from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


def load_visium(
    data_dir: str | Path,
    sample_id: str | None = None,
    load_images: bool = True,
    min_counts: int = 500,
    min_genes: int = 200,
) -> ad.AnnData:
    """
    Load a 10x Visium dataset from a standard Space Ranger output directory.
    For the GEO flat-file layout (GSE175540), use load_gse175540() instead.
    """
    data_dir = Path(data_dir)
    import squidpy as sq
    adata = sq.read.visium(
        path=data_dir,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=load_images,
    )
    adata.var_names_make_unique()

    if sample_id is not None:
        adata.obs["sample_id"] = sample_id

    sc.pp.calculate_qc_metrics(adata, inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    print(f"[{sample_id or data_dir.name}] Kept {adata.n_obs}/{n_before} spots after QC")
    adata.layers["counts"] = adata.X.copy()
    return adata


def _load_sample_from_flat_files(
    raw_dir: Path,
    stem: str,
    min_counts: int = 500,
    min_genes: int = 200,
) -> ad.AnnData | None:
    """
    Load one GSE175540 sample from its flat GEO files (all files in one directory,
    named with the pattern: {stem}_filtered_feature_bc_matrix.h5, etc.).

    Reads the h5 count matrix, decompresses the spatial CSV and JSON on the fly,
    then assembles a fully formed AnnData -- no sc.read_visium required.
    """
    h5_path    = raw_dir / f"{stem}_filtered_feature_bc_matrix.h5"
    pos_path   = raw_dir / f"{stem}_tissue_positions_list.csv.gz"
    sf_path    = raw_dir / f"{stem}_scalefactors_json.json.gz"
    tls_path   = raw_dir / f"{stem}_TLS_annotation.csv.gz"

    # --- Require at minimum: counts + positions ---
    if not h5_path.exists():
        print(f"  SKIP {stem}: {h5_path.name} not found")
        return None
    if not pos_path.exists():
        print(f"  SKIP {stem}: {pos_path.name} not found")
        return None

    # --- Count matrix ---
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # --- Tissue positions (spatial coordinates) ---
    with gzip.open(pos_path, "rt") as f:
        first = f.readline()
    # Detect whether file has a header row (newer Space Ranger adds one)
    has_header = not first.split(",")[0].strip().startswith("AAAC")
    with gzip.open(pos_path, "rt") as f:
        pos_df = pd.read_csv(
            f,
            header=0 if has_header else None,
            index_col=0,
        )
    # Standardise column names regardless of Space Ranger version
    if pos_df.shape[1] == 5:
        pos_df.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    elif pos_df.shape[1] == 6:
        # Some versions include barcode as col 0 already consumed by index_col=0
        pos_df.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]

    pos_df.index = pos_df.index.astype(str)
    adata.obs.index = adata.obs.index.astype(str)

    # Keep only barcodes that are in both counts and positions
    shared = adata.obs.index.intersection(pos_df.index)
    if len(shared) == 0:
        print(f"  SKIP {stem}: no barcode overlap between counts and positions")
        return None
    adata = adata[shared].copy()
    pos_df = pos_df.loc[shared]

    # Spatial coordinates in pixel space (row=y, col=x)
    adata.obsm["spatial"] = pos_df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    adata.obs["in_tissue"]  = pos_df["in_tissue"].values
    adata.obs["array_row"]  = pos_df["array_row"].values
    adata.obs["array_col"]  = pos_df["array_col"].values

    # Keep only spots that are under tissue
    adata = adata[adata.obs["in_tissue"] == 1].copy()

    # --- Scale factors (optional, stored in uns) ---
    if sf_path.exists():
        with gzip.open(sf_path, "rt") as f:
            scalefactors = json.load(f)
        adata.uns["spatial"] = {stem: {"scalefactors": scalefactors}}

    # --- QC and filtering ---
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata.obs["sample_id"] = stem
    print(f"  {stem}: {adata.n_obs}/{n_before} spots after QC", end="")

    # --- Author TLS annotations (bonus ground truth) ---
    # Always initialize TLS_2_cat so ad.concat keeps the column even for
    # unannotated samples (avoids fragile post-concat index reconstruction).
    adata.obs["TLS_2_cat"] = None

    if tls_path.exists():
        tls_df = pd.read_csv(tls_path, index_col=0)
        tls_df.index = tls_df.index.astype(str)
        # Normalize column name: some samples use 'TLS', others 'TLS_2_cat'
        if "TLS" in tls_df.columns and "TLS_2_cat" not in tls_df.columns:
            tls_df = tls_df.rename(columns={"TLS": "TLS_2_cat"})
        if "TLS_2_cat" in tls_df.columns:
            shared_tls = adata.obs.index.intersection(tls_df.index)
            if len(shared_tls) > 0:
                adata.obs.loc[shared_tls, "TLS_2_cat"] = tls_df.loc[shared_tls, "TLS_2_cat"]
                n_tls = adata.obs["TLS_2_cat"].notna().sum()
                print(f", {n_tls} TLS-annotated barcodes", end="")
    print()

    # Note: counts layer is added by load_gse175540 after concat to avoid
    # doubling per-sample memory during accumulation.
    return adata


def load_gse175540(
    raw_dir: str | Path,
    max_samples: int | None = None,
    min_counts: int = 500,
    min_genes: int = 200,
) -> ad.AnnData:
    """
    Load GSE175540 (RCC Visium, Immunity 2022) from the flat GEO file layout.

    Handles the flat extraction of GSE175540_RAW.tar where all files for all
    samples live in one directory with the pattern:
        {GSM_ID}_{sample_name}_{file_type}.{ext}

    Args:
        raw_dir: Directory containing the extracted GSE175540_RAW.tar files.
        max_samples: If set, load only this many samples (useful for --debug).
        min_counts: Minimum UMI count per spot.
        min_genes: Minimum detected genes per spot.
    """
    raw_dir = Path(raw_dir)

    h5_files = sorted(raw_dir.glob("GSM*_filtered_feature_bc_matrix.h5"))
    if not h5_files:
        raise FileNotFoundError(
            f"No *_filtered_feature_bc_matrix.h5 files found in {raw_dir}.\n"
            "Download and extract: bash scripts/download_data.sh rcc"
        )

    total = len(h5_files)
    if max_samples is not None:
        h5_files = h5_files[:max_samples]
        print(f"[debug] Loading {len(h5_files)}/{total} samples")

    import gc
    adatas, keys = [], []
    extra_obs_cols: list[pd.DataFrame] = []   # preserve obs columns anndata may drop

    for h5 in h5_files:
        stem = h5.name.replace("_filtered_feature_bc_matrix.h5", "")
        a = _load_sample_from_flat_files(raw_dir, stem, min_counts=min_counts, min_genes=min_genes)
        if a is not None:
            # Stash any annotation columns that exist in only some samples
            # (newer anndata concat drops obs cols absent from some inputs)
            extra_cols = [c for c in a.obs.columns
                          if c not in ("in_tissue", "array_row", "array_col",
                                       "n_genes_by_counts", "log1p_n_genes_by_counts",
                                       "total_counts", "log1p_total_counts",
                                       "pct_counts_in_top_50_genes",
                                       "pct_counts_in_top_100_genes",
                                       "pct_counts_in_top_200_genes",
                                       "pct_counts_in_top_500_genes",
                                       "n_counts", "n_genes", "sample_id",
                                       "TLS_2_cat")]  # handled in _load_sample_from_flat_files
            if extra_cols:
                extra_df = a.obs[extra_cols].copy()
                extra_obs_cols.append(extra_df)

            # Pre-normalize each sample before accumulating -- stores only the
            # log-norm matrix (not raw counts), halving peak memory for 24 samples.
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            a.layers["log_norm"] = a.X   # mark as pre-normalized (shared ref, no copy)
            adatas.append(a)
            keys.append(stem)

    if not adatas:
        raise RuntimeError(
            "No samples loaded. Check that GSE175540_RAW.tar was fully extracted into "
            f"{raw_dir}."
        )

    # join="inner" keeps only genes present in all samples -- avoids var-name
    # inflation from duplicate-gene suffixes and is more memory-efficient.
    combined = ad.concat(adatas, label="sample_id", keys=keys, join="inner")
    combined.obs_names_make_unique()

    # Free individual sample objects immediately after concat
    del adatas
    gc.collect()

    # Re-attach any annotation columns that anndata concat may have dropped
    if extra_obs_cols:
        extra_combined = pd.concat(extra_obs_cols)
        for col in extra_combined.columns:
            if col not in combined.obs.columns:
                combined.obs = combined.obs.join(extra_combined[[col]], how="left")
        for col in extra_combined.columns:
            if col in combined.obs.columns and combined.obs[col].dtype == object:
                combined.obs[col] = combined.obs[col].astype("category")

    # TLS_2_cat is initialized in all per-sample adatas; cast to category for
    # clean h5ad writing (object column with mixed str/None raises TypeError).
    if "TLS_2_cat" in combined.obs.columns:
        combined.obs["TLS_2_cat"] = combined.obs["TLS_2_cat"].astype("category")

    # No counts layer -- downstream normalize_and_reduce will detect pre-normalized
    # data via the log_norm layer and skip re-normalization, using seurat HVG flavor.

    # Clinical metadata (not included in GEO download -- add manually if available)
    meta_path = raw_dir / "clinical_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path, index_col="sample_id")
        combined.obs = combined.obs.join(meta, on="sample_id", how="left")
        print(f"Loaded clinical metadata: {list(meta.columns)}")

    print(f"\nLoaded {combined.obs['sample_id'].nunique()} samples, {combined.n_obs} spots total")
    return combined
