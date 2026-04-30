"""
Build a combined h5ad from GSE203612 Visium samples.

Each sample lives in:
  data/raw/GSE203612/{SAMPLE_NAME}/
    filtered_feature_bc_matrix.h5
    spatial/
      tissue_positions_list.csv
      scalefactors_json.json

Output: data/raw/GSE203612/multicancer_visium.h5ad
  - obs: sample_id, cancer_type
  - obsm: spatial (array_row, array_col pixel coords)
  - X: raw counts (int)

Usage:
  conda run -n tls_spatial python scripts/build_multicancer_h5ad.py
  conda run -n tls_spatial python scripts/build_multicancer_h5ad.py --subset BRCA PDAC
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

DATA_DIR = Path("data/raw/GSE203612")
OUT_PATH = DATA_DIR / "multicancer_visium.h5ad"

# Map sample_name -> cancer_type label
SAMPLE_CANCER = {
    "NYU_BRCA0": "BRCA",
    "NYU_BRCA1": "BRCA",
    "NYU_BRCA2": "BRCA",
    "NYU_GIST1": "GIST",
    "NYU_GIST2": "GIST",
    "NYU_LIHC1": "LIHC",
    "NYU_OVCA1": "OVCA",
    "NYU_OVCA3": "OVCA",
    "NYU_PDAC1": "PDAC",
    "NYU_UCEC3": "UCEC",
}


def load_sample(sample_dir: Path, sample_name: str) -> sc.AnnData:
    h5_path = sample_dir / "filtered_feature_bc_matrix.h5"
    spatial_dir = sample_dir / "spatial"
    pos_path = spatial_dir / "tissue_positions_list.csv"

    if not h5_path.exists():
        raise FileNotFoundError(f"Missing h5: {h5_path}")

    print(f"  Loading {sample_name} ...", end="", flush=True)
    t0 = time.time()

    # Read count matrix directly (no image requirement)
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # Add spatial coordinates from tissue_positions_list.csv
    if pos_path.exists():
        # Column names vary: old Space Ranger has no header; new has header
        try:
            pos = pd.read_csv(pos_path, header=0)
            if "barcode" not in pos.columns:
                # Old format: no header
                pos = pd.read_csv(pos_path, header=None,
                                  names=["barcode", "in_tissue", "array_row", "array_col",
                                         "pxl_row_in_fullres", "pxl_col_in_fullres"])
        except Exception:
            pos = pd.read_csv(pos_path, header=None,
                              names=["barcode", "in_tissue", "array_row", "array_col",
                                     "pxl_row_in_fullres", "pxl_col_in_fullres"])

        pos = pos.set_index("barcode")
        # Keep only in_tissue spots (if column exists)
        if "in_tissue" in pos.columns:
            pos = pos[pos["in_tissue"] == 1]

        # Align barcodes
        common = adata.obs_names.intersection(pos.index)
        adata = adata[common].copy()
        pos = pos.loc[common]

        # Store spatial as pixel coordinates (row, col)
        spatial_coords = pos[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(float)
        adata.obsm["spatial"] = spatial_coords
        adata.obs["array_row"] = pos["array_row"].values
        adata.obs["array_col"] = pos["array_col"].values
    else:
        print(f" [no spatial]", end="", flush=True)
        # Fake spatial coords (zeros) so pipeline doesn't crash
        adata.obsm["spatial"] = np.zeros((adata.n_obs, 2), dtype=float)
    print(f"  {adata.n_obs} spots x {adata.n_vars} genes  ({time.time()-t0:.0f}s)")

    # Annotate
    adata.obs["sample_id"] = sample_name
    adata.obs["cancer_type"] = SAMPLE_CANCER.get(sample_name, "unknown")

    # Make barcodes unique across samples
    adata.obs_names = [f"{sample_name}_{bc}" for bc in adata.obs_names]

    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        nargs="*",
        default=None,
        help="Cancer types to include (e.g. BRCA PDAC). Default: all.",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_PATH),
        help="Output h5ad path",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    # Discover available samples
    available = sorted(
        [d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / "filtered_feature_bc_matrix.h5").exists()]
    )

    if not available:
        print(f"ERROR: No samples found in {DATA_DIR}")
        print("Run: bash scripts/download_gse203612_visium.sh first")
        sys.exit(1)

    print(f"Found {len(available)} samples: {available}")

    # Filter by cancer type if requested
    if args.subset:
        subset_types = set(args.subset)
        available = [s for s in available if SAMPLE_CANCER.get(s, "unknown") in subset_types]
        print(f"  After subset filter ({subset_types}): {len(available)} samples: {available}")

    if not available:
        print("ERROR: No samples remain after filtering")
        sys.exit(1)

    # Load each sample
    adatas = []
    for sample_name in available:
        sample_dir = DATA_DIR / sample_name
        try:
            adata = load_sample(sample_dir, sample_name)
            adatas.append(adata)
        except Exception as e:
            print(f"  WARNING: Failed to load {sample_name}: {e}")

    if not adatas:
        print("ERROR: No samples loaded successfully")
        sys.exit(1)

    print(f"\nConcatenating {len(adatas)} samples ...")
    combined = sc.concat(adatas, join="outer", label=None)
    combined.obs_names_make_unique()

    # Fill NaN counts (from outer join) with 0
    if hasattr(combined.X, "toarray"):
        # sparse - outer join fills missing with 0 already in newer scanpy
        pass
    else:
        combined.X = np.nan_to_num(combined.X, nan=0.0)

    print(f"Combined: {combined.n_obs} spots x {combined.n_vars} genes")
    print(f"Samples : {combined.obs['sample_id'].value_counts().to_dict()}")
    print(f"Cancer  : {combined.obs['cancer_type'].value_counts().to_dict()}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_path} ...")
    t0 = time.time()
    combined.write_h5ad(out_path, compression="gzip")
    print(f"Saved in {time.time()-t0:.0f}s  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
