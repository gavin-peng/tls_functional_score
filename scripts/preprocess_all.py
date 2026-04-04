"""
End-to-end preprocessing pipeline.

Runs: load → normalize → TLS detection → pseudo-label generation → graph building
for all configured datasets. Saves processed AnnData objects to data/processed/.

Usage:
    python scripts/preprocess_all.py                        # full run
    python scripts/preprocess_all.py --debug                # 2 samples, fast
    python scripts/preprocess_all.py --dataset rcc_visium   # one dataset
    python scripts/preprocess_all.py --dataset rcc_visium --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import anndata as ad

from src.preprocessing.load_visium import load_gse175540
from src.preprocessing.normalize import normalize_and_reduce, ensure_gene_overlap
from src.tls_detection.signature_score import score_tls_signatures, score_tolerogenic_signatures
from src.tls_detection.spatial_correlation import flag_tls_hotspots


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def subsample_adata(adata: ad.AnnData, max_spots: int, seed: int = 42) -> ad.AnnData:
    """Randomly subsample spots/cells for debug speed."""
    if adata.n_obs <= max_spots:
        return adata
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=max_spots, replace=False)
    print(f"  [debug] Subsampled {adata.n_obs} → {max_spots} spots")
    return adata[idx].copy()


def preprocess_rcc_visium(cfg: dict, debug: bool = False) -> None:
    """Preprocess RCC Visium dataset (GSE175540) for clinical validation."""
    dcfg = cfg["datasets"]["rcc_visium"]
    dbg  = cfg.get("debug", {})
    raw_dir  = Path(dcfg["raw_dir"])
    out_path = Path(dcfg["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Preprocessing: GSE175540 RCC Visium{'  [DEBUG]' if debug else ''}")
    print("=" * 60)

    if not raw_dir.exists():
        print(f"ERROR: {raw_dir} not found. Run: bash scripts/download_data.sh rcc")
        return

    max_samples = dbg.get("max_samples", 2) if debug else cfg.get("max_samples")
    adata = load_gse175540(raw_dir, max_samples=max_samples)

    if debug:
        adata = subsample_adata(adata, max_spots=dbg.get("max_spots_per_sample", 500))

    n_hvg = dbg.get("n_hvg", 500) if debug else 3000
    batch_key = None if debug else "sample_id"
    adata = normalize_and_reduce(adata, n_hvg=n_hvg, batch_key=batch_key)
    adata = score_tls_signatures(adata, layer="log_norm")
    adata = score_tolerogenic_signatures(adata, layer="log_norm")

    min_cluster = dbg.get("min_cluster_size", 2) if debug else 3
    adata = flag_tls_hotspots(adata, min_cluster_size=min_cluster)

    adata.write_h5ad(out_path)
    print(f"Saved: {out_path}")
    print(f"  Spots: {adata.n_obs} | Genes: {adata.n_vars} | Samples: {adata.obs['sample_id'].nunique()}")


def preprocess_all(cfg: dict, debug: bool = False) -> None:
    preprocess_rcc_visium(cfg, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess spatial transcriptomics datasets")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "rcc_visium"],
    )
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: process only 2 samples, subsample spots, "
            "use fewer HVGs. Completes in ~5 min on laptop CPU."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of RCC samples loaded (e.g. --max-samples 5). "
             "Useful between --debug and full run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Merge debug section from debug.yaml if --debug
    if args.debug:
        debug_cfg = load_config("configs/debug.yaml")
        cfg["debug"] = debug_cfg.get("debug", {})
        print("[DEBUG] Using debug settings:", cfg["debug"])

    if args.max_samples:
        cfg["max_samples"] = args.max_samples
        print(f"[INFO] Limiting to {args.max_samples} samples")

    if args.dataset == "all":
        preprocess_all(cfg, debug=args.debug)
    elif args.dataset == "rcc_visium":
        preprocess_rcc_visium(cfg, debug=args.debug)
