# TLS Functional Score

Hierarchical GNN (GAT + DiffPool) to classify tertiary lymphoid structures (TLS)
as immunogenic or tolerogenic from 10x Visium spatial transcriptomics data.

**Paper**: *A Hierarchical Spatial Graph Neural Network Resolves Immunogenic and Tolerogenic Tertiary
Lymphoid Structures in Renal Cell Carcinoma*

---

## Overview

TLS are organised immune aggregates found in tumour microenvironments. Their functional
state — immunogenic (active germinal centre, effector T cells) vs. tolerogenic (Tregs,
suppressive myeloid) — predicts patient outcome and ICI response, but cannot be resolved
by bulk sequencing. This repo builds a spatial GNN that operates on spot-level Visium
graphs and classifies TLS functional state at the cluster level.

```
Visium spots  →  GAT × 2  →  DiffPool (k=5 niches)
              →  DenseGAT × 2  →  DiffPool (k=2 regions)
              →  DenseGAT  →  mean-pool  →  MLP  →  {immunogenic, tolerogenic}
```

---

## Repository Structure

```
tls_functional_score/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA of RCC Visium dataset
│   ├── 02_tls_detection.ipynb          # Signature scoring + TLS cluster detection
│   ├── 03_graph_construction.ipynb     # Build per-TLS PyG graphs, save tls_graphs.pt
│   ├── 04_gnn_training.ipynb           # Analysis: evaluation, UMAP, 5-fold CV
│   ├── 05_clinical_validation.ipynb    # IgG correlation, cohort/stage analysis
│   ├── 06_cross_cancer_validation.ipynb # Zero-shot transfer to GSE203612
│   └── 07_tcga_kirc_survival.ipynb     # Bulk z-score TLS signature + KM/Cox on TCGA-KIRC
├── scripts/
│   ├── train.py                        # Training + inference entrypoint (replaces nb04 training cells)
│   ├── preprocess_all.py               # End-to-end preprocessing pipeline
│   ├── download_data.sh                # Download GSE175540 from GEO
│   ├── install_torch.sh                # Install PyTorch + torch-geometric
│   └── build_multicancer_h5ad.py       # Build GSE203612 h5ad from 10x files
├── src/
│   ├── models/
│   │   ├── gnn.py                      # TLSFunctionalGNN (GAT + DiffPool)
│   │   └── transfer.py                 # DomainAdaptedTLSGNN (GRL, unused)
│   ├── training/
│   │   ├── losses.py                   # FocalLoss + TLSTrainingLoss (CE + contrastive + DiffPool aux)
│   │   └── evaluate.py                 # evaluate() -> AUC-ROC, AUC-PR, F1
│   └── tls_detection/                  # Signature scoring + spatial hotspot detection
├── configs/
│   ├── model.yaml                      # Architecture parameters
│   └── training.yaml                   # Training hyperparameters
├── environment.yml                     # Conda environment (Python 3.11 + scanpy stack)
└── .env.example                        # Template for required environment variables
```

---

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate tls_spatial
```

### 2. Install PyTorch + torch-geometric

PyTorch is not in `environment.yml` because the correct build depends on your CUDA version.
The install script auto-detects GPU and installs matching wheels:

```bash
bash scripts/install_torch.sh          # auto-detect
bash scripts/install_torch.sh cpu      # force CPU-only
bash scripts/install_torch.sh cu121    # force CUDA 12.1
```

### 3. Configure paths (scripts only)

`TLS_DATA_ROOT` is only required when running `scripts/train.py` or other scripts
outside the notebook environment:

```bash
cp .env.example .env
# Edit .env: set TLS_DATA_ROOT to the directory containing data/processed/ and data/splits/
```

Or pass it inline:

```bash
TLS_DATA_ROOT=/path/to/data python scripts/train.py
```

---

## Data

Raw data is publicly available on GEO. Processed files (graphs, model checkpoint) are
deposited on Zenodo at **[10.5281/zenodo.19412610]**.

### Option A — Download processed data (recommended)

Download from Zenodo and set `TLS_DATA_ROOT` to the extracted directory:

```
TLS_DATA_ROOT/
├── processed/
│   ├── tls_graphs.pt              # 915 PyG graph objects (output of nb03)
│   └── rcc_visium_labeled.h5ad   # 73,280 spots × 17,943 genes with TLS labels
└── splits/
    ├── arch_config.json           # {in_dim, k_niche, k_region} from nb03
    └── sample_splits.json         # train/val/test sample-level splits
```

### Option B — Reproduce from raw GEO data

```bash
# Download GSE175540 (RCC Visium, Meylan 2022 Immunity)
bash scripts/download_data.sh

# Then follow the Full Pipeline section below (Stage 1 → Stage 2 → Stage 3)
```

---

## Full Pipeline

The complete workflow from raw GEO data to paper figures consists of four stages.
Run everything from the repo root (`tls_functional_score/`) with the conda
environment active and `TLS_DATA_ROOT` set.

### Stage 1 — Preprocessing

```bash
# Full run (all 24 samples, ~24 GB peak RAM — use cluster high-mem node)
python scripts/preprocess_all.py

# Debug run (2 samples, ~5 min on laptop CPU — sufficient for nb01 visualization)
python scripts/preprocess_all.py --debug
```

Writes `data/processed/rcc_visium.h5ad`. Must complete before running any notebook.

### Stage 2 — Data preparation notebooks (required before training)

Run nb01–nb03 **in order**. These are both exploratory and load-bearing: nb03
writes the files that `train.py` requires.

> **nb01 — always run in debug mode for visualization.**
> Loading all 24 samples at once produces overlapping spatial scatter plots that
> are impossible to read. Run `preprocess_all.py --debug` first (2 samples),
> then open nb01. The full `rcc_visium.h5ad` can still be loaded for the
> non-spatial cells (QC metrics, score distributions), but all `sc.pl.spatial`
> and score-on-tissue plots should be run on the debug-preprocessed data.

| Notebook | What it does | Key outputs |
|---|---|---|
| nb01 `01_data_exploration.ipynb` | QC, spatial score visualization, sanity-check vs author `TLS_2_cat` labels | Visual only — no file writes |
| nb02 `02_tls_detection.ipynb` | Tune composite score thresholds (score≥0.20, cxcl13≥0.10), validate AUROC per signature | Visual only — thresholds carried forward to nb03 |
| nb03 `03_graph_construction.ipynb` | Apply tuned thresholds, pseudo-label TLS clusters, build per-TLS PyG subgraphs, define sample splits | `data/processed/tls_graphs.pt`, `data/processed/rcc_visium_labeled.h5ad`, `data/processed/tls_cluster_labels.csv`, `data/splits/arch_config.json`, `data/splits/sample_splits.json`, `data/splits/graph_splits.json` |

### Stage 3 — Training

```bash
TLS_DATA_ROOT=/path/to/data python scripts/train.py
```

Reads `tls_graphs.pt` and `arch_config.json` from `TLS_DATA_ROOT` (produced by nb03).
Outputs saved to `checkpoints/`:

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint (by val AUC-ROC) |
| `tls_predictions.csv` | Per-TLS scores for all 915 clusters |
| `tls_embeddings.pt` | 128-dim GNN embeddings |
| `training_curves.png` | Loss + AUC curves |
| `training_history.csv` | Per-epoch metrics |

Training on a V100 GPU takes ~14 minutes (100 epochs with early stopping).
CPU training is possible but slow (~2-5 hours).

### Stage 4 — Analysis notebooks (paper figures)

Run after training completes. Each notebook reads `TLS_DATA_ROOT` from the environment
and saves figures to `checkpoints/`. Run from the repo root or set `TLS_PROJECT_ROOT`
if launching from elsewhere.

| Notebook | Purpose | Key outputs |
|---|---|---|
| nb04 `04_gnn_training.ipynb` | Evaluation, UMAP, 5-fold CV | `evaluation_plots.png`, `umap_embeddings.png`, `cv_results.csv` |
| nb05 `05_clinical_validation.ipynb` | Clinical validation (IgG, cohort, stage) | `clinical_validation.png` |
| nb06 `06_cross_cancer_validation.ipynb` | Cross-cancer zero-shot transfer (GSE203612) | Spatial TLS maps per sample |
| nb07 `07_tcga_kirc_survival.ipynb` | Bulk TCGA-KIRC survival (KM + Cox) | `km_tcga_kirc.png`, `cox_forest_tcga_kirc.png` |

---

## Key Results

| Metric | Value |
|---|---|
| Val AUC-ROC (main split) | 0.718 |
| 5-fold CV AUC-ROC | 0.507 ± 0.120 |
| Clinical AUC (IgG high vs low) | 0.908 |
| Cross-cancer transfer (GSE203612) | 87 TLS detected, 78% immunogenic |

CV AUC plateau (~0.51) reflects a data limitation: 4 samples hold 94% of tolerogenic
examples, making cross-patient generalisation the bottleneck rather than model capacity.
The clinical AUC (0.908) validates discrimination in the full labeled set.

---

## Reproducibility

All scripts and notebooks use `SEED=42`. Exact numerical results may vary slightly across
hardware and PyTorch versions due to GPU non-determinism and floating-point differences:

- **Val AUC-ROC (0.718) and Clinical AUC (0.908)** are stable across runs.
- **5-fold CV AUC (~0.51)** varies by ±0.02–0.05 between runs due to the small number
  of tolerogenic examples per fold; this is expected and does not affect conclusions.

The Zenodo deposit matches the numbers reported in the paper. Independent reproductions
on different hardware may see small numerical differences but consistent qualitative results.

---

## Citation

```bibtex
@article{tbd,
  title   = {TBD},
  author  = {Peng, Gavin et al.},
  journal = {TBD},
  year    = {2026},
}
```

---

## License

MIT
