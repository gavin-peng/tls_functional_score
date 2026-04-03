# TLS Functional Score

Hierarchical GNN (GAT + DiffPool) to classify tertiary lymphoid structures (TLS)
as immunogenic or tolerogenic from 10x Visium spatial transcriptomics data.

**Paper**: *[title TBD]*

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

### 3. Configure paths

```bash
cp .env.example .env
# Edit .env and set TLS_DATA_ROOT to your local data directory
```

`TLS_DATA_ROOT` must point to a directory containing the processed data files.
See **Data** below for how to obtain them.

---

## Data

Raw data is publicly available on GEO. Processed files (graphs, model checkpoint) are
deposited on Zenodo at **[DOI TBD]**.

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

# Run notebooks 01-03 in order to regenerate all processed files
# (nb03 writes tls_graphs.pt and arch_config.json to TLS_DATA_ROOT)
```

---

## Training

```bash
TLS_DATA_ROOT=/path/to/data python scripts/train.py
```

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

---

## Analysis Notebooks

Run notebooks after training. Each notebook saves figures to `checkpoints/`.

| Notebook | Purpose | Key outputs |
|---|---|---|
| nb04 | Evaluation, UMAP, 5-fold CV | `evaluation_plots.png`, `umap_embeddings.png`, `cv_results.csv` |
| nb05 | Clinical validation (IgG, cohort, stage) | `clinical_validation.png` |
| nb06 | Cross-cancer zero-shot transfer (GSE203612) | Spatial TLS maps per sample |
| nb07 | Bulk TCGA-KIRC survival (KM + Cox) | `km_tcga_kirc.png`, `cox_forest_tcga_kirc.png` |

All notebooks read `TLS_DATA_ROOT` from the environment. Run from the repo root or
set `TLS_PROJECT_ROOT` if launching from elsewhere.

---

## Key Results

| Metric | Value |
|---|---|
| Val AUC-ROC (main split) | 0.718 |
| 5-fold CV AUC-ROC | 0.512 ± 0.144 |
| Clinical AUC (IgG high vs low) | 0.908 |
| Cross-cancer transfer (GSE203612) | 87 TLS detected, 78% immunogenic |

CV AUC plateau (~0.51) reflects a data limitation: 4 samples hold 94% of tolerogenic
examples, making cross-patient generalisation the bottleneck rather than model capacity.
The clinical AUC (0.908) validates discrimination in the full labeled set.

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
