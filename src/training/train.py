"""
Main training loop for the domain-adaptive TLS functional state GNN.

Usage:
    python src/training/train.py                    # full training
    python src/training/train.py --debug            # fast CPU smoke-test
    python src/training/train.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError("Install torch-geometric")

from src.models.gnn import TLSFunctionalGNN
from src.models.transfer import DomainAdaptedTLSGNN, compute_grl_alpha
from src.training.losses import TLSTrainingLoss
from src.training.evaluate import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_debug_config(base: dict, debug: dict) -> dict:
    """Recursively merge debug overrides into base config."""
    merged = base.copy()
    for k, v in debug.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_debug_config(merged[k], v)
        else:
            merged[k] = v
    return merged


def make_synthetic_graphs(n: int, in_dim: int, n_nodes: int = 50) -> list:
    """
    Generate tiny random graphs for a pure pipeline smoke-test when no
    real processed data exists yet (useful the very first time --debug is run).
    """
    from torch_geometric.data import Data
    graphs = []
    for i in range(n):
        n_spots = random.randint(n_nodes // 2, n_nodes)
        x = torch.randn(n_spots, in_dim)
        # Random spatial k-NN edges (k=4)
        rows, cols = [], []
        for node in range(n_spots):
            neighbors = random.sample([j for j in range(n_spots) if j != node], min(4, n_spots - 1))
            for nb in neighbors:
                rows.append(node)
                cols.append(nb)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        y = torch.tensor(random.choice([0, 1]), dtype=torch.long).unsqueeze(0)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


def load_graph_dataset(data_root: str, split: str, max_graphs: int | None = None):
    """Load a pre-built graph dataset (.pt file). Falls back to synthetic data in debug."""
    path = Path(data_root) / f"{split}_graphs.pt"
    if path.exists():
        dataset = torch.load(path, weights_only=False)
        if max_graphs is not None:
            dataset = dataset[:max_graphs]
        return dataset
    return None


def train_one_epoch(
    model: DomainAdaptedTLSGNN,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TLSTrainingLoss,
    device: torch.device,
    step: int,
    total_steps: int,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {k: 0.0 for k in ["total", "task", "domain", "contrastive", "aux"]}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        alpha = compute_grl_alpha(step, max(total_steps, 1))
        model.set_grl_alpha(alpha)

        optimizer.zero_grad()
        out = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            return_domain_logits=True,
        )

        losses = loss_fn(
            task_logits=out["task_logits"],
            task_labels=batch.y,
            embeddings=model.backbone.get_embeddings(batch.x, batch.edge_index, batch.batch),
            domain_logits=out.get("domain_logits"),
            domain_labels=batch.domain_label if hasattr(batch, "domain_label") else None,
            aux_loss=out.get("aux_loss"),
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1
        step += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}, step


def train(
    config_path: str = "configs/training.yaml",
    model_config_path: str = "configs/model.yaml",
    debug_config_path: str = "configs/debug.yaml",
    data_root: str = "data/processed",
    debug: bool = False,
) -> DomainAdaptedTLSGNN:

    # Load and optionally merge debug overrides
    tcfg = load_config(config_path)
    mcfg = load_config(model_config_path)
    if debug:
        dcfg = load_config(debug_config_path)
        tcfg = merge_debug_config(tcfg, dcfg)
        mcfg = merge_debug_config(mcfg, dcfg)
        print("=" * 50)
        print("DEBUG MODE — small model, 3 epochs, CPU-safe")
        print("=" * 50)

    tcfg = tcfg["training"]
    set_seed(tcfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load graph datasets ---
    max_graphs = dcfg["debug"]["max_graphs"] if debug else None
    train_set = load_graph_dataset(data_root, "train", max_graphs=max_graphs)
    val_set   = load_graph_dataset(data_root, "val",   max_graphs=max(4, (max_graphs or 20) // 4))

    in_dim = mcfg["gnn"]["scale1_in_dim"]

    if train_set is None:
        if debug:
            print(f"\nNo processed graphs found at {data_root}/train_graphs.pt")
            print("Generating synthetic graphs for pipeline smoke-test...\n")
            train_set = make_synthetic_graphs(16, in_dim, n_nodes=60)
            val_set   = make_synthetic_graphs(8,  in_dim, n_nodes=60)
        else:
            raise FileNotFoundError(
                f"{data_root}/train_graphs.pt not found.\n"
                "Run: python scripts/preprocess_all.py first.\n"
                "Or use --debug for a synthetic smoke-test."
            )

    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=tcfg["batch_size"], shuffle=False)
    print(f"Train graphs: {len(train_set)} | Val graphs: {len(val_set)}")

    # --- Build model ---
    backbone = TLSFunctionalGNN(
        in_dim=in_dim,
        hidden=mcfg["gnn"]["scale1_hidden"],
        n_classes=mcfg["gnn"]["n_classes"],
        n_niche_clusters=mcfg["graph"]["n_niches"],
        n_region_clusters=mcfg["graph"]["n_region_nodes"],
    ).to(device)

    use_da = mcfg["transfer"].get("use_domain_adaptation", True)
    model = DomainAdaptedTLSGNN(
        backbone=backbone,
        embedding_dim=mcfg["gnn"]["scale1_hidden"],
        n_domains=mcfg["transfer"]["n_domains"],
        grl_alpha=mcfg["transfer"]["gradient_reversal_alpha"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(tcfg["epochs"], 1))
    loss_fn = TLSTrainingLoss(
        task_weight=tcfg["task_loss_weight"],
        domain_weight=tcfg.get("domain_loss_weight", 0.1),
        contrastive_weight=tcfg.get("contrastive_loss_weight", 0.1),
    )

    # --- Training loop ---
    best_val_auc = 0.0
    patience_counter = 0
    step = 0
    total_steps = tcfg["epochs"] * len(train_loader)
    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(tcfg["epochs"]):
        train_losses, step = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, step, total_steps
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{tcfg['epochs']} | "
            f"loss={train_losses['total']:.4f} "
            f"(task={train_losses['task']:.4f}) | "
            f"val_auc={val_metrics['auc_roc']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  → Checkpoint saved (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= tcfg["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nDone. Best val AUC: {best_val_auc:.4f}")
    if debug:
        print("\nPipeline smoke-test passed. Run without --debug on GPU for real training.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TLS functional state GNN")
    parser.add_argument("--config",       default="configs/training.yaml")
    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument("--debug_config", default="configs/debug.yaml")
    parser.add_argument("--data_root",    default="data/processed")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug mode: tiny model, 3 epochs, synthetic data fallback. "
            "Runs end-to-end in ~2 min on CPU. Use to verify the pipeline "
            "before submitting a GPU training job."
        ),
    )
    args = parser.parse_args()

    train(
        config_path=args.config,
        model_config_path=args.model_config,
        debug_config_path=args.debug_config,
        data_root=args.data_root,
        debug=args.debug,
    )
