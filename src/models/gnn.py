"""
Hierarchical Graph Neural Network for TLS functional state classification.

Architecture:
  Scale 1  — GAT on spot-level graph  →  DiffPool to niche level
  Scale 2  — GAT on niche-level graph →  DiffPool to region level
  Scale 3  — GAT on region-level graph → global pooling → classification head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, dense_diff_pool
    from torch_geometric.nn.dense import DenseGATConv
    from torch_geometric.utils import to_dense_batch, to_dense_adj
except ImportError as e:
    raise ImportError("Install torch-geometric: pip install torch-geometric") from e


class GATBlock(nn.Module):
    """A single GAT layer with normalization and dropout."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return F.elu(self.dropout(x))


class DiffPoolBlock(nn.Module):
    """
    Differentiable pooling block: learns a soft cluster assignment
    to coarsen a graph from N nodes to k clusters.
    """

    def __init__(self, in_dim: int, k_clusters: int, embed_dim: int = 64):
        super().__init__()
        # Assignment network (produces soft cluster assignments)
        self.assign_net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, k_clusters),
            nn.Softmax(dim=-1),
        )
        # Embedding network (produces cluster features)
        self.embed_net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, adj, mask=None):
        """
        Args:
            x:    (B, N, F) node feature tensor in dense batch format
            adj:  (B, N, N) adjacency matrix
            mask: (B, N)    valid node mask
        Returns:
            x_pooled: (B, k, embed_dim)
            adj_pooled: (B, k, k)
            link_loss: auxiliary assignment regularization loss
            ent_loss:  auxiliary entropy regularization loss
        """
        s = self.assign_net(x)
        z = self.embed_net(x)
        x_pooled, adj_pooled, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        return x_pooled, adj_pooled, link_loss + ent_loss


class TLSFunctionalGNN(nn.Module):
    """
    Hierarchical GAT + DiffPool network for TLS functional state classification.

    Input:
        spot-level graph (Scale 1) in sparse torch_geometric format
    Output:
        per-TLS-region class logits of shape (n_tls_regions, n_classes)
    """

    def __init__(
        self,
        in_dim: int = 76,
        hidden: int = 128,
        n_classes: int = 2,
        n_niche_clusters: int = 20,
        n_region_clusters: int = 5,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # --- Scale 1: spot → niche ---
        self.gat1a = GATBlock(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.gat1b = GATBlock(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.pool1 = DiffPoolBlock(hidden, k_clusters=n_niche_clusters, embed_dim=hidden)

        # --- Scale 2: niche → region ---
        self.gat2a = DenseGATConv(hidden, hidden // heads, heads=heads, concat=True)
        self.gat2b = DenseGATConv(hidden, hidden // heads, heads=heads, concat=True)
        self.pool2 = DiffPoolBlock(hidden, k_clusters=n_region_clusters, embed_dim=hidden)

        # --- Scale 3: region → TLS score ---
        self.gat3 = DenseGATConv(hidden, hidden, heads=1, concat=False)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self._hidden = hidden

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (N, in_dim)  node features
            edge_index: (2, E)       spot-level edges
            batch:      (N,)         batch assignment per node

        Returns:
            logits:     (B, n_classes)  per-TLS prediction
            aux_loss:   scalar auxiliary loss from DiffPool
        """
        # Scale 1: sparse GAT
        h = self.gat1a(x, edge_index)
        h = self.gat1b(h, edge_index)

        # Convert to dense batch for DiffPool
        h_dense, mask = to_dense_batch(h, batch)           # (B, N_max, F)
        adj_dense = to_dense_adj(edge_index, batch)        # (B, N_max, N_max)

        # Pool: spots → niches
        h2, adj2, loss1 = self.pool1(h_dense, adj_dense, mask)   # (B, k1, F)

        # Scale 2: dense GAT on niche graph
        h2 = F.elu(self.gat2a(h2, adj2))
        h2 = F.elu(self.gat2b(h2, adj2))

        # Pool: niches → regions
        h3, adj3, loss2 = self.pool2(h2, adj2)                   # (B, k2, F)

        # Scale 3: dense GAT on region graph
        h3 = F.elu(self.gat3(h3, adj3))

        # Global readout: mean pooling over region nodes
        h_global = h3.mean(dim=1)                                 # (B, F)

        logits = self.classifier(h_global)                        # (B, n_classes)
        aux_loss = loss1 + loss2

        return logits, aux_loss

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
    ) -> torch.Tensor:
        """Return the region-level embedding (before classification head)."""
        h = self.gat1a(x, edge_index)
        h = self.gat1b(h, edge_index)
        h_dense, mask = to_dense_batch(h, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        h2, adj2, _ = self.pool1(h_dense, adj_dense, mask)
        h2 = F.elu(self.gat2a(h2, adj2))
        h2 = F.elu(self.gat2b(h2, adj2))
        h3, adj3, _ = self.pool2(h2, adj2)
        h3 = F.elu(self.gat3(h3, adj3))
        return h3.mean(dim=1)
