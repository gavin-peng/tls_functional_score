"""
Domain adaptation for cross-platform transfer learning (CosMx → Visium).

Uses a Gradient Reversal Layer (GRL) so the shared GNN backbone learns
features that are platform-agnostic while still being useful for TLS
functional state classification.

Reference: Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by
Backpropagation"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses gradients during backprop with scaling factor alpha."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha


class DomainClassifier(nn.Module):
    """
    Predicts the data platform (domain) from GNN embeddings.
    Combined with GRL, this forces the backbone to be platform-blind.
    """

    def __init__(self, in_dim: int, n_domains: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainAdaptedTLSGNN(nn.Module):
    """
    Full domain-adaptive model:
        shared backbone (TLSFunctionalGNN)  →  task classifier
                                            ↘  GRL  →  domain classifier

    During training:
      - Task loss   (cross-entropy on labeled CosMx TLS regions)
      - Domain loss (cross-entropy on platform labels, CosMx=0, Visium=1)
        with reversed gradients → backbone becomes platform-invariant

    At inference on Visium:
      - Only the backbone + task classifier are used.
    """

    def __init__(
        self,
        backbone,            # TLSFunctionalGNN instance
        embedding_dim: int = 128,
        n_domains: int = 2,
        grl_alpha: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.domain_clf = DomainClassifier(embedding_dim, n_domains=n_domains)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: torch.LongTensor,
        return_domain_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x, edge_index, batch: Graph inputs.
            return_domain_logits: If True, also return domain predictions.

        Returns:
            Dict with 'task_logits' (and optionally 'domain_logits', 'aux_loss').
        """
        task_logits, aux_loss = self.backbone(x, edge_index, batch)
        out = {"task_logits": task_logits, "aux_loss": aux_loss}

        if return_domain_logits:
            embeddings = self.backbone.get_embeddings(x, edge_index, batch)
            domain_logits = self.domain_clf(self.grl(embeddings))
            out["domain_logits"] = domain_logits

        return out

    def set_grl_alpha(self, alpha: float):
        """Gradually increase GRL alpha during training (standard practice)."""
        self.grl.set_alpha(alpha)


def compute_grl_alpha(
    current_step: int,
    total_steps: int,
    alpha_max: float = 1.0,
) -> float:
    """
    Schedule GRL alpha to increase smoothly from 0 → alpha_max.
    Formula from Ganin & Lempitsky (2015).
    """
    p = current_step / total_steps
    return alpha_max * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0)
