"""
Loss functions for TLS functional state training.

Combines:
  - Task loss: cross-entropy on functional state labels
  - Domain loss: adversarial domain confusion (platform invariance)
  - Contrastive loss: TLS regions of the same class cluster in embedding space
  - Auxiliary DiffPool loss: from graph pooling regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy (high-confidence) examples so training focuses on
    hard, misclassified ones — exactly the tolerogenic TLS problem.
    alpha_t (class_weights) handles frequency imbalance; gamma handles
    example hardness.  Lin et al. (2017) RetinaNet, ICCV.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # per-class alpha (same API as CrossEntropyLoss)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample weighted CE (no reduction)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        # Probability of the correct class (detached — only used for weighting)
        with torch.no_grad():
            p_t = torch.softmax(logits, dim=1)[
                torch.arange(len(targets), device=logits.device), targets
            ]
        return ((1.0 - p_t) ** self.gamma * ce).mean()


class TLSTrainingLoss(nn.Module):
    """
    Combined loss for domain-adaptive TLS functional state classification.
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        domain_weight: float = 0.1,
        contrastive_weight: float = 0.1,
        aux_weight: float = 0.01,
        temperature: float = 0.07,
        n_classes: int = 2,
        class_weights: torch.Tensor | None = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.task_weight        = task_weight
        self.domain_weight      = domain_weight
        self.contrastive_weight = contrastive_weight
        self.aux_weight         = aux_weight
        self.temperature        = temperature

        if use_focal:
            self.task_loss_fn = FocalLoss(gamma=focal_gamma, weight=class_weights)
        else:
            self.task_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        task_logits: torch.Tensor,
        task_labels: torch.Tensor,
        embeddings: torch.Tensor,
        domain_logits: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        aux_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            task_logits:   (N, n_classes) — TLS functional state predictions
            task_labels:   (N,) — ground truth labels {0, 1}
            embeddings:    (N, D) — GNN embeddings for contrastive loss
            domain_logits: (N, n_domains) — platform predictions (optional)
            domain_labels: (N,) — platform ground truth {0=CosMx, 1=Visium}
            aux_loss:      scalar — DiffPool auxiliary regularization

        Returns:
            Dict with 'total', 'task', 'domain', 'contrastive', 'aux' losses.
        """
        # Task loss (only on labeled samples: label != -1)
        labeled = task_labels >= 0
        if labeled.sum() == 0:
            task_loss = torch.tensor(0.0, device=task_logits.device)
        else:
            task_loss = self.task_loss_fn(task_logits[labeled], task_labels[labeled])

        # Domain adversarial loss
        domain_loss = torch.tensor(0.0, device=task_logits.device)
        if domain_logits is not None and domain_labels is not None:
            domain_loss = self.domain_loss_fn(domain_logits, domain_labels)

        # Supervised contrastive loss (NT-Xent style)
        contrastive_loss = torch.tensor(0.0, device=task_logits.device)
        if labeled.sum() >= 4:
            contrastive_loss = self._supervised_contrastive(
                embeddings[labeled], task_labels[labeled]
            )

        # DiffPool auxiliary loss
        _aux = aux_loss if aux_loss is not None else torch.tensor(0.0, device=task_logits.device)

        total = (
            self.task_weight        * task_loss
            + self.domain_weight    * domain_loss
            + self.contrastive_weight * contrastive_loss
            + self.aux_weight       * _aux
        )

        return {
            "total":       total,
            "task":        task_loss.detach(),
            "domain":      domain_loss.detach(),
            "contrastive": contrastive_loss.detach(),
            "aux":         _aux.detach(),
        }

    def _supervised_contrastive(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supervised NT-Xent contrastive loss.
        Pulls same-class TLS embeddings together, pushes different-class apart.
        """
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        # Mask: positives are same-class pairs (excluding diagonal)
        n = labels.size(0)
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=labels.device)
        pos_mask = label_eq & diag_mask

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Log-softmax over all non-self pairs, sum over positives
        exp_sim = torch.exp(sim_matrix) * diag_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(log_prob * pos_mask).sum() / pos_mask.sum()
        return loss
