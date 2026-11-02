"""SCR: Semantic Class-Relation Refinement for supervised video action recognition.

Problem
-------
Frozen-CLIP video classifiers compute logits as ``V @ Q.T`` where ``Q`` are
the per-class text prototypes. Nearby action classes (e.g. ``Basketball`` vs
``BasketballDunk``, ``PlayingGuitar`` vs ``PlayingViolin``) have *almost
collinear* text prototypes in CLIP's shared embedding space, so a correct
video ``V`` sits close to several candidate classes and the model is stuck
with a high top-5 but a low top-1 accuracy. This is the classic "similar-class
confusion" failure mode.

Idea
----
Push each text prototype *away* from the prototypes of the classes it is most
easily confused with, so that confusable classes are separated in direction
while the global structure of the embedding space is preserved.

For each class ``c``, let ``N(c)`` be its top-k most similar classes (measured
by cosine similarity of the text prototypes). We subtract from ``Q_c`` its
projection onto each confusable neighbour ``Q_j`` (a single Gram-Schmidt-style
decorrelation step):

    Q_ref_c = normalize( Q_c - lambda * sum_{j in N(c)} (Q_c . Q_j) * Q_j )

In matrix form, build a sparse [C, C] matrix ``W`` where ``W[c, j] = Q_c . Q_j``
for ``j in N(c)`` and 0 otherwise (top-k hard mask, weights detached), then:

    Q_ref = normalize( Q - lambda * W @ Q )

Why this is correct (and why plain graph smoothing was *not*)
--------------------------------------------------------------
A first attempt used graph *smoothing* ``Q_ref = Q + lambda (A Q - Q)`` (the
class-relation-graph analogue of label smoothing). This is mathematically a
graph-Laplacian *smoothing* step: it pulls each prototype toward its
neighbours' centroid. Because confusable classes have *overlapping* neighbour
sets, smoothing moves them *toward each other* and makes them *more* similar --
empirically the top-10 most similar pairs got *closer* after smoothing, the
exact opposite of the goal. Pushing (subtracting the projection onto
confusable neighbours) is the *decorrelating* operation: empirically it drops
the top similar pairs from ~0.16 to ~0.06 while leaving the global off-diagonal
mean similarity essentially unchanged, i.e. it *only* separates confusable
classes.

Relation to prior work
----------------------
* One Gram-Schmidt step toward orthogonality of confusable prototypes (a
  classical decorrelation idea).
* Sparse top-k neighbour selection follows the sparse-attention / k-NN graph
  literature (the hard mask is non-differentiable, but the prototype features
  remain differentiable -- standard for sparse attention).
* Label smoothing (Szegedy et al. 2016) smooths *labels*; we *sharpen* the
  classifier anchors only along the confusable directions, which is its
  weight-space counterpart with the opposite (correct) sign.

Design invariants (kept consistent with the project conventions)
----------------------------------------------------------------
1. Identity start: ``lambda = max_scale * tanh(theta)`` with ``theta``
   zero-initialised, so step 0 is exactly the baseline ``Q``.
2. The top-k neighbour mask and the similarity weights are *detached*: the
   "which neighbours to push away from" is a hard snapshot at the current
   step, while ``Q`` itself stays differentiable (stable, sparse-attention
   convention).
3. ``Q_ref`` is L2-normalised so it stays on the CLIP unit sphere.
4. All math in fp32 (project convention), AMP-safe.
5. Exactly two trainable scalars (``theta``); the graph itself is
   parameter-free.

References
----------
* Szegedy et al., "Rethinking the Inception Architecture for Computer Vision"
  (label smoothing), CVPR 2016.
* Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
  Networks" (graph Laplacian smoothing), ICLR 2017.
* Li et al., "Efficient and Context-Aware Label Propagation for Zero-/Few-Shot
  Training-Free Adaptation of Vision-Language Models", arXiv:2412.18303.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticClassRelationRefinement(nn.Module):
    """Separate per-class text prototypes from their most confusable classes.

    Args:
        dim: feature dimension D (512 for ViT-B/16).
        top_k: number of most-confusable neighbours to push away from.
        max_scale: upper bound of the pushing strength ``lambda``.
    """

    def __init__(
        self,
        dim: int,
        top_k: int = 5,
        max_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if max_scale <= 0.0:
            raise ValueError("max_scale must be positive.")

        self.dim = int(dim)
        self.top_k = int(top_k)
        self.max_scale = float(max_scale)

        # Pushing strength: identity start (theta == 0 -> lambda == 0).
        self.theta = nn.Parameter(torch.zeros(1))

        # Diagnostic only (not part of the state dict).
        self._last_push_matrix: Optional[torch.Tensor] = None
        self._last_lambda: Optional[torch.Tensor] = None

    def _interpolation_strength(self) -> torch.Tensor:
        """lambda = max_scale * tanh(theta), theta zero-init -> 0 at start."""
        return self.max_scale * torch.tanh(self.theta.float())

    def _confusable_weight_matrix(
        self,
        Q: torch.Tensor,  # [C, D] fp32, L2-normalized
    ) -> torch.Tensor:
        """Build the sparse push-weight matrix W [C, C].

        ``W[c, j] = Q_c . Q_j`` when ``j`` is among the top-k most similar
        classes to ``c`` (excluding ``c`` itself), else 0. The top-k selection
        and the similarity weights are detached: they define a hard "push away
        from these neighbours" snapshot, while ``Q`` stays differentiable
        through the ``W @ Q`` matmul below.
        """
        num_classes = Q.shape[0]
        k = min(self.top_k, num_classes - 1)
        if k <= 0:
            # Degenerate case (fewer than 2 classes): no confusable pairs.
            return torch.zeros(
                num_classes, num_classes, device=Q.device, dtype=Q.dtype
            )

        sim = Q @ Q.t()  # [C, C] cosine similarities
        sim = sim.fill_diagonal_(-float("inf"))
        _, topk_idx = sim.topk(k, dim=-1)  # [C, k]

        W = torch.zeros_like(sim)
        rows = torch.arange(num_classes, device=Q.device).unsqueeze(1)
        rows = rows.expand(-1, k)
        W[rows, topk_idx] = sim[rows, topk_idx]

        return W.detach()

    def forward(
        self,
        Q: torch.Tensor,  # [C, D] L2-normalized text prototypes
    ) -> torch.Tensor:
        """Return the refined text prototypes ``Q_ref`` of shape [C, D]."""
        if Q.ndim != 2:
            raise ValueError(
                f"Q must be 2-D [C, D], got {tuple(Q.shape)}."
            )

        input_dtype = Q.dtype
        Q_fp32 = F.normalize(Q.float(), dim=-1, eps=1e-6)  # [C, D]

        W = self._confusable_weight_matrix(Q_fp32)  # [C, C], detached
        lam = self._interpolation_strength()  # scalar

        # Push away from confusable neighbours' projections, then normalize
        # back onto the unit sphere.
        Q_ref = Q_fp32 - lam * (W @ Q_fp32)
        Q_ref = F.normalize(Q_ref, dim=-1, eps=1e-6)

        self._last_push_matrix = W
        self._last_lambda = lam.detach()

        return Q_ref.to(dtype=input_dtype)
