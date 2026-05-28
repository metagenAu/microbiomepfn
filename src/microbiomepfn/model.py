"""
Permutation-equivariant axial transformer for microbiome amplicon data.

Architecture sketch (one forward pass on a single draw):

  Input:
    cell_feats:   (N, T, d_cell)
    taxon_feats:  (T, d_tax)
    sample_feats: (N, d_samp)

  Embed each independently:
    h_cell  = Linear(d_cell  -> d) + LayerNorm
    h_tax   = Linear(d_tax   -> d) + LayerNorm
    h_samp  = Linear(d_samp  -> d) + LayerNorm

  Combine into a token grid:
    H[i,t]  = h_cell[i,t] + h_tax[t] + h_samp[i]                 # (N, T, d)

  L axial blocks:
    each block does:
       taxon-axis ISAB per sample (with masking for padded taxa)
       sample-axis MHA per taxon  (no positional embeddings, fully permutation-equivariant)

  Heads:
    count head:   per (i,t)   -> negative-binomial params (log_mu, log_theta)
    y head:       per sample  -> attention-pool across taxa then MLP
    effect head:  per taxon   -> aggregates sample-level info via attention-pool

Design commitments:
  - No parameter is indexed by taxon identity or column position.
  - No positional embeddings on either axis.
  - All weights are shared across the taxon axis and across the sample axis.
  - Phylogenetic info enters only through per-taxon features.

This is what makes the model generalize to (a) new taxa it has never seen, and
(b) datasets of arbitrary T at deployment time.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Multi-head attention building block (MAB)
# ===========================================================================
class MAB(nn.Module):
    """Multi-head attention block: out = LN(X + MHA(X, Y, Y)) + FFN."""
    def __init__(self, d: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ffn_mult, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Q: (B, Lq, d)  K: (B, Lk, d)
        q = self.ln_q(Q)
        k = self.ln_kv(K)
        a, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        x = Q + self.drop(a)
        x = x + self.drop(self.ffn(self.ln_ff(x)))
        return x


# ===========================================================================
# Induced Set Attention Block (ISAB)
# Keeps taxon-axis attention O(T * m) instead of O(T^2).
# ===========================================================================
class ISAB(nn.Module):
    def __init__(self, d: int, n_heads: int, m_inducing: int = 64,
                 ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m_inducing, d) * 0.02)
        self.mab1 = MAB(d, n_heads, ffn_mult, dropout)  # I attends to X
        self.mab2 = MAB(d, n_heads, ffn_mult, dropout)  # X attends to I'

    def forward(self, X: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # X: (B, T, d). key_padding_mask: (B, T) bool (True = pad to ignore).
        B = X.size(0)
        I = self.I.expand(B, -1, -1)                              # (B, m, d)
        I_prime = self.mab1(I, X, key_padding_mask=key_padding_mask)
        return self.mab2(X, I_prime, key_padding_mask=None)


# ===========================================================================
# Axial block: taxon-axis ISAB + sample-axis MHA
# ===========================================================================
class AxialBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, m_inducing: int = 64,
                 ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # Taxon-axis: batch dimension = N (samples), set dimension = T (taxa)
        self.tax_axis = ISAB(d, n_heads, m_inducing, ffn_mult, dropout)
        # Sample-axis: batch dimension = T (taxa), set dimension = N (samples)
        self.sam_axis = MAB(d, n_heads, ffn_mult, dropout)

    def forward(self, H: torch.Tensor,
                taxon_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # H: (N, T, d)
        # Taxon-axis: treat each sample as one "batch element"
        H = self.tax_axis(H, key_padding_mask=taxon_pad_mask)     # (N, T, d)

        # Sample-axis: transpose to (T, N, d), full MHA over N
        Ht = H.transpose(0, 1).contiguous()                       # (T, N, d)
        # Self-attention over samples
        Ht = self.sam_axis(Ht, Ht, key_padding_mask=None)         # (T, N, d)
        return Ht.transpose(0, 1).contiguous()                    # (N, T, d)


# ===========================================================================
# Attention-pooling head
# ===========================================================================
class AttnPool(nn.Module):
    """Pool a (B, L, d) tensor to (B, d) using a learned query."""
    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, X: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = X.size(0)
        q = self.q.expand(B, -1, -1)
        out, _ = self.attn(q, self.ln(X), self.ln(X),
                           key_padding_mask=key_padding_mask, need_weights=False)
        return out.squeeze(1)  # (B, d)


# ===========================================================================
# The full model
# ===========================================================================
class MicrobiomePFN(nn.Module):
    def __init__(self,
                 d_cell_in: int = 6,
                 d_tax_in: int = 25,
                 d_samp_in: int = 70,
                 d: int = 256,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 m_inducing: int = 64,
                 ffn_mult: int = 4,
                 dropout: float = 0.0,
                 max_log_theta: float = 6.0):
        super().__init__()
        self.d = d
        self.max_log_theta = max_log_theta

        # Input projections + LayerNorms
        self.cell_norm  = nn.LayerNorm(d_cell_in)
        self.tax_norm   = nn.LayerNorm(d_tax_in)
        self.samp_norm  = nn.LayerNorm(d_samp_in)
        self.cell_proj  = nn.Linear(d_cell_in, d)
        self.tax_proj   = nn.Linear(d_tax_in, d)
        self.samp_proj  = nn.Linear(d_samp_in, d)

        # Learned [MASK] embedding for masked cells (added to cell_proj output).
        # Cell features already zero out count-derived channels for masked cells;
        # this gives the model an explicit "this is masked" signal in the d-dim
        # representation space.
        self.mask_embed = nn.Parameter(torch.randn(d) * 0.02)

        # Axial trunk
        self.blocks = nn.ModuleList([
            AxialBlock(d, n_heads, m_inducing, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d)

        # Heads
        # Count head: NB(mu = lib * softplus(log_p), theta)
        #   We predict log_p (relative-abundance log) and log_theta from cell embedding
        self.count_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, 3),  # [log_mu_per_sample, log_theta, zi_logit]
        )

        # y head: attention-pool over taxa per sample, then MLP
        self.y_pool = AttnPool(d, n_heads=4)
        self.y_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, 1),  # single scalar (regression or binary logit)
        )

        # Effect head: per-taxon, attention-pool across samples weighted by
        # covariate value. For prototyping, we just predict effect *magnitudes*
        # per taxon as an auxiliary regularizer.
        # In a full system we'd condition on the covariate of interest; for now
        # we predict a (P-sized) effect vector per taxon, given a covariate embedding.
        # We'll keep it simple: aux loss disabled by default in the loss module.
        self.effect_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, 1),  # taxon-level "effect magnitude" summary
        )

    def embed(self, cell_feats: torch.Tensor, tax_feats: torch.Tensor,
              samp_feats: torch.Tensor, visible_cell: torch.Tensor) -> torch.Tensor:
        """
        cell_feats:   (N, T, d_cell_in)
        tax_feats:    (T, d_tax_in)
        samp_feats:   (N, d_samp_in)
        visible_cell: (N, T) bool
        Returns H: (N, T, d)
        """
        N, T, _ = cell_feats.shape
        # Normalize and project each
        c = self.cell_proj(self.cell_norm(cell_feats))           # (N, T, d)
        t = self.tax_proj(self.tax_norm(tax_feats))              # (T, d)
        s = self.samp_proj(self.samp_norm(samp_feats))           # (N, d)

        H = c + t.unsqueeze(0) + s.unsqueeze(1)                  # (N, T, d)

        # Inject mask embedding where cells are masked
        not_visible = (~visible_cell).float().unsqueeze(-1)       # (N, T, 1)
        H = H + not_visible * self.mask_embed.view(1, 1, -1)

        return H

    def forward(self, cell_feats: torch.Tensor, tax_feats: torch.Tensor,
                samp_feats: torch.Tensor, visible_cell: torch.Tensor,
                log_library: torch.Tensor,
                taxon_pad_mask: Optional[torch.Tensor] = None
                ) -> dict:
        """
        log_library: (N,) — log of sample library size, used as offset in NB mean
        """
        H = self.embed(cell_feats, tax_feats, samp_feats, visible_cell)

        for block in self.blocks:
            H = block(H, taxon_pad_mask=taxon_pad_mask)
        H = self.final_ln(H)                                      # (N, T, d)

        # ---- Count head ----
        count_out = self.count_head(H)                            # (N, T, 3)
        # Interpret:
        #   log_p_logit -> softmax over taxa per sample gives relative abundance pi
        #   log_theta   -> dispersion (clamped)
        #   zi_logit    -> structural-zero probability logit
        log_p_logit = count_out[..., 0]                           # (N, T)
        log_theta   = count_out[..., 1].clamp(-self.max_log_theta, self.max_log_theta)
        zi_logit    = count_out[..., 2]                           # (N, T)

        # NB mean: mu = lib * softmax(log_p_logit) per sample
        # Doing this via log_softmax + log_lib in log space is more numerically stable
        log_pi = F.log_softmax(log_p_logit, dim=-1)               # (N, T)
        log_mu = log_pi + log_library.view(-1, 1)                 # (N, T)

        # ---- y head ----
        sample_repr = self.y_pool(H)                              # (N, d)
        y_pred = self.y_head(sample_repr).squeeze(-1)             # (N,)

        # ---- effect head: per-taxon scalar summary ----
        # Pool over samples for each taxon
        Ht = H.transpose(0, 1).contiguous()                        # (T, N, d)
        taxon_repr = Ht.mean(dim=1)                                # (T, d) simple mean for now
        effect_pred = self.effect_head(taxon_repr).squeeze(-1)     # (T,)

        return dict(
            H=H,
            log_mu=log_mu,         # (N, T)
            log_theta=log_theta,   # (N, T)
            zi_logit=zi_logit,     # (N, T)
            y_pred=y_pred,         # (N,)
            effect_pred=effect_pred,  # (T,)
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
