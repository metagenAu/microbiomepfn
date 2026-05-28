"""
Deployment / inference on real microbiome data.

The model was trained on synthetic draws where each draw has its own tree and
its own taxa, with no shared namespace. At deployment, the same input contract
applies: pass in counts, covariates, and a tree (or precomputed phylo features),
and the model treats each taxon as a feature vector with no identity.

Two entry points:

  predict_counts(model, counts, covariates, ..., query_mask):
    Returns predicted NB parameters (log_mu, log_theta, zi) for every cell.
    For cells marked as queries (query_mask=True), counts are masked out before
    being passed to the model — the model predicts these from the rest.

  predict_y(model, counts, covariates, ..., y_context, query_samples):
    Few-shot outcome prediction. Pass labeled context samples (y_context),
    get predicted y on query samples.

Required real-data inputs:

  counts:       (N, T) int array, raw read counts (no rarefaction)
  X:            (N, P) array or DataFrame, with covariates
  cov_kinds:    list of 'cont' or 'cat' for each column of X
  cat_levels:   list of n_levels for each categorical column (in column order)
  tree:         a prior.Tree object covering the T taxa
                OR an (T, k_phylo) array of precomputed phylo coordinates

If no tree is available, you can use sequence-derived embeddings instead — just
pass them as the phylo_features argument.
"""
from __future__ import annotations
from typing import Optional, Union, List
import numpy as np
import torch

from microbiomepfn.prior import Tree
from microbiomepfn.features import phylo_pcs, taxon_marginals, sample_features, cell_features
from microbiomepfn.model import MicrobiomePFN


def _assemble_features(counts: np.ndarray, X_obs: np.ndarray,
                       cov_kinds: List[str], cat_levels: List[int],
                       library_sizes: np.ndarray,
                       phylo_features: np.ndarray,
                       visible_cell: np.ndarray,
                       visible_sample: np.ndarray,
                       d_samp_max: int):
    """Build the three feature tensors for inference."""
    N, T = counts.shape
    marg = taxon_marginals(counts, visible_cell, library_sizes)
    taxon_feats = np.concatenate([phylo_features.astype(np.float32),
                                  marg.astype(np.float32)], axis=1)
    sf = sample_features(X_obs, cov_kinds, cat_levels,
                         library_sizes, counts, visible_sample)
    log_N = np.full((N, 1), np.log(N), dtype=np.float32)
    log_T = np.full((N, 1), np.log(T), dtype=np.float32)
    sf = np.concatenate([sf, log_N, log_T], axis=1)
    cur = sf.shape[1]
    if cur < d_samp_max:
        sf = np.concatenate([sf, np.zeros((N, d_samp_max - cur), dtype=np.float32)], axis=1)
    elif cur > d_samp_max:
        sf = sf[:, :d_samp_max]
    cf = cell_features(counts, library_sizes, visible_cell)
    return cf, taxon_feats, sf


def load_model(checkpoint_path: str, device: torch.device = torch.device('cpu')
               ) -> tuple:
    """Load a trained model and its config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model_kwargs = {k: cfg[k] for k in
                    ['d', 'n_layers', 'n_heads', 'm_inducing',
                     'd_cell_in', 'd_tax_in', 'd_samp_in']}
    model = MicrobiomePFN(**model_kwargs).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_counts(model: MicrobiomePFN,
                   counts: np.ndarray,
                   X_obs: np.ndarray,
                   cov_kinds: List[str],
                   cat_levels: List[int],
                   tree: Optional[Tree] = None,
                   phylo_features: Optional[np.ndarray] = None,
                   query_mask: Optional[np.ndarray] = None,
                   k_phylo: int = 16,
                   d_samp_max: int = 200,
                   device: torch.device = torch.device('cpu'),
                   rng: Optional[np.random.Generator] = None,
                   ) -> dict:
    """
    Predict NB parameters for all cells.

      counts:      (N, T) — true counts (will be masked for query cells)
      query_mask:  (N, T) bool — True where the model should predict
                                  (cells are hidden from input)
                                  Default: predict everywhere (no masking)
      phylo_features: (T, k_phylo) precomputed coords; if None, computed from tree
      tree:        prior.Tree object covering the T taxa; required if
                   phylo_features is None
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T = counts.shape

    if phylo_features is None:
        if tree is None:
            raise ValueError('Must pass either tree or phylo_features')
        phylo_features = phylo_pcs(tree, k=k_phylo, rng=rng, sign_flip=False)

    if query_mask is None:
        query_mask = np.zeros((N, T), dtype=bool)
    visible_cell = ~query_mask
    visible_sample = np.ones(N, dtype=bool)
    library_sizes = counts.sum(axis=1).clip(min=1)

    cf, tf, sf = _assemble_features(
        counts, X_obs, cov_kinds, cat_levels, library_sizes,
        phylo_features, visible_cell, visible_sample, d_samp_max)

    cf_t = torch.from_numpy(cf).to(device)
    tf_t = torch.from_numpy(tf).to(device)
    sf_t = torch.from_numpy(sf).to(device)
    vc_t = torch.from_numpy(visible_cell).to(device)
    log_lib = torch.log(torch.from_numpy(library_sizes).float().clamp(min=1)).to(device)

    out = model(cell_feats=cf_t, tax_feats=tf_t, samp_feats=sf_t,
                visible_cell=vc_t, log_library=log_lib)

    return dict(
        log_mu=out['log_mu'].cpu().numpy(),
        log_theta=out['log_theta'].cpu().numpy(),
        zi_logit=out['zi_logit'].cpu().numpy(),
        # convenience
        predicted_mean=out['log_mu'].exp().cpu().numpy(),
        predicted_relative=torch.softmax(out['log_mu'], dim=-1).cpu().numpy(),
        y_pred=out['y_pred'].cpu().numpy(),
        effect_pred=out['effect_pred'].cpu().numpy(),
    )


@torch.no_grad()
def predict_y(model: MicrobiomePFN,
              counts: np.ndarray,
              X_obs: np.ndarray,
              cov_kinds: List[str],
              cat_levels: List[int],
              y_context: np.ndarray,
              context_mask: np.ndarray,
              tree: Optional[Tree] = None,
              phylo_features: Optional[np.ndarray] = None,
              k_phylo: int = 16,
              d_samp_max: int = 200,
              device: torch.device = torch.device('cpu'),
              rng: Optional[np.random.Generator] = None,
              ) -> np.ndarray:
    """
    Few-shot y prediction. Context samples have known y; query samples don't.

      y_context:    (N,) — values are ignored where context_mask is False
      context_mask: (N,) bool — True for samples whose y is known
    Returns predicted y for all N samples; useful entries are query samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T = counts.shape
    if phylo_features is None:
        if tree is None:
            raise ValueError('Must pass either tree or phylo_features')
        phylo_features = phylo_pcs(tree, k=k_phylo, rng=rng, sign_flip=False)

    visible_cell = np.ones((N, T), dtype=bool)   # don't mask any cells
    visible_sample = context_mask.astype(bool)
    library_sizes = counts.sum(axis=1).clip(min=1)

    cf, tf, sf = _assemble_features(
        counts, X_obs, cov_kinds, cat_levels, library_sizes,
        phylo_features, visible_cell, visible_sample, d_samp_max)

    cf_t = torch.from_numpy(cf).to(device)
    tf_t = torch.from_numpy(tf).to(device)
    sf_t = torch.from_numpy(sf).to(device)
    vc_t = torch.from_numpy(visible_cell).to(device)
    log_lib = torch.log(torch.from_numpy(library_sizes).float().clamp(min=1)).to(device)

    out = model(cell_feats=cf_t, tax_feats=tf_t, samp_feats=sf_t,
                visible_cell=vc_t, log_library=log_lib)
    return out['y_pred'].cpu().numpy()


# ---------------------------------------------------------------------------
# Sequence-embedding option for when no tree is available
# ---------------------------------------------------------------------------
def kmer_phylo_features(sequences: List[str], k: int = 6,
                        n_components: int = 16) -> np.ndarray:
    """
    Build a phylo-like embedding from raw ASV sequences using k-mer profiles + PCA.

      sequences: list of T strings (DNA, ACGT)
      Returns (T, n_components) embedding.

    This is a fallback for deployment when no tree is given. Sequence similarity
    tracks 16S phylogeny closely, so the resulting embedding has roughly the
    structure the model expects from BM coordinates.
    """
    from collections import Counter
    T = len(sequences)
    # Build k-mer vocabulary on the fly
    vocab = {}
    profiles = []
    for s in sequences:
        s = s.upper()
        counts = Counter(s[i:i+k] for i in range(len(s) - k + 1))
        for kmer in counts:
            if kmer not in vocab:
                vocab[kmer] = len(vocab)
        profiles.append(counts)
    V = len(vocab)
    M = np.zeros((T, V), dtype=np.float32)
    for i, prof in enumerate(profiles):
        for kmer, c in prof.items():
            M[i, vocab[kmer]] = c
    # Normalize rows
    M = M / (M.sum(axis=1, keepdims=True) + 1e-8)
    # Center and SVD
    M = M - M.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    k_eff = min(n_components, U.shape[1])
    pcs = (U[:, :k_eff] * S[:k_eff][None, :]).astype(np.float32)
    pcs = pcs / (pcs.std(axis=0, keepdims=True) + 1e-8)
    if k_eff < n_components:
        pcs = np.concatenate([pcs, np.zeros((T, n_components - k_eff), dtype=np.float32)], axis=1)
    return pcs
