"""
Feature computation: turn a sampled draw into the tensors the model consumes.

Three classes of features, all designed to be free of taxon-identity parameters:

  - Cell features (N, T, d_cell):   log_count, is_zero, log_rel, CLR, log_lib
  - Taxon features (T, d_tax):      phylo_PCs, per-taxon marginals (mean/var/prev/etc.)
  - Sample features (N, d_samp):    covariates (+ missing indicators), lib/richness/shannon,
                                    log(N), log(T)

Key correctness rules:

  - Taxon marginals are computed from *visible* cells only (cells not masked for MLM).
    This avoids label leakage at training time and matches the deployment regime
    where some samples are queries.
  - Phylo PCs are recomputed per draw via BM sampling + SVD. They live in a per-draw
    basis (no cross-draw alignment), which is fine because the model treats them as
    descriptive coordinates, not as identity.
  - Random sign flips of phylo PCs at training time act as augmentation against
    arbitrary basis orientation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from microbiomepfn.prior import Tree, MicrobiomeDataset, bm_on_tree


# ---------------------------------------------------------------------------
# Phylogenetic coordinates
# ---------------------------------------------------------------------------
def phylo_pcs(tree: Tree, k: int, rng: np.random.Generator,
              n_bm_samples: int = 256, sign_flip: bool = True) -> np.ndarray:
    """
    Approximate top-k phylogenetic PCs via randomized SVD on sampled BM realizations.

    Each BM realization is a draw from N(0, C) where C is the tree covariance
    (C[s,t] = depth of MRCA(s,t)). Stacking R draws into Y ∈ R^(T x R) gives
    Y Y^T / R ≈ C, so the top-k left singular vectors of Y approximate the
    top-k eigenvectors of C.

    Returns: (T, k) array. PCs are scaled by their singular values so larger
    eigenvalues map to larger feature magnitudes.
    """
    T = len(tree.tip_ids)
    R = max(n_bm_samples, 2 * k)
    # Sample R BM processes; bm_on_tree returns (R, T)
    Y = bm_on_tree(tree, R, sigma=1.0, rng=rng).T  # (T, R)
    # Randomized SVD via numpy (T is at most a few thousand, R is small)
    U, S, _ = np.linalg.svd(Y, full_matrices=False)  # U: (T, R), S: (R,)
    k_eff = min(k, U.shape[1])
    pcs = U[:, :k_eff] * S[:k_eff][None, :] / np.sqrt(R)
    # Pad if k_eff < k (only matters for tiny trees)
    if k_eff < k:
        pcs = np.concatenate([pcs, np.zeros((T, k - k_eff))], axis=1)
    if sign_flip:
        # Random sign flips: makes the model invariant to arbitrary basis orientation
        signs = rng.choice([-1.0, 1.0], size=k)
        pcs = pcs * signs[None, :]
    # Normalize each PC column to roughly unit variance — keeps scales
    # comparable across draws of very different tree size
    pcs = pcs / (pcs.std(axis=0, keepdims=True) + 1e-8)
    return pcs.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-taxon marginals (dataset-internal, computed from visible cells)
# ---------------------------------------------------------------------------
def taxon_marginals(counts: np.ndarray, visible: np.ndarray,
                    library_sizes: np.ndarray) -> np.ndarray:
    """
    Compute per-taxon statistics from cells that are visible (not masked).

    counts: (N, T) int
    visible: (N, T) bool — True where the cell is observable (not masked)
    library_sizes: (N,) — total counts per sample (always known)

    Returns (T, n_stats). Stats per taxon t:
        mean_rel:     mean relative abundance over visible samples
        var_rel:      variance
        prevalence:   fraction of visible samples where count > 0
        log_mean:     log of mean_rel (clipped)
        log_var:      log of var_rel (clipped)
        log_disp:     log( (var - mean^2/lib_mean) / mean^2 ) approx, dispersion proxy
        mean_when_present:   mean rel when present
        frac_high:    fraction of visible samples with count > median library * 0.001
        n_visible:    log of number of visible samples for this taxon
    """
    N, T = counts.shape
    eps = 1e-12
    safe_lib = library_sizes.clip(min=1).astype(np.float64)
    rel = counts.astype(np.float64) / safe_lib[:, None]   # (N, T)

    vis = visible.astype(np.float64)                       # (N, T)
    n_vis = vis.sum(axis=0).clip(min=1.0)                  # (T,)

    mean_rel = (rel * vis).sum(axis=0) / n_vis             # (T,)
    var_rel = ((rel - mean_rel[None, :]) ** 2 * vis).sum(axis=0) / n_vis
    present = (counts > 0).astype(np.float64) * vis
    prevalence = present.sum(axis=0) / n_vis
    mean_when_present_num = (rel * present).sum(axis=0)
    present_count = present.sum(axis=0).clip(min=1.0)
    mean_when_present = mean_when_present_num / present_count

    log_mean = np.log(mean_rel + eps)
    log_var = np.log(var_rel + eps)
    # Dispersion proxy: NB has var = mu + mu^2/theta, so (var - mu)/mu^2 ~ 1/theta.
    # Use mu_count = mean_rel * mean_lib instead of mu_rel for scale stability.
    mean_lib = safe_lib.mean()
    mu_count = mean_rel * mean_lib
    var_count = var_rel * (mean_lib ** 2)
    disp_proxy = (var_count - mu_count).clip(min=eps) / (mu_count ** 2 + eps)
    log_disp = np.log(disp_proxy + eps)

    thresh = 0.001
    frac_high = ((rel > thresh).astype(np.float64) * vis).sum(axis=0) / n_vis
    log_n_vis = np.log(n_vis)

    feats = np.stack([
        mean_rel,
        var_rel,
        prevalence,
        log_mean,
        log_var,
        log_disp,
        mean_when_present,
        frac_high,
        log_n_vis,
    ], axis=1).astype(np.float32)
    # Replace any non-finite entries (e.g., taxa with zero visible samples)
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return feats


# ---------------------------------------------------------------------------
# Per-sample features
# ---------------------------------------------------------------------------
def sample_features(X_obs: np.ndarray, covariate_kinds, cat_levels,
                    library_sizes: np.ndarray, counts: np.ndarray,
                    visible_sample: np.ndarray) -> np.ndarray:
    """
    Per-sample feature vector. Includes:
      - X_obs (continuous: standardized, NaN→0)
      - missingness indicators for continuous
      - one-hot for categoricals (with extra column for missing)
      - log_lib, log richness, shannon entropy
      - log(N), log(T) at the dataset level (added later by caller)

    visible_sample: (N,) bool — whether each sample is in 'context' (count info trusted)
    """
    N, P = X_obs.shape
    safe_lib = library_sizes.clip(min=1).astype(np.float64)

    cont_cols = [j for j, k in enumerate(covariate_kinds) if k == 'cont']
    cat_cols = [j for j, k in enumerate(covariate_kinds) if k == 'cat']

    feats = []
    # Continuous: standardize, replace NaN with 0, append missing indicator
    if cont_cols:
        Xc = X_obs[:, cont_cols].astype(np.float64)
        miss = np.isnan(Xc)
        # Per-column mean/std on observed
        means = np.nanmean(Xc, axis=0)
        stds = np.nanstd(Xc, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
        stds = np.where((stds < 1e-8) | np.isnan(stds), 1.0, stds)
        Xc_norm = (Xc - means) / stds
        Xc_norm = np.where(miss, 0.0, Xc_norm)
        feats.append(Xc_norm.astype(np.float32))
        feats.append(miss.astype(np.float32))

    # Categorical: one-hot with extra column for missing
    for j_in_obs, k in zip(cat_cols, cat_levels):
        col = X_obs[:, j_in_obs]
        oh = np.zeros((N, k + 1), dtype=np.float32)
        is_miss = np.isnan(col)
        for kk in range(k):
            oh[:, kk] = ((col == kk) & ~is_miss).astype(np.float32)
        oh[:, k] = is_miss.astype(np.float32)
        feats.append(oh)

    # Per-sample summaries from observed counts (always known: counts always observed
    # at sample level — masking happens at the cell level)
    log_lib = np.log(safe_lib).astype(np.float32)[:, None]
    richness = (counts > 0).sum(axis=1).astype(np.float32)
    log_richness = np.log(richness + 1.0)[:, None]
    rel = counts.astype(np.float64) / safe_lib[:, None]
    shannon = -(rel * np.log(rel.clip(min=1e-12))).sum(axis=1).astype(np.float32)
    shannon = shannon[:, None]
    # Visible flag (1 for context samples, 0 for query)
    vis_flag = visible_sample.astype(np.float32)[:, None]

    feats.extend([log_lib, log_richness, shannon, vis_flag])
    return np.concatenate(feats, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-cell features
# ---------------------------------------------------------------------------
def cell_features(counts: np.ndarray, library_sizes: np.ndarray,
                  visible_cell: np.ndarray) -> np.ndarray:
    """
    Per-cell features. For masked cells, the count-dependent features are zeroed
    out; the model is told the cell is masked via the visible indicator.

    Returns (N, T, 6):
        [log(count+1), is_zero, log_rel, CLR, log_lib, is_visible]
    Masked cells get zeros for count-dependent features but keep log_lib and
    the visibility flag.
    """
    N, T = counts.shape
    safe_lib = library_sizes.clip(min=1).astype(np.float64)
    counts_f = counts.astype(np.float64)
    rel = counts_f / safe_lib[:, None]
    log_count = np.log1p(counts_f)
    is_zero = (counts_f == 0).astype(np.float32)
    log_rel = np.log(rel + 1e-6)
    # CLR with masking: compute geo mean over the sample
    clr = log_rel - log_rel.mean(axis=1, keepdims=True)
    log_lib_broadcast = np.log(safe_lib)[:, None].repeat(T, axis=1)

    vis = visible_cell.astype(np.float32)
    # Zero out count-derived features where not visible
    log_count = log_count * vis
    is_zero_kept = is_zero * vis
    log_rel = log_rel * vis
    clr = clr * vis

    feats = np.stack([
        log_count.astype(np.float32),
        is_zero_kept,
        log_rel.astype(np.float32),
        clr.astype(np.float32),
        log_lib_broadcast.astype(np.float32),
        vis,
    ], axis=-1)
    return feats


# ---------------------------------------------------------------------------
# Top-level: assemble a batch dictionary from a draw
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    cell_feats: np.ndarray         # (N, T, d_cell)
    taxon_feats: np.ndarray        # (T, d_tax)
    sample_feats: np.ndarray       # (N, d_samp)
    visible_cell: np.ndarray       # (N, T) bool
    counts: np.ndarray             # (N, T) int  (ground truth, used for loss)
    library_sizes: np.ndarray      # (N,)
    visible_sample: np.ndarray     # (N,) bool — for y prediction context/query split
    y: Optional[np.ndarray]        # (N,) or None
    y_kind: Optional[str]
    true_effects: Optional[np.ndarray]   # (P_active_continuous, T) for aux head
    # Meta
    N: int
    T: int


def build_batch(ds: MicrobiomeDataset, rng: np.random.Generator,
                k_phylo: int = 32,
                cell_mask_frac: float = 0.15,
                sample_query_frac: float = 0.3) -> Batch:
    """
    Convert a sampled draw into the dictionary the model consumes.

      - Mask cell_mask_frac of cells uniformly at random for MLM.
      - Mark sample_query_frac of samples as 'queries' for y-prediction context.
    """
    N, T = ds.counts.shape
    # Cell masking for MLM
    visible_cell = rng.random((N, T)) >= cell_mask_frac     # True = visible
    # Sample query flag (for y prediction; cell features are not affected)
    visible_sample = rng.random(N) >= sample_query_frac

    # Phylo PCs
    pcs = phylo_pcs(ds.tree, k=k_phylo, rng=rng)
    # Per-taxon marginals from visible cells
    marg = taxon_marginals(ds.counts, visible_cell, ds.library_sizes)
    taxon_feats = np.concatenate([pcs, marg], axis=1).astype(np.float32)

    # Sample features
    sf = sample_features(ds.X_obs, ds.covariate_kinds, ds.cat_levels,
                         ds.library_sizes, ds.counts, visible_sample)
    # Append global log(N), log(T) as constant columns
    log_N = np.full((N, 1), np.log(N), dtype=np.float32)
    log_T = np.full((N, 1), np.log(T), dtype=np.float32)
    sample_feats = np.concatenate([sf, log_N, log_T], axis=1)

    # Cell features
    cf = cell_features(ds.counts, ds.library_sizes, visible_cell)

    return Batch(
        cell_feats=cf,
        taxon_feats=taxon_feats,
        sample_feats=sample_feats,
        visible_cell=visible_cell,
        counts=ds.counts.astype(np.int64),
        library_sizes=ds.library_sizes.astype(np.int64),
        visible_sample=visible_sample,
        y=ds.y,
        y_kind=ds.y_kind,
        true_effects=ds.true_effects,
        N=N, T=T,
    )
