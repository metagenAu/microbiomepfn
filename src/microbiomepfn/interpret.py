"""
Interpretability for a (pretrained + fine-tuned) MicrobiomePFN.

Main entry points:

  counterfactual_treatment_effect(model, ds, treatment_col, ...)
      For each taxon, estimate the model's predicted log-fold change in
      relative abundance when the treatment value is flipped (or set to
      each non-reference level), averaging across samples.
      Returns a per-taxon ranking of treatment-responsive taxa.

  bootstrap_treatment_ranking(model, ds, treatment_col, n_boot=100, ...)
      Bootstrap-resample samples and re-compute the counterfactual ranking
      per resample. Returns per-taxon point estimate plus 90% confidence
      interval for treatment effect.

  attention_introspection(model, ds, ...)
      Report which taxa the model attends to when producing the per-sample
      y prediction (via the y_pool attention weights). High-attention taxa
      are 'used by the model' for outcome prediction.

  rank_table(effects, names=None, top_k=30, ci=None)
      Convenience for printing a ranked treatment-response table.

Design notes:

  - Counterfactual perturbation is the *correct* tool here: the model is
    a black box, but a conditional one. To answer 'how does treatment
    affect taxon t?', we ask the model directly: 'predict t given
    treatment=A vs treatment=B, holding all other covariates fixed'.
    The difference is the model's estimated treatment effect.
  - We use the count head's log_mu output. After fine-tuning on MLM,
    this is the head with the strongest training signal; the effect_head
    is auxiliary and not used here.
  - For binary treatments, we compute (treated - control). For multi-level
    treatments, we compute each non-reference level minus level 0.
  - For continuous "treatments" (e.g. dosage), pass small_perturbation
    and we'll do (X + δ) − (X − δ) / 2δ as a numerical derivative.
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Union
import numpy as np
import torch

from microbiomepfn.prior import MicrobiomeDataset
from microbiomepfn.features import phylo_pcs, build_batch
from microbiomepfn.model import MicrobiomePFN
from microbiomepfn.train import _pad_sample_feats, batch_to_torch
from microbiomepfn.deploy import _assemble_features


# ---------------------------------------------------------------------------
# Build features for counterfactual: like build_batch but lets us override
# the X matrix without re-randomizing other features
# ---------------------------------------------------------------------------
def _features_with_override(ds: MicrobiomeDataset, X_override: np.ndarray,
                            phylo_features: np.ndarray, d_samp_max: int,
                            ) -> dict:
    """Build feature tensors using X_override for covariates but keeping
    all other dataset properties (counts, library sizes, tree) fixed.

    No cell masking — we want the model's prediction with full information.
    No sample-level query split — we use all samples.
    """
    N, T = ds.counts.shape
    visible_cell = np.ones((N, T), dtype=bool)
    visible_sample = np.ones(N, dtype=bool)

    cf, tf, sf = _assemble_features(
        counts=ds.counts, X_obs=X_override,
        cov_kinds=ds.covariate_kinds, cat_levels=ds.cat_levels,
        library_sizes=ds.library_sizes,
        phylo_features=phylo_features,
        visible_cell=visible_cell, visible_sample=visible_sample,
        d_samp_max=d_samp_max)
    return dict(
        cell_feats=cf, tax_feats=tf, samp_feats=sf,
        visible_cell=visible_cell, library_sizes=ds.library_sizes,
    )


@torch.no_grad()
def _predict_log_mu(model: MicrobiomePFN, ds: MicrobiomeDataset,
                    X_override: np.ndarray, phylo_features: np.ndarray,
                    d_samp_max: int, device: torch.device) -> np.ndarray:
    """Run the model and return predicted log_mu (N, T)."""
    f = _features_with_override(ds, X_override, phylo_features, d_samp_max)
    cf = torch.from_numpy(f['cell_feats']).to(device)
    tf = torch.from_numpy(f['tax_feats']).to(device)
    sf = torch.from_numpy(f['samp_feats']).to(device)
    vc = torch.from_numpy(f['visible_cell']).to(device)
    log_lib = torch.log(torch.from_numpy(f['library_sizes']).float().clamp(min=1)).to(device)
    model.eval()
    out = model(cell_feats=cf, tax_feats=tf, samp_feats=sf,
                visible_cell=vc, log_library=log_lib)
    return out['log_mu'].cpu().numpy()


# ---------------------------------------------------------------------------
# Counterfactual treatment effect
# ---------------------------------------------------------------------------
def counterfactual_treatment_effect(
    model: MicrobiomePFN,
    ds: MicrobiomeDataset,
    treatment_col: int,
    cfg: dict,
    reference_level: Union[int, float] = 0,
    target_levels: Optional[List] = None,
    is_categorical: Optional[bool] = None,
    continuous_perturbation: float = 1.0,
    device: torch.device = torch.device('cpu'),
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Per-taxon treatment effect via counterfactual perturbation.

    treatment_col: index of the treatment covariate in ds.X_obs (column 0 = first).
    reference_level: the "control" value (default 0 for binary/cat;
                     for continuous, pass the baseline value, e.g. mean).
    target_levels: list of target values to compare against reference.
                   For categorical: list of level indices (default: all non-reference).
                   For continuous: list of perturbed values (default: [reference + continuous_perturbation]).
    is_categorical: bool override; if None, inferred from ds.covariate_kinds.

    Returns dict with:
      - per_level_log_fold_change: (n_targets, T) — model-predicted
            log-fold-change in relative abundance for each target level vs reference
      - per_level_per_sample: (n_targets, N, T) — same but per-sample
      - target_levels: list of target values
      - ranked_taxa: (T,) array of taxon indices sorted by max |effect| across levels
      - effect_magnitudes: (T,) max |effect| per taxon across levels
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if is_categorical is None:
        is_categorical = (ds.covariate_kinds[treatment_col] == 'cat')

    N, T = ds.counts.shape
    phylo_features = phylo_pcs(ds.tree, k=cfg['k_phylo'], rng=rng,
                               sign_flip=False)
    d_samp_max = cfg['d_samp_in']

    X_base = ds.X_obs.copy()

    # Decide target levels
    if target_levels is None:
        if is_categorical:
            cat_offset = sum(1 for k in ds.covariate_kinds[:treatment_col]
                             if k == 'cat')
            n_levels = ds.cat_levels[cat_offset]
            target_levels = [lvl for lvl in range(n_levels)
                             if lvl != reference_level]
        else:
            target_levels = [float(reference_level) + continuous_perturbation]

    # Predict under reference
    X_ref = X_base.copy()
    X_ref[:, treatment_col] = reference_level
    log_mu_ref = _predict_log_mu(model, ds, X_ref, phylo_features,
                                  d_samp_max, device)            # (N, T)
    # Convert to log relative-abundance by subtracting log_library
    log_lib = np.log(ds.library_sizes.clip(min=1).astype(np.float64))
    log_rel_ref = log_mu_ref - log_lib[:, None]                  # (N, T)

    per_level_per_sample = []
    per_level_log_fold_change = []
    for tgt in target_levels:
        X_tgt = X_base.copy()
        X_tgt[:, treatment_col] = tgt
        log_mu_tgt = _predict_log_mu(model, ds, X_tgt, phylo_features,
                                      d_samp_max, device)
        log_rel_tgt = log_mu_tgt - log_lib[:, None]
        log_fc_per_sample = log_rel_tgt - log_rel_ref            # (N, T)
        per_level_per_sample.append(log_fc_per_sample)
        per_level_log_fold_change.append(log_fc_per_sample.mean(axis=0))

    per_level_log_fold_change = np.stack(per_level_log_fold_change)   # (L, T)
    per_level_per_sample = np.stack(per_level_per_sample)             # (L, N, T)

    # Rank taxa by max |log-fold-change| across levels
    eff_mag = np.abs(per_level_log_fold_change).max(axis=0)           # (T,)
    ranked = np.argsort(-eff_mag)

    return dict(
        per_level_log_fold_change=per_level_log_fold_change,
        per_level_per_sample=per_level_per_sample,
        target_levels=list(target_levels),
        ranked_taxa=ranked,
        effect_magnitudes=eff_mag,
    )


# ---------------------------------------------------------------------------
# Bootstrap CIs for treatment effects
# ---------------------------------------------------------------------------
def bootstrap_treatment_ranking(
    model: MicrobiomePFN,
    ds: MicrobiomeDataset,
    treatment_col: int,
    cfg: dict,
    n_boot: int = 100,
    reference_level: Union[int, float] = 0,
    target_level: Optional[Union[int, float]] = None,
    ci_alpha: float = 0.1,
    is_categorical: Optional[bool] = None,
    continuous_perturbation: float = 1.0,
    device: torch.device = torch.device('cpu'),
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Bootstrap the treatment effect for each taxon.

    For each bootstrap iteration: resample samples with replacement, run
    counterfactual_treatment_effect, store the per-taxon effect for the
    chosen target level. After all iterations, compute per-taxon median and
    confidence interval.

    Returns:
      point_estimate: (T,) — full-sample estimate
      median: (T,) — median across bootstrap reps
      lo, hi: (T,) — lower and upper (ci_alpha/2, 1 - ci_alpha/2) percentiles
      ranked_by_robust:  taxa sorted by min(|lo|, |hi|) — i.e. most robustly nonzero
      ranked_by_point:   taxa sorted by |point_estimate|
    """
    rng = np.random.default_rng(seed)
    if is_categorical is None:
        is_categorical = (ds.covariate_kinds[treatment_col] == 'cat')

    # Determine the target level if not specified
    if target_level is None:
        if is_categorical:
            cat_offset = sum(1 for k in ds.covariate_kinds[:treatment_col]
                             if k == 'cat')
            n_levels = ds.cat_levels[cat_offset]
            # Pick first non-reference level
            target_level = next((lvl for lvl in range(n_levels)
                                 if lvl != reference_level), 1)
        else:
            target_level = float(reference_level) + continuous_perturbation

    # Full-sample point estimate
    cte_full = counterfactual_treatment_effect(
        model, ds, treatment_col, cfg,
        reference_level=reference_level,
        target_levels=[target_level],
        is_categorical=is_categorical,
        continuous_perturbation=continuous_perturbation,
        device=device, rng=rng)
    T = ds.counts.shape[1]
    point = cte_full['per_level_log_fold_change'][0]    # (T,)

    # Bootstrap
    N = ds.counts.shape[0]
    boot_effects = np.zeros((n_boot, T), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        ds_b = _bootstrap_subset(ds, idx)
        cte_b = counterfactual_treatment_effect(
            model, ds_b, treatment_col, cfg,
            reference_level=reference_level,
            target_levels=[target_level],
            is_categorical=is_categorical,
            continuous_perturbation=continuous_perturbation,
            device=device, rng=rng)
        boot_effects[b] = cte_b['per_level_log_fold_change'][0]
        if verbose and (b + 1) % 20 == 0:
            print(f'  bootstrap {b+1}/{n_boot}')

    median = np.median(boot_effects, axis=0)
    lo = np.quantile(boot_effects, ci_alpha / 2, axis=0)
    hi = np.quantile(boot_effects, 1 - ci_alpha / 2, axis=0)

    # Robust ranking: by minimum absolute value of CI (smallest plausible effect)
    # — taxa whose CI is far from zero rank highly. Use sign of the median.
    ci_min_abs = np.where(
        (lo > 0) | (hi < 0),
        np.minimum(np.abs(lo), np.abs(hi)),
        0.0,
    )
    ranked_robust = np.argsort(-ci_min_abs)
    ranked_point = np.argsort(-np.abs(point))

    return dict(
        point_estimate=point,
        median=median,
        lo=lo, hi=hi,
        boot_effects=boot_effects,
        ranked_by_robust=ranked_robust,
        ranked_by_point=ranked_point,
        target_level=target_level,
        reference_level=reference_level,
        ci_alpha=ci_alpha,
    )


def _bootstrap_subset(ds: MicrobiomeDataset, idx: np.ndarray) -> MicrobiomeDataset:
    from copy import copy
    new = copy(ds)
    new.counts = ds.counts[idx]
    new.X_obs = ds.X_obs[idx]
    new.X_true = ds.X_true[idx]
    new.library_sizes = ds.library_sizes[idx]
    new.presence_mask = ds.presence_mask[idx]
    if ds.y is not None:
        new.y = ds.y[idx]
    new.hyperparams = dict(ds.hyperparams)
    new.hyperparams['n'] = len(idx)
    return new


# ---------------------------------------------------------------------------
# Attention introspection (which taxa does the y-head use?)
# ---------------------------------------------------------------------------
@torch.no_grad()
def attention_introspection(
    model: MicrobiomePFN,
    ds: MicrobiomeDataset,
    cfg: dict,
    device: torch.device = torch.device('cpu'),
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Inspect the y_pool attention weights — which taxa does the model attend
    to when producing the per-sample outcome prediction?

    Returns:
      attn_weights: (N, T) — attention weights from the y_pool, per sample
                              over taxa
      mean_attn_per_taxon: (T,) — average attention each taxon gets
      ranked_taxa: (T,) — taxon indices sorted by mean attention
    """
    if rng is None:
        rng = np.random.default_rng(0)
    phylo_features = phylo_pcs(ds.tree, k=cfg['k_phylo'], rng=rng,
                               sign_flip=False)
    d_samp_max = cfg['d_samp_in']
    f = _features_with_override(ds, ds.X_obs, phylo_features, d_samp_max)

    cf = torch.from_numpy(f['cell_feats']).to(device)
    tf = torch.from_numpy(f['tax_feats']).to(device)
    sf = torch.from_numpy(f['samp_feats']).to(device)
    vc = torch.from_numpy(f['visible_cell']).to(device)
    log_lib = torch.log(torch.from_numpy(f['library_sizes']).float().clamp(min=1)).to(device)

    model.eval()
    # Run the trunk to get H
    H = model.embed(cf, tf, sf, vc)
    for block in model.blocks:
        H = block(H, taxon_pad_mask=None)
    H = model.final_ln(H)                                 # (N, T, d)

    # Manually replicate y_pool but ask for attention weights
    pool = model.y_pool
    N, T, d = H.shape
    q = pool.q.expand(N, -1, -1)                          # (N, 1, d)
    H_ln = pool.ln(H)
    _, attn_weights = pool.attn(q, H_ln, H_ln, need_weights=True,
                                 average_attn_weights=True)
    attn = attn_weights.squeeze(1).cpu().numpy()          # (N, T)
    mean_attn = attn.mean(axis=0)
    ranked = np.argsort(-mean_attn)
    return dict(
        attn_weights=attn,
        mean_attn_per_taxon=mean_attn,
        ranked_taxa=ranked,
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def rank_table(effects: np.ndarray, names: Optional[List[str]] = None,
               top_k: int = 30, ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               title: str = 'Top taxa by |effect|') -> str:
    """Format a per-taxon effect ranking. effects: (T,). Optional ci=(lo, hi)."""
    T = len(effects)
    order = np.argsort(-np.abs(effects))
    lines = [title, '-' * len(title)]
    if ci is None:
        lines.append(f"{'rank':>4} {'taxon':>30} {'effect':>10}")
    else:
        lines.append(f"{'rank':>4} {'taxon':>30} {'effect':>10}  "
                     f"{'CI low':>9} {'CI high':>9}")
    for r in range(min(top_k, T)):
        t = int(order[r])
        name = names[t] if names is not None else f'taxon_{t}'
        if ci is None:
            lines.append(f"{r+1:>4} {name:>30} {effects[t]:>10.4f}")
        else:
            lo, hi = ci
            sig = '*' if (lo[t] > 0 or hi[t] < 0) else ' '
            lines.append(f"{r+1:>4} {name:>30} {effects[t]:>10.4f}"
                         f"  {lo[t]:>9.4f} {hi[t]:>9.4f} {sig}")
    return '\n'.join(lines)


if __name__ == '__main__':
    # Demo: load a fine-tuned model and run counterfactual on a simulated trial
    import argparse
    from microbiomepfn.prior_treatment import sample_dataset_with_treatment, TreatmentPriorConfig
    from microbiomepfn.deploy import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str,
                        help='path to fine-tuned model checkpoint')
    parser.add_argument('--n_boot', type=int, default=30)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(args.checkpoint, device=device)

    # Mock trial
    rng = np.random.default_rng(7)
    pcfg = TreatmentPriorConfig()
    pcfg.n_samples_range = (60, 70)
    pcfg.n_taxa_range = (150, 200)
    pcfg.p_treatment_study = 1.0
    ds = sample_dataset_with_treatment(cfg=pcfg, rng=rng)
    treat_idx = ds.hyperparams['treatment_cov_idx']
    print(f'Trial: N={ds.counts.shape[0]}, T={ds.counts.shape[1]}, '
          f'treatment at col {treat_idx} '
          f'(n_levels={ds.hyperparams["treatment_n_levels"]}, '
          f'asym={ds.hyperparams["treatment_asymmetry"]:.2f})')

    # Point-estimate counterfactual
    print('\nCounterfactual (point estimate):')
    cte = counterfactual_treatment_effect(
        model, ds, treatment_col=treat_idx, cfg=cfg,
        reference_level=0, device=device, rng=rng)
    print(rank_table(cte['per_level_log_fold_change'][0],
                     top_k=args.top_k, title='Top responsive taxa'))

    # Compare to ground truth (we have true_effects from the prior)
    true_effects = ds.true_effects[treat_idx]
    overlap = len(set(cte['ranked_taxa'][:args.top_k].tolist())
                  & set(np.argsort(-np.abs(true_effects))[:args.top_k].tolist()))
    print(f'\nOverlap with ground-truth top-{args.top_k}: {overlap}/{args.top_k}')

    # Bootstrap
    print(f'\nBootstrapping with {args.n_boot} resamples...')
    boot = bootstrap_treatment_ranking(
        model, ds, treatment_col=treat_idx, cfg=cfg,
        n_boot=args.n_boot, device=device, verbose=True)
    print(rank_table(boot['median'], top_k=args.top_k,
                     ci=(boot['lo'], boot['hi']),
                     title='Top responsive taxa (bootstrap median + 90% CI)'))
    n_sig = ((boot['lo'] > 0) | (boot['hi'] < 0)).sum()
    print(f'\nTaxa with CI excluding zero: {n_sig}')
