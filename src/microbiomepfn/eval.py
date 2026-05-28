"""
Evaluation and inference utilities.

Two main capabilities:

  1. evaluate_held_out_cells:
     Given a model and a draw, mask a fraction of cells, run inference, and
     compare predicted vs true counts using:
       - Spearman correlation of predicted vs true on masked cells (log-scale)
       - Calibration: fraction of true counts in the model's central 80% interval
       - NLL per masked cell
     Useful as a held-out diagnostic during pretraining.

  2. predict_on_new_dataset:
     Given a trained model and a real (or simulated) dataset where the user
     wants predictions on specific samples or cells, set those as 'queries' and
     return predicted counts / y / per-taxon summaries.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from microbiomepfn.prior import MicrobiomeDataset, sample_dataset, PriorConfig
from microbiomepfn.features import build_batch
from microbiomepfn.model import MicrobiomePFN
from microbiomepfn.losses import nb_log_prob, zinb_log_prob


def _to_torch_batch(batch, device, d_samp_max):
    sf = batch.sample_feats
    cur = sf.shape[1]
    if cur < d_samp_max:
        pad = np.zeros((sf.shape[0], d_samp_max - cur), dtype=np.float32)
        sf = np.concatenate([sf, pad], axis=1)
    elif cur > d_samp_max:
        sf = sf[:, :d_samp_max]
    return dict(
        cell_feats=torch.from_numpy(batch.cell_feats).to(device),
        tax_feats=torch.from_numpy(batch.taxon_feats).to(device),
        samp_feats=torch.from_numpy(sf).to(device),
        visible_cell=torch.from_numpy(batch.visible_cell).to(device),
        counts=torch.from_numpy(batch.counts).to(device),
        library_sizes=torch.from_numpy(batch.library_sizes).to(device),
    )


@torch.no_grad()
def evaluate_held_out_cells(model: MicrobiomePFN, ds: MicrobiomeDataset,
                            rng: np.random.Generator,
                            d_samp_max: int = 200,
                            k_phylo: int = 16,
                            cell_mask_frac: float = 0.15,
                            device: torch.device = torch.device('cpu'),
                            use_zinb: bool = True,
                            ) -> dict:
    """Run model on a draw with cell_mask_frac of cells masked, return metrics."""
    model.eval()
    batch = build_batch(ds, rng, k_phylo=k_phylo,
                        cell_mask_frac=cell_mask_frac,
                        sample_query_frac=0.0)  # no y queries for cell eval
    bt = _to_torch_batch(batch, device, d_samp_max)
    log_lib = torch.log(bt['library_sizes'].float().clamp(min=1))

    out = model(
        cell_feats=bt['cell_feats'], tax_feats=bt['tax_feats'],
        samp_feats=bt['samp_feats'], visible_cell=bt['visible_cell'],
        log_library=log_lib,
    )

    masked = ~bt['visible_cell']
    counts = bt['counts']
    log_mu = out['log_mu']
    log_theta = out['log_theta']
    zi_logit = out['zi_logit']

    if use_zinb:
        lp = zinb_log_prob(counts, log_mu, log_theta, zi_logit)
    else:
        lp = nb_log_prob(counts, log_mu, log_theta)

    nll = -(lp * masked.float()).sum() / masked.sum().clamp(min=1)

    # Predicted mean count and true count on masked cells
    pred = log_mu.exp()
    pred_m = pred[masked].cpu().numpy()
    true_m = counts[masked].cpu().numpy()

    # Spearman on log-scale
    if len(pred_m) > 1:
        rho, _ = spearmanr(np.log1p(pred_m), np.log1p(true_m))
    else:
        rho = float('nan')

    # Baseline: predict marginal mean per taxon from visible cells
    visible = ~masked
    counts_vis = counts.cpu().numpy()
    visible_np = visible.cpu().numpy()
    libsz = bt['library_sizes'].cpu().numpy()
    rel = counts_vis / libsz[:, None].clip(min=1)
    # mean relative abundance from visible cells per taxon
    n_vis = visible_np.sum(axis=0).clip(min=1)
    mean_rel_t = (rel * visible_np).sum(axis=0) / n_vis  # (T,)
    masked_np = masked.cpu().numpy()
    # Baseline predicted mean count = mean_rel_t * lib
    baseline_pred = mean_rel_t[None, :] * libsz[:, None]
    base_m = baseline_pred[masked_np]
    if len(base_m) > 1:
        base_rho, _ = spearmanr(np.log1p(base_m), np.log1p(true_m))
    else:
        base_rho = float('nan')

    # Calibration: fraction of true counts in predicted 80% interval
    # For NB, approximate the central 80% interval by Monte Carlo:
    with torch.no_grad():
        mu = log_mu.exp()
        theta = log_theta.exp()
        # NB sampling via gamma-Poisson
        gamma = torch.distributions.Gamma(theta, theta / mu.clamp(min=1e-8))
        samples = torch.distributions.Poisson(gamma.sample((50,))).sample()  # (50, N, T)
        lo = torch.quantile(samples, 0.1, dim=0)
        hi = torch.quantile(samples, 0.9, dim=0)
        in_interval = ((counts >= lo) & (counts <= hi)).float()
        coverage = (in_interval * masked.float()).sum() / masked.sum().clamp(min=1)

    return dict(
        nll_per_cell=float(nll.item()),
        spearman_model=float(rho),
        spearman_baseline=float(base_rho),
        coverage_80=float(coverage.item()),
        n_masked=int(masked.sum().item()),
        T=ds.counts.shape[1],
        N=ds.counts.shape[0],
    )


def quick_eval(model_path: str, n_eval: int = 10, device: str = 'cpu',
               seed: int = 999) -> list:
    """Load a checkpoint and evaluate on fresh draws."""
    device = torch.device(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = MicrobiomePFN(**{k: cfg[k] for k in
                             ['d', 'n_layers', 'n_heads', 'm_inducing',
                              'd_cell_in', 'd_tax_in', 'd_samp_in']}).to(device)
    model.load_state_dict(ckpt['model_state'])
    rng = np.random.default_rng(seed)
    prior_cfg = PriorConfig()
    results = []
    for i in range(n_eval):
        ds = sample_dataset(cfg=prior_cfg, rng=rng)
        # Cap size for eval speed
        N, T = ds.counts.shape
        if T > 300:
            tax_idx = rng.choice(T, size=300, replace=False)
            from microbiomepfn.train import _subsample_taxa
            ds = _subsample_taxa(ds, tax_idx)
        if N > 100:
            samp_idx = rng.choice(N, size=100, replace=False)
            from microbiomepfn.train import _subsample_samples
            ds = _subsample_samples(ds, samp_idx)
        res = evaluate_held_out_cells(model, ds, rng,
                                       d_samp_max=cfg['d_samp_in'],
                                       k_phylo=cfg['k_phylo'],
                                       device=device)
        res['draw'] = i
        results.append(res)
    return results


def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: microbiomepfn-eval <model_path.pt>')
        sys.exit(1)
    res = quick_eval(sys.argv[1])
    print(f"{'draw':>4} {'N':>4} {'T':>4} {'nll':>7} "
          f"{'rho_model':>10} {'rho_base':>9} {'cov80':>6}")
    for r in res:
        print(f"{r['draw']:>4} {r['N']:>4} {r['T']:>4} "
              f"{r['nll_per_cell']:>7.3f} {r['spearman_model']:>10.3f} "
              f"{r['spearman_baseline']:>9.3f} {r['coverage_80']:>6.3f}")
    nll = np.mean([r['nll_per_cell'] for r in res])
    rm = np.mean([r['spearman_model'] for r in res])
    rb = np.mean([r['spearman_baseline'] for r in res])
    cov = np.mean([r['coverage_80'] for r in res])
    print(f'\nmean: nll={nll:.3f}  rho_model={rm:.3f}  rho_baseline={rb:.3f}  cov80={cov:.3f}')


if __name__ == '__main__':
    main()
