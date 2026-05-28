"""
Fine-tune a pretrained MicrobiomePFN on real trial data.

Two modes:

  Mode A — single-trial fine-tune (data = one MicrobiomeDataset).
  Mode B — multi-trial fine-tune (data = list of MicrobiomeDataset).
           Trials are cycled through; each step is one trial.

Plus a trunk-freezing toggle:

  freeze_trunk=True   — only the heads (count_head, y_pool, y_head,
                        effect_head) get gradients. The axial trunk and
                        all input projections are frozen at their
                        pretrained values. Recommended for small trials.
  freeze_trunk=False  — everything trains. Use only with larger trial
                        sets and careful held-out monitoring.

Real-trial data adapter:

  build_real_dataset(counts, X, cov_kinds, cat_levels, tree, y=None,
                     y_kind=None) — package real data into the same
                     MicrobiomeDataset shape the rest of the pipeline
                     expects. Sets true_effects/edge_mask/presence_mask
                     to sensible defaults (zeros / ones) since we don't
                     have ground truth on real data.

Training details:

  - Hold out a fraction of samples as a validation set; never touch them
    during training. Track validation count-NLL each eval epoch.
  - Early-stopping on validation loss with patience.
  - Low learning rate (default 3e-5) and short schedule (default 200 steps).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import torch

from microbiomepfn.prior import MicrobiomeDataset, Tree
from microbiomepfn.features import build_batch
from microbiomepfn.model import MicrobiomePFN
from microbiomepfn.losses import compute_loss
from microbiomepfn.train import batch_to_torch, _pad_sample_feats, WarmupCosine
from microbiomepfn.deploy import load_model


# ---------------------------------------------------------------------------
# Real-trial → MicrobiomeDataset adapter
# ---------------------------------------------------------------------------
def build_real_dataset(counts: np.ndarray,
                       X: np.ndarray,
                       cov_kinds: List[str],
                       cat_levels: List[int],
                       tree: Tree,
                       y: Optional[np.ndarray] = None,
                       y_kind: Optional[str] = None,
                       ) -> MicrobiomeDataset:
    """
    Wrap real trial data in the MicrobiomeDataset shape.

    counts: (N, T) integer count matrix
    X: (N, P) covariate matrix; continuous columns first, then categorical
       Categorical columns hold integer level indices 0..k-1.
    cov_kinds: list of 'cont' or 'cat' per column of X
    cat_levels: number of levels per categorical column, in column order
    tree: a prior.Tree object covering the T taxa
    y: optional outcome (N,)
    y_kind: 'cont' or 'binary' if y is given

    Sets unknown ground-truth fields (true_effects, edge_mask, presence_mask)
    to sensible defaults — they're not used during fine-tuning except as
    optional weak supervision signals (which we disable here).
    """
    N, T = counts.shape
    P = X.shape[1]
    n_cont = sum(1 for k in cov_kinds if k == 'cont')

    library_sizes = counts.sum(axis=1).clip(min=1).astype(np.int64)

    return MicrobiomeDataset(
        counts=counts.astype(np.int64),
        X_obs=X.astype(float),
        X_true=X.astype(float),
        covariate_kinds=list(cov_kinds),
        cat_levels=list(cat_levels),
        library_sizes=library_sizes,
        true_effects=np.zeros((P, T), dtype=np.float32),
        edge_mask=np.zeros((P, T), dtype=bool),
        tree=tree,
        habitat_centers=None,
        habitat_dims=None,
        presence_mask=np.ones((N, T), dtype=bool),
        y=y, y_kind=y_kind,
        count_model='unknown',
        hyperparams=dict(n=N, T=T, P=P, n_cont=n_cont, n_cat=P - n_cont,
                         real=True),
    )


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------
def freeze_trunk(model: MicrobiomePFN):
    """Freeze everything except the readout heads."""
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze just the heads (and the y_pool which is part of the y readout)
    for module in [model.count_head, model.y_pool, model.y_head, model.effect_head]:
        for p in module.parameters():
            p.requires_grad = True


def unfreeze_all(model: MicrobiomePFN):
    for p in model.parameters():
        p.requires_grad = True


def n_trainable(model: MicrobiomePFN) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Held-out validation split (sample-level)
# ---------------------------------------------------------------------------
def split_validation_samples(ds: MicrobiomeDataset, val_frac: float,
                              rng: np.random.Generator
                              ) -> tuple:
    """Return (train_ds, val_ds) with samples split."""
    N = ds.counts.shape[0]
    n_val = max(1, int(val_frac * N))
    val_idx = rng.choice(N, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(N), val_idx)
    return _subset_samples(ds, train_idx), _subset_samples(ds, val_idx)


def _subset_samples(ds: MicrobiomeDataset, idx: np.ndarray) -> MicrobiomeDataset:
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
# Validation pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_val(model: MicrobiomePFN, val_ds: MicrobiomeDataset,
                 rng: np.random.Generator, d_samp_max: int,
                 k_phylo: int, device: torch.device,
                 cell_mask_frac: float = 0.15,
                 use_zinb: bool = True,
                 ) -> dict:
    """Evaluate count-NLL on held-out validation samples with cell masking."""
    model.eval()
    batch = build_batch(val_ds, rng, k_phylo=k_phylo,
                        cell_mask_frac=cell_mask_frac,
                        sample_query_frac=0.0)
    batch = _pad_sample_feats(batch, d_samp_max)
    bt = batch_to_torch(batch, device)
    log_lib = torch.log(bt['library_sizes'].float().clamp(min=1))
    out = model(cell_feats=bt['cell_feats'], tax_feats=bt['tax_feats'],
                samp_feats=bt['samp_feats'], visible_cell=bt['visible_cell'],
                log_library=log_lib)
    losses = compute_loss(out, bt['counts'], bt['visible_cell'],
                          y=bt['y'], y_kind=bt['y_kind'],
                          visible_sample=bt['visible_sample'],
                          true_effects=None, use_zinb=use_zinb,
                          y_weight=0.0, effect_weight=0.0)
    return dict(
        val_count_nll=float(losses['loss_count'].item()),
        val_y_loss=float(losses['loss_y'].item()),
    )


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------
def finetune(
    pretrain_checkpoint: str,
    trials: Union[MicrobiomeDataset, List[MicrobiomeDataset]],
    n_steps: int = 200,
    lr: float = 3e-5,
    warmup_steps: int = 20,
    val_frac: float = 0.2,
    freeze_trunk_weights: bool = True,
    cell_mask_frac: float = 0.15,
    sample_query_frac: float = 0.3,
    y_weight: float = 0.3,
    effect_weight: float = 0.0,            # disable aux during fine-tune
    use_zinb: bool = True,
    grad_clip: float = 1.0,
    log_every: int = 20,
    eval_every: int = 20,
    patience: int = 5,                      # early-stop patience (eval epochs)
    seed: int = 0,
    device_str: str = 'auto',
    save_path: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Fine-tune a pretrained model on real trial data.

    Returns the fine-tuned model and a history dict.
    """
    device = torch.device('cuda' if device_str == 'auto' and torch.cuda.is_available()
                          else 'cpu' if device_str == 'auto' else device_str)
    if rng is None:
        rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if isinstance(trials, MicrobiomeDataset):
        trials = [trials]
    n_trials = len(trials)
    print(f'Fine-tuning on {n_trials} trial(s)')

    # Load pretrained model
    model, cfg = load_model(pretrain_checkpoint, device=device)
    d_samp_max = cfg['d_samp_in']
    k_phylo = cfg['k_phylo']
    print(f'Loaded checkpoint (d={cfg["d"]}, n_layers={cfg["n_layers"]})')

    # Sample-level train/val split per trial
    train_trials, val_trials = [], []
    for ds in trials:
        tr, va = split_validation_samples(ds, val_frac, rng)
        train_trials.append(tr)
        val_trials.append(va)
    print(f'Per trial sample sizes: train={[t.counts.shape[0] for t in train_trials]}, '
          f'val={[t.counts.shape[0] for t in val_trials]}')

    # Freeze / unfreeze
    total_params = sum(p.numel() for p in model.parameters())
    if freeze_trunk_weights:
        freeze_trunk(model)
        print(f'Frozen trunk; trainable params: {n_trainable(model)/1e3:.1f}K '
              f'(of {total_params/1e3:.1f}K total)')
    else:
        unfreeze_all(model)
        print(f'All params trainable: {n_trainable(model)/1e3:.1f}K')

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = WarmupCosine(opt, lr, warmup_steps, n_steps)

    history = []
    best_val = float('inf')
    best_state = None
    stale_evals = 0
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # Pick which trial to use this step
        ds_train = train_trials[(step - 1) % n_trials]
        if ds_train.counts.shape[0] < 5:
            continue

        batch = build_batch(ds_train, rng,
                            k_phylo=k_phylo,
                            cell_mask_frac=cell_mask_frac,
                            sample_query_frac=sample_query_frac)
        batch = _pad_sample_feats(batch, d_samp_max)
        bt = batch_to_torch(batch, device)

        model.train()
        log_lib = torch.log(bt['library_sizes'].float().clamp(min=1))
        out = model(cell_feats=bt['cell_feats'], tax_feats=bt['tax_feats'],
                    samp_feats=bt['samp_feats'], visible_cell=bt['visible_cell'],
                    log_library=log_lib)
        losses = compute_loss(out, bt['counts'], bt['visible_cell'],
                              y=bt['y'], y_kind=bt['y_kind'],
                              visible_sample=bt['visible_sample'],
                              true_effects=None, use_zinb=use_zinb,
                              y_weight=y_weight, effect_weight=effect_weight)
        opt.zero_grad(set_to_none=True)
        losses['loss'].backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip)
        opt.step()
        sched.step()

        record = dict(step=step,
                      loss=float(losses['loss'].item()),
                      loss_count=float(losses['loss_count'].item()),
                      loss_y=float(losses['loss_y'].item()),
                      grad_norm=float(gn.item()))
        history.append(record)

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f'step {step:4d} | loss {record["loss"]:6.3f} '
                  f'(count {record["loss_count"]:5.3f} y {record["loss_y"]:5.3f}) '
                  f'| grad {record["grad_norm"]:4.2f} | {step/elapsed:.2f} steps/s')

        # Eval on validation samples (averaged across trials)
        if step % eval_every == 0:
            val_records = []
            for vds in val_trials:
                if vds.counts.shape[0] < 2:
                    continue
                vr = evaluate_val(model, vds, rng, d_samp_max=d_samp_max,
                                  k_phylo=k_phylo, device=device,
                                  cell_mask_frac=cell_mask_frac, use_zinb=use_zinb)
                val_records.append(vr)
            if val_records:
                avg_val = np.mean([v['val_count_nll'] for v in val_records])
                print(f'  >> val_count_nll = {avg_val:.4f}', end='')
                if avg_val < best_val - 1e-4:
                    best_val = avg_val
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    stale_evals = 0
                    print(' [best]')
                else:
                    stale_evals += 1
                    print(f' (stale {stale_evals}/{patience})')
                if stale_evals >= patience:
                    print(f'Early stopping at step {step}.')
                    break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Restored best model (val_count_nll = {best_val:.4f}).')

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            model_state=model.state_dict(),
            config=cfg,
            best_val_nll=best_val,
            pretrain_checkpoint=pretrain_checkpoint,
        ), save_path)
        print(f'Saved fine-tuned model to {save_path}')

    return model, history, dict(train_trials=train_trials,
                                val_trials=val_trials,
                                best_val_nll=best_val)


if __name__ == '__main__':
    # Demo: fine-tune on a single simulated "trial" drawn from the prior
    # (just to verify the fine-tune loop runs; real usage swaps the data source)
    import argparse
    from microbiomepfn.prior_treatment import sample_dataset_with_treatment, TreatmentPriorConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('pretrain_checkpoint', type=str)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--no_freeze', action='store_true')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    print('Generating mock trial dataset(s)...')
    rng = np.random.default_rng(123)
    cfg = TreatmentPriorConfig()
    cfg.n_samples_range = (60, 80)
    cfg.n_taxa_range = (150, 250)
    cfg.p_treatment_study = 1.0   # force a treatment for the demo
    trials = [sample_dataset_with_treatment(cfg=cfg, rng=rng) for _ in range(3)]

    model, hist, info = finetune(
        pretrain_checkpoint=args.pretrain_checkpoint,
        trials=trials,
        n_steps=args.n_steps,
        lr=args.lr,
        freeze_trunk_weights=not args.no_freeze,
        save_path=args.save_path,
    )
