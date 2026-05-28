"""
Training entry point.

Each step:
  1. Sample one draw from the prior.
  2. Build a Batch (features + visibility masks).
  3. Move tensors to device.
  4. Forward → loss → backward → step.
  5. Log every N steps.

Single-draw-per-step is the natural batching mode here:
each draw has its own (N, T, P) shape, and a draw already contains
hundreds of samples and hundreds of taxa — that *is* the batch.
Multi-draw batching would require padding to max(T) and max(N) across
draws, which is mostly wasted compute. If GPU utilization is low, raise N and T.
"""
from __future__ import annotations
import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from microbiomepfn.prior import sample_dataset, PriorConfig
from microbiomepfn.features import build_batch, Batch
from microbiomepfn.model import MicrobiomePFN
from microbiomepfn.losses import compute_loss


# ===========================================================================
# Batch -> torch on device
# ===========================================================================
def batch_to_torch(batch: Batch, device: torch.device) -> dict:
    return dict(
        cell_feats=torch.from_numpy(batch.cell_feats).to(device),
        tax_feats=torch.from_numpy(batch.taxon_feats).to(device),
        samp_feats=torch.from_numpy(batch.sample_feats).to(device),
        visible_cell=torch.from_numpy(batch.visible_cell).to(device),
        counts=torch.from_numpy(batch.counts).to(device),
        library_sizes=torch.from_numpy(batch.library_sizes).to(device),
        visible_sample=torch.from_numpy(batch.visible_sample).to(device),
        y=(torch.from_numpy(np.asarray(batch.y)).to(device) if batch.y is not None else None),
        y_kind=batch.y_kind,
        true_effects=(torch.from_numpy(batch.true_effects.astype(np.float32)).to(device)
                      if batch.true_effects is not None else None),
        N=batch.N, T=batch.T,
    )


# ===========================================================================
# Training step
# ===========================================================================
def train_step(model: MicrobiomePFN, batch_t: dict,
               optimizer: torch.optim.Optimizer,
               grad_clip: float = 1.0,
               use_zinb: bool = True,
               y_weight: float = 0.3,
               effect_weight: float = 0.0) -> dict:
    model.train()
    log_lib = torch.log(batch_t['library_sizes'].float().clamp(min=1))
    out = model(
        cell_feats=batch_t['cell_feats'],
        tax_feats=batch_t['tax_feats'],
        samp_feats=batch_t['samp_feats'],
        visible_cell=batch_t['visible_cell'],
        log_library=log_lib,
    )
    losses = compute_loss(
        out=out,
        counts=batch_t['counts'],
        visible_cell=batch_t['visible_cell'],
        y=batch_t['y'],
        y_kind=batch_t['y_kind'],
        visible_sample=batch_t['visible_sample'],
        true_effects=batch_t['true_effects'],
        use_zinb=use_zinb,
        y_weight=y_weight,
        effect_weight=effect_weight,
    )
    optimizer.zero_grad(set_to_none=True)
    losses['loss'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return dict(
        loss=float(losses['loss'].item()),
        loss_count=float(losses['loss_count'].item()),
        loss_y=float(losses['loss_y'].item()),
        loss_effect=float(losses['loss_effect'].item()),
        grad_norm=float(grad_norm.item()),
    )


# ===========================================================================
# LR schedule (linear warmup -> cosine decay)
# ===========================================================================
class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.opt = optimizer
        self.base = base_lr
        self.warm = warmup_steps
        self.total = total_steps
        self.min = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warm:
            lr = self.base * (self.step_num / max(1, self.warm))
        else:
            t = (self.step_num - self.warm) / max(1, self.total - self.warm)
            t = min(1.0, t)
            import math
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr


# ===========================================================================
# Main training driver
# ===========================================================================
def train(
    n_steps: int = 1000,
    d: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    m_inducing: int = 64,
    k_phylo: int = 16,
    base_lr: float = 3e-4,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    cell_mask_frac: float = 0.15,
    sample_query_frac: float = 0.3,
    log_every: int = 20,
    save_every: int = 500,
    use_zinb: bool = True,
    y_weight: float = 0.3,
    effect_weight: float = 0.05,
    n_taxa_cap: int = 600,           # subsample taxa if larger, for training speed
    n_samples_cap: int = 200,        # subsample samples if larger
    seed: int = 0,
    device_str: str = 'auto',
    save_dir: str = 'checkpoints',
    use_treatment: bool = False,             # if True, use prior_treatment instead
    p_treatment_study: float = 0.4,          # only used when use_treatment=True
):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f'Device: {device}')

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Decide which sampler to use
    if use_treatment:
        from microbiomepfn.prior_treatment import (
            sample_dataset_with_treatment as sample_fn,
            TreatmentPriorConfig,
        )
        cfg = TreatmentPriorConfig()
        cfg.p_treatment_study = p_treatment_study
        print(f'Using validated prior + treatment extension '
              f'(p_treatment_study={p_treatment_study})')
    else:
        sample_fn = sample_dataset
        cfg = PriorConfig()
        print('Using validated prior (no treatment extension)')

    # Probe one draw to find input feature dimensions
    probe_ds = sample_fn(cfg=cfg, rng=np.random.default_rng(seed + 1))
    probe_batch = build_batch(probe_ds, np.random.default_rng(seed + 1), k_phylo=k_phylo)
    d_cell_in = probe_batch.cell_feats.shape[-1]
    d_tax_in = probe_batch.taxon_feats.shape[-1]
    d_samp_in = probe_batch.sample_feats.shape[-1]
    print(f'Input dims: cell={d_cell_in}, tax={d_tax_in}, samp={d_samp_in}')
    # NOTE: d_samp_in varies with the number of covariates! We have to fix it.
    # Solution: pad/truncate sample features to a fixed max.

    D_SAMP_MAX = 200
    d_samp_in_eff = D_SAMP_MAX

    model = MicrobiomePFN(
        d_cell_in=d_cell_in,
        d_tax_in=d_tax_in,
        d_samp_in=d_samp_in_eff,
        d=d,
        n_layers=n_layers,
        n_heads=n_heads,
        m_inducing=m_inducing,
    ).to(device)
    print(f'Model params: {model.count_params()/1e6:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    sched = WarmupCosine(opt, base_lr, warmup_steps, n_steps)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()
    ema_loss = None
    for step in range(1, n_steps + 1):
        # ----- Sample a draw -----
        ds = sample_fn(cfg=cfg, rng=rng)

        # ----- Subsample taxa & samples to fit compute budget -----
        N, T = ds.counts.shape
        if T > n_taxa_cap:
            tax_idx = rng.choice(T, size=n_taxa_cap, replace=False)
            ds = _subsample_taxa(ds, tax_idx)
            N, T = ds.counts.shape
        if N > n_samples_cap:
            samp_idx = rng.choice(N, size=n_samples_cap, replace=False)
            ds = _subsample_samples(ds, samp_idx)
            N, T = ds.counts.shape

        # Skip degenerate draws
        if T < 10 or N < 5:
            continue

        # ----- Build features -----
        batch = build_batch(ds, rng,
                            k_phylo=k_phylo,
                            cell_mask_frac=cell_mask_frac,
                            sample_query_frac=sample_query_frac)
        # Pad sample features to fixed dim
        batch = _pad_sample_feats(batch, D_SAMP_MAX)

        # ----- Move to device, take step -----
        bt = batch_to_torch(batch, device)
        try:
            stats = train_step(model, bt, opt, grad_clip=grad_clip,
                               use_zinb=use_zinb, y_weight=y_weight,
                               effect_weight=effect_weight)
        except torch.cuda.OutOfMemoryError:
            print(f'  step {step}: OOM at N={N}, T={T}; skipping')
            torch.cuda.empty_cache()
            continue

        sched.step()
        ema_loss = stats['loss'] if ema_loss is None else 0.98 * ema_loss + 0.02 * stats['loss']
        history.append(dict(step=step, **stats, N=N, T=T,
                            y_kind=batch.y_kind))

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f'step {step:5d} | loss {stats["loss"]:7.3f} '
                  f'(ema {ema_loss:7.3f}) | '
                  f'count {stats["loss_count"]:6.3f} '
                  f'y {stats["loss_y"]:6.3f} '
                  f'eff {stats["loss_effect"]:6.3f} | '
                  f'grad {stats["grad_norm"]:5.2f} | '
                  f'N={N:3d} T={T:4d} y={str(batch.y_kind):>6} | '
                  f'{step/elapsed:.2f} steps/s')

        if step % save_every == 0 or step == n_steps:
            ckpt_path = Path(save_dir) / f'model_step{step}.pt'
            torch.save(dict(
                model_state=model.state_dict(),
                step=step,
                config=dict(
                    d=d, n_layers=n_layers, n_heads=n_heads,
                    m_inducing=m_inducing, k_phylo=k_phylo,
                    d_cell_in=d_cell_in, d_tax_in=d_tax_in,
                    d_samp_in=d_samp_in_eff,
                ),
            ), ckpt_path)
            print(f'  saved {ckpt_path}')

    return model, history


# ===========================================================================
# Helpers for subsampling and padding
# ===========================================================================
def _subsample_taxa(ds, tax_idx):
    from copy import copy
    new = copy(ds)
    new.counts = ds.counts[:, tax_idx]
    new.true_effects = ds.true_effects[:, tax_idx]
    new.edge_mask = ds.edge_mask[:, tax_idx]
    new.presence_mask = ds.presence_mask[:, tax_idx]
    # Tree: subset tips. We rebuild a minimal Tree object with re-indexed tips.
    new.tree = _subset_tree_tips(ds.tree, tax_idx)
    if ds.habitat_centers is not None:
        new.habitat_centers = ds.habitat_centers[tax_idx]
    new.hyperparams = dict(ds.hyperparams)
    new.hyperparams['T'] = len(tax_idx)
    return new


def _subset_tree_tips(tree, tax_idx):
    """Build a subtree containing only the tips in tax_idx.
    Internal nodes with single descendants are NOT collapsed (kept simple).
    Branches are kept with original lengths; relabel tips."""
    # We keep all original nodes and just record which tip indices we keep.
    # Branch lengths are aggregated by collapsing chains of degree-2 internal nodes.
    from microbiomepfn.prior import Tree
    keep_tip_set = set(int(tree.tip_ids[i]) for i in tax_idx)

    # 1) Mark every node that has at least one descendant tip in keep_tip_set
    n_total = len(tree.parent)
    keep = np.zeros(n_total, dtype=bool)
    # Postorder via reverse preorder
    for v in tree.preorder[::-1]:
        if int(v) in keep_tip_set:
            keep[v] = True
        p = int(tree.parent[v])
        if p >= 0 and keep[v]:
            keep[p] = True

    # 2) For each kept node, find its kept parent and accumulate branch length.
    # Walk up from each node to find nearest kept ancestor.
    new_parent_of = np.full(n_total, -1, dtype=np.int64)
    new_branch_of = np.zeros(n_total, dtype=np.float64)
    for v in range(n_total):
        if not keep[v]:
            continue
        # Sum branch lengths walking up until kept ancestor (or root)
        bl = 0.0
        u = v
        while True:
            p = int(tree.parent[u])
            if p < 0:
                new_parent_of[v] = -1
                new_branch_of[v] = 0.0
                break
            bl += tree.branch_len[u]
            if keep[p]:
                # parent is p, but p's index will be remapped
                new_parent_of[v] = p
                new_branch_of[v] = bl
                break
            u = p

    # 3) Remap kept nodes to dense indices
    old_to_new = -np.ones(n_total, dtype=np.int64)
    new_nodes = [v for v in tree.preorder if keep[v]]
    for new_i, old_v in enumerate(new_nodes):
        old_to_new[old_v] = new_i

    n_new = len(new_nodes)
    parent_new = np.full(n_new, -1, dtype=np.int64)
    branch_new = np.zeros(n_new, dtype=np.float64)
    for new_i, old_v in enumerate(new_nodes):
        old_par = new_parent_of[old_v]
        if old_par >= 0:
            parent_new[new_i] = old_to_new[old_par]
        branch_new[new_i] = new_branch_of[old_v]

    # Tip ids in new indexing, in the order of tax_idx (so tips are in the
    # caller's expected order)
    tip_ids_new = np.array(
        [old_to_new[int(tree.tip_ids[i])] for i in tax_idx], dtype=np.int64)

    # Build a preorder traversal
    kids = [[] for _ in range(n_new)]
    for i, p in enumerate(parent_new):
        if p >= 0:
            kids[p].append(i)
    pre = []
    # Find root(s): the node with parent == -1
    roots = [i for i, p in enumerate(parent_new) if p < 0]
    stack = list(roots)
    while stack:
        v = stack.pop()
        pre.append(v)
        stack.extend(kids[v])
    preorder_new = np.asarray(pre, dtype=np.int64)

    # Recompute depths
    # Note: validated Tree dataclass doesn't include a 'depth' field;
    # downstream features (phylo_pcs) don't need it (BM sampling is the
    # primitive used, not direct depth values).
    return Tree(parent=parent_new, branch_len=branch_new,
                tip_ids=tip_ids_new, preorder=preorder_new,
                root=int(roots[0]) if roots else 0)


def _subsample_samples(ds, samp_idx):
    from copy import copy
    new = copy(ds)
    new.counts = ds.counts[samp_idx]
    new.X_obs = ds.X_obs[samp_idx]
    new.X_true = ds.X_true[samp_idx]
    new.library_sizes = ds.library_sizes[samp_idx]
    new.presence_mask = ds.presence_mask[samp_idx]
    if ds.y is not None:
        new.y = ds.y[samp_idx]
    new.hyperparams = dict(ds.hyperparams)
    new.hyperparams['n'] = len(samp_idx)
    return new


def _pad_sample_feats(batch: Batch, target_dim: int) -> Batch:
    cur = batch.sample_feats.shape[1]
    if cur == target_dim:
        return batch
    if cur > target_dim:
        batch.sample_feats = batch.sample_feats[:, :target_dim]
    else:
        pad = np.zeros((batch.sample_feats.shape[0], target_dim - cur),
                       dtype=np.float32)
        batch.sample_feats = np.concatenate([batch.sample_feats, pad], axis=1)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--d', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_taxa_cap', type=int, default=600)
    parser.add_argument('--n_samples_cap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--use_treatment', action='store_true',
                        help='Use prior_treatment (validated prior + treatment extension)')
    parser.add_argument('--p_treatment_study', type=float, default=0.4,
                        help='Only used if --use_treatment is set')
    args = parser.parse_args()

    train(n_steps=args.n_steps, d=args.d, n_layers=args.n_layers,
          n_heads=args.n_heads, base_lr=args.lr,
          n_taxa_cap=args.n_taxa_cap, n_samples_cap=args.n_samples_cap,
          seed=args.seed, device_str=args.device, save_dir=args.save_dir,
          use_treatment=args.use_treatment,
          p_treatment_study=args.p_treatment_study)


if __name__ == '__main__':
    main()
