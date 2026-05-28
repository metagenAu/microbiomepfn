"""5-step training smoke test on a tiny prior.

Exercises the real training path (build_batch -> batch_to_torch -> train_step):
losses stay finite and at least one parameter moves between step 0 and step 5.
"""
import numpy as np
import torch

from microbiomepfn.prior import PriorConfig, sample_dataset
from microbiomepfn.features import build_batch
from microbiomepfn.model import MicrobiomePFN
from microbiomepfn.train import batch_to_torch, train_step, _pad_sample_feats

K_PHYLO = 16
D_SAMP = 64


def tiny_cfg():
    cfg = PriorConfig()
    cfg.n_samples_range = (30, 40)
    cfg.n_taxa_range = (50, 60)
    cfg.n_covariates_range = (4, 8)
    return cfg


def test_five_step_training_run():
    cfg = tiny_cfg()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Probe one draw to fix input feature dims.
    probe = build_batch(sample_dataset(cfg=cfg, rng=np.random.default_rng(1)),
                        np.random.default_rng(1), k_phylo=K_PHYLO)
    d_cell_in = probe.cell_feats.shape[-1]
    d_tax_in = probe.taxon_feats.shape[-1]

    model = MicrobiomePFN(
        d_cell_in=d_cell_in, d_tax_in=d_tax_in, d_samp_in=D_SAMP,
        d=32, n_layers=2, n_heads=4, m_inducing=8,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Snapshot a parameter that is guaranteed to receive gradient (count head).
    before = model.count_head[0].weight.detach().clone()

    losses = []
    for _ in range(5):
        ds = sample_dataset(cfg=cfg, rng=rng)
        batch = build_batch(ds, rng, k_phylo=K_PHYLO,
                            cell_mask_frac=0.15, sample_query_frac=0.3)
        batch = _pad_sample_feats(batch, D_SAMP)
        bt = batch_to_torch(batch, torch.device("cpu"))
        stats = train_step(model, bt, opt)
        losses.append(stats["loss"])

    assert len(losses) == 5
    for loss in losses:
        assert np.isfinite(loss), f"non-finite loss: {loss}"

    after = model.count_head[0].weight.detach()
    assert not torch.allclose(before, after), "parameters did not change after training"
