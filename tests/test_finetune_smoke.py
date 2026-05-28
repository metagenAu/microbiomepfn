"""Fine-tune smoke test: load a tiny checkpoint, fine-tune 3 steps with the
trunk frozen, and verify that ONLY head parameters changed.

The trunk (input projections, mask embedding, axial blocks, final LayerNorm)
must be bit-for-bit unchanged; any changed parameter must belong to one of the
readout heads.
"""
import numpy as np
import torch

from microbiomepfn.prior import PriorConfig, sample_dataset
from microbiomepfn.model import MicrobiomePFN
from microbiomepfn.finetune import build_real_dataset, finetune

K_PHYLO = 16
D_SAMP = 64
HEAD_PREFIXES = ("count_head", "y_pool", "y_head", "effect_head")


def tiny_cfg():
    cfg = PriorConfig()
    cfg.n_samples_range = (30, 40)
    cfg.n_taxa_range = (50, 60)
    cfg.n_covariates_range = (4, 8)
    return cfg


def _make_real_trial(seed):
    ds = sample_dataset(cfg=tiny_cfg(), rng=np.random.default_rng(seed))
    return build_real_dataset(
        counts=ds.counts, X=ds.X_obs,
        cov_kinds=ds.covariate_kinds, cat_levels=ds.cat_levels,
        tree=ds.tree, y=None, y_kind=None,
    )


def test_finetune_frozen_trunk_only_heads_change(tmp_path):
    torch.manual_seed(0)
    model = MicrobiomePFN(
        d_cell_in=6, d_tax_in=K_PHYLO + 9, d_samp_in=D_SAMP,
        d=32, n_layers=2, n_heads=4, m_inducing=8,
    )
    config = dict(
        d=32, n_layers=2, n_heads=4, m_inducing=8, k_phylo=K_PHYLO,
        d_cell_in=6, d_tax_in=K_PHYLO + 9, d_samp_in=D_SAMP,
    )
    ckpt_path = tmp_path / "pretrained.pt"
    torch.save(dict(model_state=model.state_dict(), step=0, config=config), ckpt_path)

    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    trials = [_make_real_trial(10), _make_real_trial(11)]
    ft_model, _history, _info = finetune(
        pretrain_checkpoint=str(ckpt_path),
        trials=trials,
        n_steps=3,
        lr=1e-3,
        warmup_steps=1,
        freeze_trunk_weights=True,
        device_str="cpu",
        seed=0,
    )

    final_state = ft_model.state_dict()
    changed = [
        k for k in init_state
        if not torch.allclose(init_state[k], final_state[k])
    ]

    assert changed, "no parameters changed during fine-tuning"
    for name in changed:
        assert name.split(".")[0] in HEAD_PREFIXES, (
            f"trunk parameter {name!r} changed under a frozen trunk"
        )
