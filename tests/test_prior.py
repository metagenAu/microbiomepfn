"""Validated-prior smoke checks: shapes, finiteness, finite marginals.

All draws use a tiny config so the whole file runs in well under a second.
We deliberately do NOT assert finiteness on X_obs: missingness (NaN) is an
intentional part of the observed-covariate model.
"""
import numpy as np

from microbiomepfn.prior import PriorConfig, sample_dataset, compute_marginals


def tiny_cfg():
    cfg = PriorConfig()
    cfg.n_samples_range = (30, 40)
    cfg.n_taxa_range = (50, 60)
    cfg.n_covariates_range = (4, 8)
    return cfg


def test_shapes_and_dtypes():
    rng = np.random.default_rng(0)
    ds = sample_dataset(cfg=tiny_cfg(), rng=rng)
    n, T, P = ds.hyperparams["n"], ds.hyperparams["T"], ds.hyperparams["P"]

    assert ds.counts.shape == (n, T)
    assert ds.X_obs.shape == (n, P)
    assert ds.X_true.shape == (n, P)
    assert ds.library_sizes.shape == (n,)
    assert ds.true_effects.shape == (P, T)
    assert ds.edge_mask.shape == (P, T)
    assert ds.presence_mask.shape == (n, T)
    assert np.issubdtype(ds.counts.dtype, np.integer)
    assert len(ds.covariate_kinds) == P


def test_counts_nonnegative_and_finite():
    for seed in range(3):
        ds = sample_dataset(cfg=tiny_cfg(), rng=np.random.default_rng(seed))
        assert (ds.counts >= 0).all()
        assert np.isfinite(ds.counts).all()
        assert (ds.library_sizes > 0).all()
        # Each sample's counts cannot exceed its library size.
        assert (ds.counts.sum(axis=1) <= ds.library_sizes).all()


def test_marginals_are_finite():
    for seed in range(3):
        ds = sample_dataset(cfg=tiny_cfg(), rng=np.random.default_rng(seed))
        marg = compute_marginals(ds.counts)
        for name, arr in marg.items():
            arr = np.asarray(arr)
            assert arr.size > 0, f"marginal {name!r} is empty"
            assert np.isfinite(arr).all(), f"marginal {name!r} has non-finite values"
