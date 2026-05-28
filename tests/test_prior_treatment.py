"""Treatment-extension behaviour checks.

Two load-bearing properties of the optional treatment injection:
  1. The fraction of draws that become "treatment studies" tracks
     ``p_treatment_study``.
  2. ``p_treatment_study = 0.0`` recovers the validated prior exactly — i.e.
     no draw is ever turned into a treatment study.

The covariate-count range is chosen so every draw has at least one categorical
column (treatment injection requires one).
"""
import numpy as np

from microbiomepfn.prior_treatment import (
    TreatmentPriorConfig,
    sample_dataset_with_treatment,
)


def tiny_treatment_cfg(p_treatment_study):
    cfg = TreatmentPriorConfig()
    cfg.n_samples_range = (30, 40)
    cfg.n_taxa_range = (50, 60)
    cfg.n_covariates_range = (4, 8)  # guarantees n_cat >= 1
    cfg.p_treatment_study = p_treatment_study
    return cfg


def _treatment_fraction(cfg, n_draws, seed):
    rng = np.random.default_rng(seed)
    n_treat = 0
    for _ in range(n_draws):
        ds = sample_dataset_with_treatment(cfg=cfg, rng=rng)
        if ds.hyperparams["treatment_cov_idx"] >= 0:
            n_treat += 1
    return n_treat / n_draws


def test_treatment_frequency_matches_config():
    p = 0.5
    n_draws = 80
    frac = _treatment_fraction(tiny_treatment_cfg(p), n_draws, seed=123)
    # ~3.2 sigma binomial tolerance (sigma ~= 0.056 at p=0.5, n=80).
    assert abs(frac - p) < 0.18, f"treatment fraction {frac:.3f} far from {p}"


def test_zero_p_treatment_produces_no_treatment_draws():
    cfg = tiny_treatment_cfg(0.0)
    rng = np.random.default_rng(7)
    for _ in range(40):
        ds = sample_dataset_with_treatment(cfg=cfg, rng=rng)
        assert ds.hyperparams["treatment_cov_idx"] == -1
        assert ds.hyperparams["treatment_n_levels"] == 0
