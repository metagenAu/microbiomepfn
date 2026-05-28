"""
Treatment-injection extension on top of the validated prior.

The base prior (`prior.py`) has been calibrated against real ASV data
via KS-distance testing on marginal statistics (per-sample log library
size, richness, Shannon; per-taxon log mean abundance, log variance,
prevalence). Its defaults are *not* something to second-guess casually
— they were tuned to match real microbiome marginals.

This module adds ONE optional extension: a treatment-effect injection
mechanism. In a configurable fraction of draws (default 40%), one
categorical column is overridden to become a "treatment" — a 2-4 level
variable with asymmetric, optionally phylogenetically-focused effects
on taxon abundance.

Why add this on top of an already-validated prior:
  The base prior's categorical effects are drawn symmetrically (each
  level gets a normal score, effects per taxon are i.i.d.). Real
  treatment data (antibiotics, dietary shifts, soil amendments) often
  shows *asymmetric* patterns: antibiotics deplete many taxa, probiotics
  enrich a few. The treatment extension models this asymmetry directly.

Disabling the extension:
  Set `p_treatment_study = 0.0` in TreatmentPriorConfig to recover
  the exact behavior of the validated base prior. Then this module is
  transparent.

A note of caution:
  This extension was not KS-validated. If your real deployment data
  doesn't show asymmetric treatment patterns, training with a high
  `p_treatment_study` may bias the model toward looking for asymmetry
  that isn't there. The safe default for general training is
  `p_treatment_study = 0.25 - 0.40`; for treatment-heavy data
  (e.g. specifically training for RCT analysis), `0.5 - 0.7` is
  defensible.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from microbiomepfn.prior import (
    PriorConfig, MicrobiomeDataset, Tree,
    sample_yule_tree, bm_on_tree,
    _safe_softmax, _sample_continuous_X,
)


@dataclass
class TreatmentPriorConfig(PriorConfig):
    """Inherits all validated defaults from PriorConfig, adds treatment knobs."""
    # Probability that a draw is a "treatment study"
    p_treatment_study: float = 0.4

    # Number of levels for the treatment categorical
    # (2 = binary control/treated; 3-4 = multi-arm)
    treatment_n_levels_range: tuple = (2, 5)        # high is exclusive

    # Effect strength scaling
    treatment_strength_range: tuple = (1.5, 6.0)

    # Asymmetry: -1.0 = mostly enrichment (probiotic-like);
    #            +1.0 = mostly depletion (antibiotic-like);
    #             0.0 = balanced contrast
    treatment_asymmetry_range: tuple = (-0.8, 0.8)

    # P(responsive taxa form a phylogenetic clade vs being scattered)
    treatment_clade_focus_prob: float = 0.5

    # Fraction of taxa that respond to treatment
    treatment_p_tax_range: tuple = (0.15, 0.6)


def sample_dataset_with_treatment(cfg: Optional[TreatmentPriorConfig] = None,
                                  rng: Optional[np.random.Generator] = None
                                  ) -> MicrobiomeDataset:
    """Validated prior + optional treatment injection.

    This re-implements the prior.sample_dataset() pipeline inline because the
    treatment override has to happen in the middle (after cat sampling, before
    eta computation). We can't just post-process.

    If p_treatment_study=0, output is bit-identical to prior.sample_dataset().
    """
    if cfg is None:
        cfg = TreatmentPriorConfig()
    if rng is None:
        rng = np.random.default_rng()

    n = int(rng.integers(*cfg.n_samples_range))
    T = int(rng.integers(*cfg.n_taxa_range))
    P = int(rng.integers(*cfg.n_covariates_range))
    n_cont = int(round(P * cfg.p_continuous))
    n_cat = P - n_cont

    log_lib_mean = rng.uniform(*cfg.log_lib_mean_range)
    log_lib_sd = rng.uniform(*cfg.log_lib_sd_range)
    p_cov = rng.beta(*cfg.p_cov_active_beta)
    p_tax = rng.beta(*cfg.p_taxon_responsive_beta)
    log_base_bm_sd = rng.uniform(*cfg.log_base_bm_sd_range)
    edge_bm_sd = rng.uniform(*cfg.edge_bm_sd_range)
    effect_bm_sd = rng.uniform(*cfg.effect_bm_sd_range)
    yule_rate = rng.uniform(*cfg.yule_rate_range)
    cov_scale_df = rng.uniform(*cfg.cov_scale_df_range)
    cov_scale_loc = rng.uniform(*cfg.cov_scale_loc_range)
    od_scale = rng.uniform(*cfg.od_scale_range)
    use_dm = rng.random() < cfg.p_dirichlet_multinomial
    log_alpha0 = rng.uniform(*cfg.dm_log_alpha0_range)
    zi_rate = rng.beta(*cfg.zi_beta) * cfg.zi_max
    use_habitat = rng.random() < cfg.p_habitat_active
    n_hab_dims = int(rng.integers(*cfg.n_habitat_dims_range))
    hab_width_log = rng.uniform(*cfg.habitat_width_log_range)
    hab_temp = rng.uniform(*cfg.habitat_temp_range)
    noise_scale = rng.exponential(1.0 / cfg.noise_scale_rate)
    missing_rate = rng.beta(*cfg.missing_rate_beta)
    use_nonlin = rng.random() < cfg.p_nonlinear

    has_y = rng.random() < cfg.p_outcome
    y_binary = has_y and (rng.random() < cfg.p_outcome_binary)
    y_p_taxa = rng.beta(*cfg.outcome_p_taxa_active_beta)
    y_p_cov = rng.beta(*cfg.outcome_p_cov_active_beta)
    y_noise = rng.exponential(1.0 / cfg.outcome_noise_rate)

    # Tree + baseline
    tree = sample_yule_tree(T, rng, lam=yule_rate)
    log_base = bm_on_tree(tree, 1, log_base_bm_sd, rng).ravel()

    # Latent covariates
    X_cont_true = _sample_continuous_X(n, n_cont, rng)
    cat_levels = [int(rng.integers(2, 6)) for _ in range(n_cat)]
    X_cat_true = (np.column_stack([rng.integers(0, k, size=n) for k in cat_levels])
                  if n_cat else np.zeros((n, 0), dtype=int))
    cat_design = []
    for j, k in enumerate(cat_levels):
        scores = rng.standard_normal(k)
        cat_design.append(scores[X_cat_true[:, j]])
    X_cat_design = (np.column_stack(cat_design) if cat_design
                    else np.zeros((n, 0)))
    X_design = np.concatenate([X_cont_true, X_cat_design], axis=1)

    # BM-driven activity and effects
    active_cov = rng.random(P) < p_cov
    n_active = int(active_cov.sum())
    edge_mask = np.zeros((P, T), dtype=bool)
    effects = np.zeros((P, T))
    if n_active > 0:
        z = bm_on_tree(tree, n_active, edge_bm_sd, rng)
        thr = np.quantile(z, 1.0 - p_tax, axis=1, keepdims=True)
        act = z > thr
        e_bm = bm_on_tree(tree, n_active, effect_bm_sd, rng)
        per_cov_scale = (cov_scale_loc * np.abs(
            rng.standard_t(df=cov_scale_df, size=(n_active, 1))))
        e_active = e_bm * per_cov_scale * act
        idx = np.where(active_cov)[0]
        edge_mask[idx] = act
        effects[idx] = e_active

    # --- TREATMENT INJECTION ---
    # Override one categorical to be a treatment with asymmetric effects.
    # No structural label tells the model which column is the treatment.
    treatment_cov_idx = -1
    treatment_asymmetry = 0.0
    treatment_strength = 1.0
    treatment_n_levels = 0
    treatment_clade_focused = False
    if n_cat > 0 and rng.random() < cfg.p_treatment_study:
        j_cat = int(rng.integers(n_cat))
        treat_idx = n_cont + j_cat
        new_n_levels = int(rng.integers(*cfg.treatment_n_levels_range))

        # Override the categorical's level count and resample its values
        cat_levels[j_cat] = new_n_levels
        new_levels = rng.integers(0, new_n_levels, size=n)
        X_cat_true[:, j_cat] = new_levels

        # Asymmetric design scores: reference level at 0, others biased by asym
        asym = float(rng.uniform(*cfg.treatment_asymmetry_range))
        scores = np.zeros(new_n_levels)
        if new_n_levels > 1:
            raw = rng.standard_normal(new_n_levels - 1)
            scores[1:] = (raw - asym) * (1.0 + 0.3 * abs(asym))
        cat_design[j_cat] = scores[new_levels]
        X_design[:, treat_idx] = cat_design[j_cat]

        # Responsive taxa: clade-focused (drug-class-like) or scattered
        p_tax_treat = float(rng.uniform(*cfg.treatment_p_tax_range))
        z_treat = bm_on_tree(tree, 1, edge_bm_sd, rng).ravel()
        if rng.random() < cfg.treatment_clade_focus_prob:
            thr_treat = np.quantile(z_treat, 1.0 - p_tax_treat)
            resp_mask = z_treat > thr_treat
            treatment_clade_focused = True
        else:
            shuffled = rng.permutation(T)
            n_resp = int(p_tax_treat * T)
            resp_mask = np.zeros(T, dtype=bool)
            resp_mask[shuffled[:n_resp]] = True

        treat_strength = float(rng.uniform(*cfg.treatment_strength_range))
        eff_mag = np.abs(bm_on_tree(tree, 1, effect_bm_sd, rng).ravel())
        # Sign bias: positive asym -> mostly negative effects (depletion)
        p_neg = 0.5 + asym / 2.0
        signs = np.where(rng.random(T) < p_neg, -1.0, 1.0)
        new_effects = eff_mag * signs * treat_strength * resp_mask.astype(float)

        # Overwrite any prior random effect on this column with the treatment signature
        effects[treat_idx] = new_effects
        edge_mask[treat_idx] = resp_mask

        treatment_cov_idx = treat_idx
        treatment_asymmetry = asym
        treatment_strength = treat_strength
        treatment_n_levels = new_n_levels
    # --- END TREATMENT INJECTION ---

    # Habitat presence/absence (validated default p_habitat_active=0.35)
    presence = np.ones((n, T), dtype=bool)
    habitat_centers = None
    habitat_dims = None
    if use_habitat and n_cont > 0 and n_hab_dims > 0:
        n_hab_dims = min(n_hab_dims, n_cont)
        habitat_dims = rng.choice(n_cont, size=n_hab_dims, replace=False)
        centers_bm = bm_on_tree(tree, n_hab_dims, 0.6, rng).T
        habitat_centers = centers_bm
        logw = bm_on_tree(tree, 1, 0.5, rng).ravel() + hab_width_log
        width = np.exp(logw)
        Xh = X_cont_true[:, habitat_dims]
        d2 = ((Xh[:, None, :] - centers_bm[None, :, :]) ** 2).sum(axis=-1)
        logit = (width[None, :] - d2) / hab_temp
        p_pres = 1.0 / (1.0 + np.exp(-logit))
        presence = rng.random((n, T)) < p_pres

    # Linear predictor and counts
    eta = X_design @ effects
    if use_nonlin and np.abs(eta).max() > 0:
        gain = rng.uniform(0.5, 1.5)
        scale = float(np.abs(eta).max())
        eta = np.tanh(eta * gain / scale) * scale

    log_lambda = log_base[None, :] + eta
    log_lambda += rng.standard_normal((n, T)) * od_scale
    pi = _safe_softmax(log_lambda, axis=1)

    lib = np.clip(rng.lognormal(log_lib_mean, log_lib_sd, size=n).astype(int),
                  100, None)

    counts = np.zeros((n, T), dtype=np.int64)
    if use_dm:
        alpha0 = np.exp(log_alpha0)
        for i in range(n):
            alpha = pi[i] * alpha0 + 1e-9
            pi_s = rng.dirichlet(alpha)
            counts[i] = rng.multinomial(lib[i], pi_s)
        count_model = 'dm'
    else:
        for i in range(n):
            counts[i] = rng.multinomial(lib[i], pi[i])
        count_model = 'ln_mult'

    if use_habitat:
        counts = counts * presence
    if zi_rate > 0:
        counts[rng.random(counts.shape) < zi_rate] = 0

    # Outcome
    y = None
    y_kind = None
    if has_y:
        rel = counts / counts.sum(1, keepdims=True).clip(1)
        log_rel = np.log(rel + 1e-6)
        clr = log_rel - log_rel.mean(axis=1, keepdims=True)
        w_taxa_bm = bm_on_tree(tree, 1, 1.0, rng).ravel()
        thr_t = np.quantile(w_taxa_bm, 1.0 - y_p_taxa)
        active_taxa = w_taxa_bm > thr_t
        w_taxa = np.where(active_taxa, w_taxa_bm, 0.0)
        if w_taxa.any():
            w_taxa = w_taxa / (np.abs(w_taxa).sum() + 1e-8)
        active_cv = rng.random(P) < y_p_cov
        w_cov = rng.standard_t(df=3, size=P) * 0.5 * active_cv
        y_lin = clr @ w_taxa + X_design @ w_cov
        if rng.random() < 0.5:
            y_lin = np.tanh(y_lin / (np.abs(y_lin).std() + 1e-8))
        y_lin = y_lin + rng.standard_normal(n) * y_noise * (np.abs(y_lin).std() + 1e-3)
        if y_binary:
            thr = np.median(y_lin) + rng.normal(0, 0.1)
            y = (y_lin > thr).astype(np.int64)
            y_kind = 'binary'
        else:
            y = y_lin.astype(np.float64)
            y_kind = 'cont'

    # Observed covariates: noise + missing
    X_cont_obs = X_cont_true + rng.standard_normal(X_cont_true.shape) * noise_scale
    X_cont_obs = X_cont_obs.astype(float)
    X_cont_obs[rng.random(X_cont_obs.shape) < missing_rate] = np.nan
    X_cat_obs = X_cat_true.astype(float).copy()
    if n_cat > 0:
        flip = rng.random(X_cat_obs.shape) < (noise_scale * 0.1)
        for j, k in enumerate(cat_levels):
            n_flip = int(flip[:, j].sum())
            if n_flip:
                X_cat_obs[flip[:, j], j] = rng.integers(0, k, size=n_flip)
        X_cat_obs[rng.random(X_cat_obs.shape) < missing_rate] = np.nan
    X_obs = np.concatenate([X_cont_obs, X_cat_obs], axis=1)
    X_true = np.concatenate([X_cont_true, X_cat_true.astype(float)], axis=1)
    kinds = ['cont'] * n_cont + ['cat'] * n_cat

    return MicrobiomeDataset(
        counts=counts, X_obs=X_obs, X_true=X_true,
        covariate_kinds=kinds, cat_levels=cat_levels,
        library_sizes=lib, true_effects=effects, edge_mask=edge_mask,
        tree=tree, habitat_centers=habitat_centers, habitat_dims=habitat_dims,
        presence_mask=presence, y=y, y_kind=y_kind,
        count_model=count_model,
        hyperparams=dict(
            n=n, T=T, P=P, n_cont=n_cont, n_cat=n_cat,
            log_lib_mean=log_lib_mean, log_lib_sd=log_lib_sd,
            p_cov=p_cov, p_tax=p_tax,
            log_base_bm_sd=log_base_bm_sd, edge_bm_sd=edge_bm_sd,
            effect_bm_sd=effect_bm_sd, yule_rate=yule_rate,
            cov_scale_df=cov_scale_df, cov_scale_loc=cov_scale_loc,
            od_scale=od_scale, use_dm=bool(use_dm), log_alpha0=log_alpha0,
            zi_rate=zi_rate, use_habitat=bool(use_habitat),
            n_hab_dims=n_hab_dims if use_habitat else 0,
            hab_width_log=hab_width_log, hab_temp=hab_temp,
            noise_scale=noise_scale, missing_rate=missing_rate,
            use_nonlin=bool(use_nonlin),
            has_y=bool(has_y), y_binary=bool(y_binary),
            treatment_cov_idx=treatment_cov_idx,
            treatment_asymmetry=treatment_asymmetry,
            treatment_strength=treatment_strength,
            treatment_n_levels=treatment_n_levels,
            treatment_clade_focused=bool(treatment_clade_focused),
        ),
    )


# ---------------------------------------------------------------------------
# Treatment diagnostic
# ---------------------------------------------------------------------------
def diagnose_treatment_distribution(n_draws: int = 80, seed: int = 456,
                                    cfg: Optional[TreatmentPriorConfig] = None):
    """Check what fraction of draws have a treatment and what its statistics
    look like across draws. Useful for sanity-checking p_treatment_study."""
    from numpy.linalg import lstsq
    rng = np.random.default_rng(seed)
    if cfg is None:
        cfg = TreatmentPriorConfig()

    treatment_r2 = []
    frac_responsive = []
    frac_depleted = []
    n_with_treatment = 0

    for _ in range(n_draws):
        ds = sample_dataset_with_treatment(cfg=cfg, rng=rng)
        treat_idx = ds.hyperparams.get('treatment_cov_idx', -1)
        if treat_idx < 0:
            continue
        n_with_treatment += 1

        rel = ds.counts / ds.library_sizes[:, None].clip(min=1)
        log_rel = np.log(rel + 1e-6)
        sample_mean = log_rel.mean(axis=1)

        x = ds.X_true[:, treat_idx:treat_idx+1]
        x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        coef, _, _, _ = lstsq(x_aug, sample_mean, rcond=None)
        pred = x_aug @ coef
        ss_res = ((sample_mean - pred) ** 2).sum()
        ss_tot = ((sample_mean - sample_mean.mean()) ** 2).sum() + 1e-12
        treatment_r2.append(1 - ss_res / ss_tot)

        eff_col = ds.true_effects[treat_idx]
        resp = eff_col != 0
        frac_responsive.append(float(resp.mean()))
        if resp.any():
            frac_depleted.append(float((eff_col[resp] < 0).mean()))

    print(f'Treatment draws: {n_with_treatment}/{n_draws} '
          f'({n_with_treatment/n_draws*100:.1f}%) — '
          f'expected ~{cfg.p_treatment_study*100:.0f}%')
    if n_with_treatment == 0:
        return
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)

    def q(a):
        return [np.quantile(a, p) for p in qs]
    print(f"\n  (over {n_with_treatment} treatment draws)        p05    p25    p50    p75    p95")
    for name, vals in [('treatment-column R² on samples', treatment_r2),
                       ('fraction responsive taxa', frac_responsive),
                       ('fraction depleted (of responsive)', frac_depleted)]:
        if not vals:
            continue
        qq = q(vals)
        print(f"  {name:<32} {qq[0]:6.3f} {qq[1]:6.3f} {qq[2]:6.3f} {qq[3]:6.3f} {qq[4]:6.3f}")


if __name__ == '__main__':
    print("Treatment distribution at p_treatment_study=0.4 (default):")
    diagnose_treatment_distribution(n_draws=100, seed=456)
    print("\n\nTreatment distribution at p_treatment_study=0.0 (extension disabled):")
    cfg_off = TreatmentPriorConfig()
    cfg_off.p_treatment_study = 0.0
    diagnose_treatment_distribution(n_draws=100, seed=456, cfg=cfg_off)
