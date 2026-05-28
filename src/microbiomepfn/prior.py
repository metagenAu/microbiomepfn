"""
Synthetic microbiome amplicon-count prior, inspired by TabPFN's SCM-based prior.

This version adds four extensions on top of the v1 sketch:

  1. Dirichlet-multinomial branch  (alternates with logistic-normal multinomial)
  2. Random Yule tree + Brownian-motion effects along the tree
  3. Compositional structural zeros via per-taxon habitat preferences
  4. Optional outcome head y from a sparse random function of (counts, X)

Per-dataset generative flow:

  hyperparams ~ HyperPrior
  tree ~ Yule(T tips)
  log_base[t] = BM_tree(sigma_base)              # phylo signal in baseline
  for each active covariate p:
      edge_mask[p,t]  = (BM_tree(z) > thr_p)     # phylo-clustered active edges
      effects[p,t]    = BM_tree(eff_scale_p) * edge_mask[p,t]
  habitat[t] ~ position in X-space; width ~ BM_tree
  presence[i,t] = Bernoulli(sigmoid((width_t - dist(X_i, habitat_t)) / temp))
  eta[i,t] = (X @ effects)[i,t]  (optional tanh)
  log_lambda = log_base + eta + N(0, od_scale)
  pi = softmax(log_lambda)
  counts ~ branch:
        Multinomial(N, pi)              # LN-multinomial branch
     or DirichletMultinomial(N, alpha)  # DM branch with sampled concentration
  counts[~presence | extra_ZI] = 0
  y (optional) = f_sparse(log(rel)+CLR, X) + noise
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


# ===========================================================================
# Hyperprior config
# ===========================================================================
@dataclass
class PriorConfig:
    n_samples_range: tuple = (30, 500)
    n_taxa_range: tuple = (50, 1500)
    n_covariates_range: tuple = (3, 30)
    p_continuous: float = 0.7

    # Library size
    log_lib_mean_range: tuple = (6.5, 10.5)
    log_lib_sd_range: tuple = (0.1, 0.8)

    # Phylogenetic BM scales
    log_base_bm_sd_range: tuple = (0.15, 0.8)     # BM scale for baseline log-abund
    edge_bm_sd_range: tuple = (0.5, 2.0)         # BM scale for activity field
    effect_bm_sd_range: tuple = (0.1, 1.2)       # BM scale for effect magnitudes
    yule_rate_range: tuple = (0.5, 2.0)

    # Sparsity (active covariates / per-cov active fraction)
    p_cov_active_beta: tuple = (1.0, 9.0)
    p_taxon_responsive_beta: tuple = (1.0, 4.0)

    # Heavy-tailed per-covariate scale multiplier
    cov_scale_df_range: tuple = (2.0, 8.0)
    cov_scale_loc_range: tuple = (0.1, 1.5)

    # Per-sample/taxon overdispersion (log-normal noise on log_lambda)
    od_scale_range: tuple = (0.05, 0.8)

    # Count model branch
    p_dirichlet_multinomial: float = 0.5
    dm_log_alpha0_range: tuple = (1.3, 3.2)       # DM concentration scale; floored to avoid degeneracy

    # Extra random zero-inflation rate (on top of habitat-based zeros)
    zi_beta: tuple = (1.0, 8.0)
    zi_max: float = 0.4

    # Habitat-preference structural zeros
    p_habitat_active: float = 0.35               # P(this dataset uses habitats)
    n_habitat_dims_range: tuple = (1, 4)         # how many cov dims define habitat
    habitat_width_log_range: tuple = (0.5, 3.0)  # log of base width
    habitat_temp_range: tuple = (0.3, 2.0)

    # Covariate observation noise & missingness
    noise_scale_rate: float = 2.0
    missing_rate_beta: tuple = (1.0, 9.0)

    # Nonlinearity
    p_nonlinear: float = 0.5

    # Outcome head
    p_outcome: float = 0.7                       # P(produce y)
    p_outcome_binary: float = 0.5                # if y exists, P(binarize)
    outcome_p_taxa_active_beta: tuple = (1.0, 8.0)
    outcome_p_cov_active_beta: tuple = (1.0, 5.0)
    outcome_noise_rate: float = 2.0


# ===========================================================================
# Output containers
# ===========================================================================
@dataclass
class Tree:
    parent: np.ndarray            # (2T-1,) parent index; -1 at root
    branch_len: np.ndarray        # (2T-1,)
    tip_ids: np.ndarray           # (T,) indices into the node array
    preorder: np.ndarray          # (2T-1,) traversal from root
    root: int


@dataclass
class MicrobiomeDataset:
    counts: np.ndarray
    X_obs: np.ndarray
    X_true: np.ndarray
    covariate_kinds: List[str]
    cat_levels: List[int]
    library_sizes: np.ndarray
    true_effects: np.ndarray
    edge_mask: np.ndarray
    tree: Tree
    habitat_centers: Optional[np.ndarray]        # (T, n_habitat_dims) or None
    habitat_dims: Optional[np.ndarray]           # which X dims are habitat axes
    presence_mask: np.ndarray                    # (n, T) bool: structural presence
    y: Optional[np.ndarray]                      # outcome, may be None
    y_kind: Optional[str]                        # 'cont', 'binary', or None
    count_model: str                             # 'ln_mult' or 'dm'
    hyperparams: dict


# ===========================================================================
# Tree sampling + Brownian motion
# ===========================================================================
def sample_yule_tree(T: int, rng: np.random.Generator, lam: float = 1.0) -> Tree:
    """Forward Yule simulation. Tips end up as the first T 'active' lineages."""
    if T == 1:
        return Tree(parent=np.array([-1]), branch_len=np.array([0.0]),
                    tip_ids=np.array([0]), preorder=np.array([0]), root=0)

    parents: List[int] = [-1]
    branch: List[float] = [0.0]
    active: List[int] = [0]
    while len(active) < T:
        dt = rng.exponential(1.0 / (lam * len(active)))
        for l in active:
            branch[l] += dt
        i = int(rng.integers(len(active)))
        node = active[i]
        c1 = len(parents); parents.append(node); branch.append(0.0)
        c2 = len(parents); parents.append(node); branch.append(0.0)
        active[i] = c1
        active.append(c2)

    parent = np.asarray(parents, dtype=np.int64)
    branch_len = np.asarray(branch)
    n_total = len(parent)

    # Children list for pre-order traversal
    kids: List[List[int]] = [[] for _ in range(n_total)]
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    pre = []
    stack = [0]
    while stack:
        v = stack.pop()
        pre.append(v)
        stack.extend(kids[v])
    return Tree(parent=parent, branch_len=branch_len,
                tip_ids=np.asarray(active, dtype=np.int64),
                preorder=np.asarray(pre, dtype=np.int64), root=0)


def bm_on_tree(tree: Tree, k: int, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """Sample k independent BM processes along the tree.
    Returns (k, T) values at the tips."""
    n_total = len(tree.parent)
    vals = np.zeros((k, n_total))
    for node in tree.preorder:
        p = tree.parent[node]
        if p >= 0:
            bl = tree.branch_len[node]
            vals[:, node] = (vals[:, p]
                             + rng.standard_normal(k) * np.sqrt(max(bl, 1e-12)) * sigma)
    return vals[:, tree.tip_ids]


# ===========================================================================
# Helpers
# ===========================================================================
def _safe_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _sample_continuous_X(n: int, n_cont: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((n, n_cont))
    if n_cont > 2:
        k = max(1, n_cont // 4)
        L = rng.standard_normal((n_cont, k)) * 0.5
        X += rng.standard_normal((n, k)) @ L.T
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X


# ===========================================================================
# Main sampler
# ===========================================================================
def sample_dataset(cfg: Optional[PriorConfig] = None,
                   rng: Optional[np.random.Generator] = None
                   ) -> MicrobiomeDataset:
    if cfg is None:
        cfg = PriorConfig()
    if rng is None:
        rng = np.random.default_rng()

    # ---------- Dataset-level hyperparameters ----------
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

    # ---------- Tree ----------
    tree = sample_yule_tree(T, rng, lam=yule_rate)

    # ---------- Baseline log abundances (BM along tree) ----------
    log_base = bm_on_tree(tree, 1, log_base_bm_sd, rng).ravel()

    # ---------- Latent covariates ----------
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

    # ---------- Structural graph: BM-driven activity & effects ----------
    active_cov = rng.random(P) < p_cov
    n_active = int(active_cov.sum())

    edge_mask = np.zeros((P, T), dtype=bool)
    effects = np.zeros((P, T))

    if n_active > 0:
        # Activity field per active covariate via BM, thresholded to hit p_tax
        z = bm_on_tree(tree, n_active, edge_bm_sd, rng)             # (n_active, T)
        thr = np.quantile(z, 1.0 - p_tax, axis=1, keepdims=True)
        act = z > thr                                                 # (n_active, T)

        # Effect magnitudes via BM on tree
        e_bm = bm_on_tree(tree, n_active, effect_bm_sd, rng)         # (n_active, T)
        # Heavy-tailed per-covariate scale to allow occasional strong covariates
        per_cov_scale = (cov_scale_loc * np.abs(
            rng.standard_t(df=cov_scale_df, size=(n_active, 1))))
        e_active = e_bm * per_cov_scale * act

        idx = np.where(active_cov)[0]
        edge_mask[idx] = act
        effects[idx] = e_active

    # ---------- Habitat-preference presence ----------
    presence = np.ones((n, T), dtype=bool)
    habitat_centers = None
    habitat_dims = None
    if use_habitat and n_cont > 0 and n_hab_dims > 0:
        n_hab_dims = min(n_hab_dims, n_cont)
        habitat_dims = rng.choice(n_cont, size=n_hab_dims, replace=False)
        # Per-taxon habitat centers (phylo-correlated via BM along tree)
        centers_bm = bm_on_tree(tree, n_hab_dims, 0.6, rng).T          # (T, n_hab_dims)
        habitat_centers = centers_bm
        # Per-taxon log-width (also phylo-correlated)
        logw = bm_on_tree(tree, 1, 0.5, rng).ravel() + hab_width_log    # (T,)
        width = np.exp(logw)                                            # (T,)
        # Distance² between samples and taxa habitat centers
        Xh = X_cont_true[:, habitat_dims]                               # (n, h)
        # d2[i,t] = sum_d (Xh[i,d] - C[t,d])^2
        d2 = (Xh[:, None, :] - centers_bm[None, :, :]) ** 2
        d2 = d2.sum(axis=-1)                                            # (n, T)
        logit = (width[None, :] - d2) / hab_temp
        p_pres = 1.0 / (1.0 + np.exp(-logit))
        presence = rng.random((n, T)) < p_pres

    # ---------- Linear predictor ----------
    eta = X_design @ effects
    if use_nonlin and np.abs(eta).max() > 0:
        gain = rng.uniform(0.5, 1.5)
        scale = float(np.abs(eta).max())
        eta = np.tanh(eta * gain / scale) * scale

    log_lambda = log_base[None, :] + eta
    log_lambda += rng.standard_normal((n, T)) * od_scale

    pi = _safe_softmax(log_lambda, axis=1)

    # ---------- Library sizes ----------
    lib = np.clip(rng.lognormal(log_lib_mean, log_lib_sd, size=n).astype(int),
                  100, None)

    # ---------- Count model ----------
    counts = np.zeros((n, T), dtype=np.int64)
    if use_dm:
        # Dirichlet-multinomial: pi_sample ~ Dir(alpha0 * pi); counts ~ Multinom(N, pi_sample)
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

    # ---------- Apply structural zeros from habitat ----------
    if use_habitat:
        counts = counts * presence

    # ---------- Extra random ZI ----------
    if zi_rate > 0:
        counts[rng.random(counts.shape) < zi_rate] = 0

    # ---------- Outcome head ----------
    y = None
    y_kind = None
    if has_y:
        # CLR-like taxa features
        rel = counts / counts.sum(1, keepdims=True).clip(1)
        log_rel = np.log(rel + 1e-6)
        clr = log_rel - log_rel.mean(axis=1, keepdims=True)

        # Sparse taxa weights (phylo-clustered): use BM-on-tree thresholded
        w_taxa_bm = bm_on_tree(tree, 1, 1.0, rng).ravel()
        thr_t = np.quantile(w_taxa_bm, 1.0 - y_p_taxa)
        active_taxa = w_taxa_bm > thr_t
        w_taxa = np.where(active_taxa, w_taxa_bm, 0.0)
        if w_taxa.any():
            w_taxa = w_taxa / (np.abs(w_taxa).sum() + 1e-8)            # normalize

        # Sparse covariate weights
        active_cv = rng.random(P) < y_p_cov
        w_cov = rng.standard_t(df=3, size=P) * 0.5 * active_cv

        y_lin = clr @ w_taxa + X_design @ w_cov
        # Optional nonlinearity
        if rng.random() < 0.5:
            y_lin = np.tanh(y_lin / (np.abs(y_lin).std() + 1e-8))
        y_lin = y_lin + rng.standard_normal(n) * y_noise * (np.abs(y_lin).std() + 1e-3)

        if y_binary:
            # threshold at the median for ~balanced classes; jitter
            thr = np.median(y_lin) + rng.normal(0, 0.1)
            y = (y_lin > thr).astype(np.int64)
            y_kind = 'binary'
        else:
            y = y_lin.astype(np.float64)
            y_kind = 'cont'

    # ---------- Observed covariates: noise + missing ----------
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
        ),
    )


# ===========================================================================
# Marginal-statistics computation (used by both prior and real data)
# ===========================================================================
def compute_marginals(counts: np.ndarray, min_richness: int = 2
                       ) -> Dict[str, np.ndarray]:
    """Microbiome-flavored marginal stats for prior-predictive checks.
    Empty samples and samples with richness < min_richness are filtered, as are
    zero-prevalence taxa — matching how real ASV tables are QC'd upstream."""
    counts = np.asarray(counts)
    rich_all = (counts > 0).sum(1)
    counts = counts[rich_all >= min_richness]    # drop empty + single-taxon samples
    lib = counts.sum(1)
    safe_lib = lib.clip(1)
    rel = counts / safe_lib[:, None]
    richness = (counts > 0).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        shannon = -(rel * np.log(rel.clip(min=1e-300))).sum(1)
        pielou = np.where(richness > 1,
                          shannon / np.log(richness.clip(2)), 0.0)
    prevalence_all = (counts > 0).mean(0)
    # Drop never-present taxa: real data filters these out upstream
    keep = prevalence_all > 0
    rel_kept = rel[:, keep]
    prevalence = prevalence_all[keep]
    mean_t = rel_kept.mean(0).clip(1e-12)
    var_t  = rel_kept.var(0).clip(1e-24)
    log_mean_t = np.log(mean_t)
    log_var_t  = np.log(var_t)
    log_mean_abund = log_mean_t
    return dict(
        library_size=lib.astype(float),
        log_library_size=np.log(safe_lib),
        richness=richness.astype(float),
        shannon=shannon,
        pielou=pielou,
        prevalence=prevalence,
        log_mean_abund=log_mean_abund,
        log_mean_t=log_mean_t,
        log_var_t=log_var_t,
        max_rel=rel.max(1),
        zero_frac_per_sample=(counts == 0).mean(1),
    )


def summarize_dataset(ds: MicrobiomeDataset) -> dict:
    m = compute_marginals(ds.counts)
    out = dict(
        n=ds.hyperparams['n'], T=ds.hyperparams['T'],
        count_model=ds.count_model,
        zero_fraction=float((ds.counts == 0).mean()),
        richness_mean=float(m['richness'].mean()),
        richness_sd=float(m['richness'].std()),
        lib_size_mean=float(ds.library_sizes.mean()),
        lib_size_cv=float(ds.library_sizes.std() / max(ds.library_sizes.mean(), 1)),
        shannon_mean=float(m['shannon'].mean()),
        pielou_mean=float(m['pielou'].mean()),
        max_rel_abund_mean=float(m['max_rel'].mean()),
        cov_missing_frac=float(np.isnan(ds.X_obs).mean()),
        edge_density=float(ds.edge_mask.mean()),
        n_active_edges=int(ds.edge_mask.sum()),
        has_y=ds.y is not None,
        y_kind=ds.y_kind,
    )
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Prior-predictive sweep: 10 datasets\n")
    print(f"{'#':>2} {'n':>4} {'T':>5} {'mdl':>4} {'zero%':>6} {'rich':>11} "
          f"{'shan':>5} {'lib_CV':>7} {'miss%':>6} {'edges':>6} "
          f"{'hab':>4} {'nlin':>5} {'y':>5}")
    for i in range(10):
        ds = sample_dataset(rng=rng)
        s = summarize_dataset(ds)
        print(f"{i+1:>2} {s['n']:>4} {s['T']:>5} "
              f"{s['count_model']:>4} "
              f"{s['zero_fraction']*100:>5.1f}% "
              f"{s['richness_mean']:>5.0f}±{s['richness_sd']:>3.0f}  "
              f"{s['shannon_mean']:>5.2f} "
              f"{s['lib_size_cv']:>7.2f} "
              f"{s['cov_missing_frac']*100:>5.1f}% "
              f"{s['n_active_edges']:>6d} "
              f"{str(ds.hyperparams['use_habitat']):>4} "
              f"{str(ds.hyperparams['use_nonlin']):>5} "
              f"{str(s['y_kind']):>5}")
