"""Feature-assembly checks: batch shapes and that cell masking blanks the
count-derived channels while preserving library size and the visibility flag.

Cell-feature channel order (see features.cell_features):
    0 log(count+1)   1 is_zero   2 log_rel   3 CLR   4 log_lib   5 is_visible
Channels 0-3 and 5 are gated by visibility; channel 4 (log_lib) is kept for
masked cells by design.
"""
import numpy as np

from microbiomepfn.prior import PriorConfig, sample_dataset
from microbiomepfn.features import build_batch

K_PHYLO = 16
N_TAXON_MARGINALS = 9  # taxon_marginals emits 9 stats per taxon


def tiny_ds(seed=0):
    cfg = PriorConfig()
    cfg.n_samples_range = (30, 40)
    cfg.n_taxa_range = (50, 60)
    cfg.n_covariates_range = (4, 8)
    return sample_dataset(cfg=cfg, rng=np.random.default_rng(seed))


def test_build_batch_shapes():
    ds = tiny_ds()
    rng = np.random.default_rng(1)
    batch = build_batch(ds, rng, k_phylo=K_PHYLO,
                        cell_mask_frac=0.3, sample_query_frac=0.3)
    N, T = ds.counts.shape

    assert batch.N == N and batch.T == T
    assert batch.cell_feats.shape == (N, T, 6)
    assert batch.taxon_feats.shape == (T, K_PHYLO + N_TAXON_MARGINALS)
    assert batch.sample_feats.shape[0] == N
    assert batch.visible_cell.shape == (N, T)
    assert batch.visible_cell.dtype == bool
    assert batch.counts.shape == (N, T)


def test_cell_masking_blanks_count_channels():
    ds = tiny_ds(seed=2)
    rng = np.random.default_rng(2)
    batch = build_batch(ds, rng, k_phylo=K_PHYLO,
                        cell_mask_frac=0.3, sample_query_frac=0.3)

    masked = ~batch.visible_cell
    # Masking actually happened (some masked, some visible).
    assert masked.any() and (~masked).any()

    cf = batch.cell_feats
    for ch in (0, 1, 2, 3):  # count-derived channels
        assert np.allclose(cf[..., ch][masked], 0.0), f"channel {ch} not blanked"

    # Visibility flag (channel 5) matches the mask exactly.
    assert np.array_equal(cf[..., 5].astype(bool), batch.visible_cell)


def test_taxon_marginals_finite():
    ds = tiny_ds(seed=3)
    rng = np.random.default_rng(3)
    batch = build_batch(ds, rng, k_phylo=K_PHYLO)
    assert np.isfinite(batch.taxon_feats).all()
    assert np.isfinite(batch.sample_feats).all()
    assert np.isfinite(batch.cell_feats).all()
