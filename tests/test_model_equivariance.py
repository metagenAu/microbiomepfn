"""Permutation-equivariance over taxa — the load-bearing architectural property.

The model must contain no parameter indexed by taxon identity or position. If
we permute the taxon axis of every per-taxon / per-cell input, then:

  * log_mu, log_theta, zi_logit, effect_pred  must permute the SAME way, and
  * y_pred (a per-sample readout pooled over taxa) must be INVARIANT.

If this ever regresses, the central deployment claim ("predict on any new
taxon") is false — so this is a regression test, not a nicety. Residual
differences are floating-point reduction-order noise (~1e-6), hence the 1e-4
tolerance.
"""
import torch

from microbiomepfn.model import MicrobiomePFN

D_CELL_IN = 6
D_TAX_IN = 12
D_SAMP_IN = 20
N, T = 12, 24


def _build_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    cell = torch.randn(N, T, D_CELL_IN, generator=g)
    tax = torch.randn(T, D_TAX_IN, generator=g)
    samp = torch.randn(N, D_SAMP_IN, generator=g)
    visible = torch.rand(N, T, generator=g) > 0.3
    log_lib = torch.rand(N, generator=g) * 3 + 5
    return cell, tax, samp, visible, log_lib


def _model():
    torch.manual_seed(0)
    m = MicrobiomePFN(
        d_cell_in=D_CELL_IN, d_tax_in=D_TAX_IN, d_samp_in=D_SAMP_IN,
        d=32, n_layers=2, n_heads=4, m_inducing=8, dropout=0.0,
    )
    m.eval()  # disable dropout / nondeterminism
    return m


def test_permutation_equivariance_over_taxa():
    model = _model()
    cell, tax, samp, visible, log_lib = _build_inputs()

    perm = torch.randperm(T, generator=torch.Generator().manual_seed(42))

    with torch.no_grad():
        out = model(cell, tax, samp, visible, log_lib)
        out_perm = model(
            cell[:, perm, :], tax[perm, :], samp, visible[:, perm], log_lib
        )

    atol = 1e-4

    # Per-cell / per-taxon outputs must permute equivalently.
    for key in ("log_mu", "log_theta", "zi_logit"):
        a = out[key][:, perm]
        b = out_perm[key]
        assert torch.allclose(a, b, atol=atol), (
            f"{key} not equivariant: max|diff|={(a - b).abs().max():.2e}"
        )

    a = out["effect_pred"][perm]
    b = out_perm["effect_pred"]
    assert torch.allclose(a, b, atol=atol), (
        f"effect_pred not equivariant: max|diff|={(a - b).abs().max():.2e}"
    )

    # Per-sample readout must be invariant to taxon order.
    assert torch.allclose(out["y_pred"], out_perm["y_pred"], atol=atol), (
        f"y_pred not invariant: max|diff|="
        f"{(out['y_pred'] - out_perm['y_pred']).abs().max():.2e}"
    )


def test_outputs_have_expected_shapes():
    model = _model()
    cell, tax, samp, visible, log_lib = _build_inputs()
    with torch.no_grad():
        out = model(cell, tax, samp, visible, log_lib)
    assert out["log_mu"].shape == (N, T)
    assert out["log_theta"].shape == (N, T)
    assert out["zi_logit"].shape == (N, T)
    assert out["effect_pred"].shape == (T,)
    assert out["y_pred"].shape == (N,)
