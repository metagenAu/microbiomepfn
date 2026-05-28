"""Microbiome PFN: a TabPFN-style foundation model for amplicon microbiome data.

Submodules are imported explicitly (e.g. ``from microbiomepfn.prior import
sample_dataset``) rather than eagerly here, so that ``import microbiomepfn`` stays
cheap and the numpy-only prior can be used without pulling in torch.
"""

__version__ = "0.1.0"
