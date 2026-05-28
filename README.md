# Microbiome PFN

[![CI](https://github.com/metagenAu/microbiomepfn/actions/workflows/ci.yml/badge.svg)](https://github.com/metagenAu/microbiomepfn/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Microbiome PFN is a TabPFN-style foundation model for amplicon (16S) microbiome count
data. Instead of being fit to one dataset, it is trained on a stream of synthetic draws
from a phylogeny-aware, compositional prior that has been validated against real ASV data
via KS-distance testing on marginal statistics. The architecture is permutation-equivariant
over taxa and carries no taxon-identity parameters, so a trained model can predict on taxa
and datasets it has never seen — the basis for fine-tuning on real trials and ranking
treatment-responsive taxa via counterfactual perturbation.

## Installation

Requires Python 3.10+.

```bash
# Core package (numpy, torch, scipy)
pip install -e .

# Plus the optional dependencies used by the prior-validation notebook (pandas, matplotlib)
pip install -e .[validation]
```

This installs two console scripts: `microbiomepfn-train` and `microbiomepfn-eval`.

TabPFN-style foundation model for amplicon (16S) microbiome count data. Trained on a stream of synthetic draws from a phylogeny-aware compositional prior that has been **validated against real ASV data via KS-distance testing on marginal statistics**.

## Files

| File | Purpose |
|---|---|
| `prior.py` | **Validated prior.** KS-tested against real ASV data. Don't change defaults casually — they've been calibrated. Includes `compute_marginals()` and `summarize_dataset()` helpers used by the validation notebook. |
| `prior_treatment.py` | Optional treatment-injection extension on top of the validated prior. Off by default. Adds asymmetric, clade-focused treatment effects to a configurable fraction of draws. |
| `features.py` | Phylo PCs (BM+SVD), per-taxon marginals, per-sample features, per-cell features. Builds a `Batch` from a draw. |
| `model.py` | `MicrobiomePFN`: axial transformer with ISAB on taxon axis, full MHA on sample axis, three output heads. |
| `losses.py` | NB / ZINB log-likelihoods, total loss assembly (MLM count + y + auxiliary effect). |
| `train.py` | Main training loop. Use `--use_treatment` to enable the treatment extension. |
| `eval.py` | Held-out cell evaluation with Spearman / coverage diagnostics vs. marginal-mean baseline. |
| `deploy.py` | Inference on real data: `predict_counts`, `predict_y`, k-mer phylo features as tree fallback. |
| `finetune.py` | Fine-tune a pretrained checkpoint on real trial data, with optional trunk freezing. |
| `interpret.py` | Counterfactual treatment-effect attribution + bootstrap CIs + attention introspection. |

## What's in the validated prior

The prior is calibrated against real ASV data. Don't change its defaults casually. Important calibrated ranges:

- `log_lib_mean_range = (6.5, 10.5)` — matches real library size distributions
- `log_base_bm_sd_range = (0.15, 0.8)` — narrow BM scale on baseline log-abundance, matching real per-taxon variation
- `dm_log_alpha0_range = (1.3, 3.2)` — DM concentration floored to avoid degenerate maximum-entropy draws
- `p_habitat_active = 0.35` — habitat preferences active in 35% of draws
- `od_scale_range = (0.05, 0.8)` — overdispersion calibrated to real data
- `n_taxa_range = (50, 1500)` — capped at 1500; the validation notebook found that letting T grow much beyond real-data T pollutes marginals

The validation pipeline is in `validate_prior.ipynb` (provided separately). It draws prior-predictive datasets, computes their marginal statistics, and compares to real ASV data via 2-sample KS tests.

## Treatment extension (optional)

`prior_treatment.py` adds one extension: in 40% of draws (configurable via `p_treatment_study`), one categorical is overridden to become a "treatment" with:
- 2-4 levels (binary control/treated through multi-arm)
- Asymmetric effects (probiotic-like enrichment → balanced contrast → antibiotic-like depletion)
- Optionally phylogenetically-clustered responsive taxa (drug-class-like)

When enabled, treatment draws have:
- Median treatment-column R² ~ 0.36, range 0.04 (weak) to 0.99 (dominant axis)
- 25-55% of taxa typically responsive
- Asymmetry varies from enrichment-dominated to depletion-dominated across draws

To disable: `p_treatment_study=0.0` recovers the validated prior exactly.

**Caveat:** the treatment extension was NOT KS-validated against real data. If your deployment doesn't show asymmetric treatment patterns, training with high `p_treatment_study` may bias the model. Defensible settings:
- General training: `0.25-0.40`
- Treatment-heavy training (e.g. specifically for RCT analysis): `0.5-0.7`
- Pure validated prior (most conservative): `0.0`

## Core architectural commitments

Three commitments make "predict on any new taxon at deployment" a real claim:

**1. No parameter indexed by taxon identity or position.** No `nn.Embedding(T_max, d)`, no positional embeddings on the taxon axis, no per-column running stats. Every taxon is purely a feature vector. **Verified by direct permutation-equivariance test.**

**2. Three feature classes, all identity-free.**
- *Cell features* (N, T, 6): `log(count+1)`, `is_zero`, `log_rel`, `CLR`, `log_lib`, `is_visible`
- *Taxon features* (T, ~25): phylo PCs (16-32d) + dataset-internal marginals
- *Sample features* (N, ~70): covariates with missing indicators, log_lib, richness, shannon, log(N), log(T)

**3. Phylo coordinates via per-draw BM+SVD.** Each draw gets its own phylo basis (no cross-draw alignment). Random sign flips at training provide invariance to arbitrary basis orientation.

## Workflow

### 1. Pretrain

> For a one-click GPU run, open [`notebooks/train_t4.ipynb`](notebooks/train_t4.ipynb) in Google Colab (T4 runtime). It installs the package, pretrains with mixed precision, and evaluates a checkpoint. Mixed precision (fp16 AMP) is on by default on CUDA; add `--no_amp` to force fp32.

```bash
# With the validated prior alone:
microbiomepfn-train --n_steps 100000 --d 512 --n_layers 8 --lr 3e-4 \
                    --n_taxa_cap 800 --n_samples_cap 250 --device cuda

# With treatment extension (recommended if you'll deploy on RCT-style data):
microbiomepfn-train --n_steps 100000 --d 512 --n_layers 8 --lr 3e-4 \
                    --use_treatment --p_treatment_study 0.4 \
                    --n_taxa_cap 800 --n_samples_cap 250 --device cuda
```

### 2. Evaluate held-out cells
```bash
microbiomepfn-eval checkpoints/model_step100000.pt
```

> For a fuller validation battery — held-out prediction vs baseline, calibration curve, generalization across dataset size, permutation-equivariance of the trained weights, the compositional constraint, and treatment-effect recovery vs ground truth — open [`notebooks/validation.ipynb`](notebooks/validation.ipynb) (point `CHECKPOINT` at your model).

### 3. Fine-tune on your trials
```python
from microbiomepfn.finetune import build_real_dataset, finetune

trials = [
    build_real_dataset(counts, X, cov_kinds, cat_levels, tree, y, y_kind)
    for trial in your_real_trials
]

model, history, info = finetune(
    pretrain_checkpoint='checkpoints/pretrained.pt',
    trials=trials,
    n_steps=300, lr=3e-5,
    freeze_trunk_weights=True,           # recommended for small trials
    save_path='checkpoints/ft_yourdata.pt',
)
```

### 4. Interpret: which taxa respond to treatment
```python
from microbiomepfn.interpret import bootstrap_treatment_ranking, rank_table

boot = bootstrap_treatment_ranking(
    model, your_trial_ds,
    treatment_col=2,                       # whichever X column is the treatment
    cfg=info['cfg'],
    n_boot=200,
)
print(rank_table(boot['median'], names=taxon_names,
                 ci=(boot['lo'], boot['hi']), top_k=30,
                 title='Top responsive taxa (90% bootstrap CI)'))
```

## Verified properties (smoke tests passing)

- End-to-end pipeline runs with the validated prior swapped in
- Treatment extension produces target frequency (40% draws) and correct asymmetry distribution
- Permutation-equivariance over taxa verified (diffs ~1e-6, floating-point noise)
- Predicted relative abundances sum to exactly 1.0 per sample (compositional constraint)
- Heterogeneous T per draw works; no padding across draws

## Validation notebook

Your `validate_prior.ipynb` works with this package after one small import change. In cell 1, change:
```python
from microbiome_prior import (
    PriorConfig, sample_dataset, compute_marginals,
)
```
to:
```python
from microbiomepfn.prior import (
    PriorConfig, sample_dataset, compute_marginals,
)
```

All three functions exist in `microbiomepfn.prior` with the same signatures.

## Deployment-to-soil-trials workflow

The intended use case for this package, in order:

1. **Pretrain** with `microbiomepfn-train --use_treatment` on a GPU for as long as you can afford. The treatment extension matters for soil RCT data.
2. **External validation** before fine-tuning: run the pretrained model on a public soil dataset where DA has already been published (e.g., Earth Microbiome Project soil samples) and compare top-N responsive taxa to published findings. If they don't agree, debug before moving forward.
3. **Fine-tune** with frozen trunk on your trial data via `finetune.py`. Hold out 20% of samples per trial; watch validation NLL.
4. **Interpret** with `interpret.bootstrap_treatment_ranking`. Rank taxa by robust CI (excludes zero), not by point estimate alone.
5. **Confirm with conventional DA**. Run MaAsLin2 or ANCOM-BC2 on the same data; report only taxa where both methods agree as confirmed hits. Use disagreements as a debugging signal, not as model failure.

## Known limitations

- **Sample feature dim padded to 200.** Hacky. Cleaner long-term solution: covariates-as-tokens.
- **No phylo attention bias.** Phylo info enters via per-taxon features only.
- **Effect head predicts L2 norm of effects** as weak auxiliary; for actual differential-abundance interpretation, use `interpret.py` counterfactual perturbation, not the effect head.
- **CPU is too slow** for serious training. Use a GPU — mixed precision (fp16 AMP) is built into `train.py` and on by default on CUDA; serious runs still need ~1M+ steps.
- **Treatment extension is not KS-validated.** Use it judiciously and check whether your real deployment data shows the asymmetric patterns it models.
