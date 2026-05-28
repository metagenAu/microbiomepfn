# Notes from bootstrap

This file records things noticed while packaging the existing source into a
proper Python project. **In the initial bootstrap commit, no scientific/numerical
logic was changed** (a later, explicitly-requested change added mixed precision —
see "Post-bootstrap changes" at the bottom). The only edits to source files in the
bootstrap commit were:

1. Rewriting intra-package imports (`from prior import …` → `from microbiomepfn.prior import …`, etc.).
2. Adding two thin `main()` wrappers (in `train.py` and `eval.py`) around the
   pre-existing `if __name__ == "__main__"` CLI blocks, so the console scripts
   `microbiomepfn-train` / `microbiomepfn-eval` have an entry point. The wrapped
   logic is unchanged (in `eval.py` a redundant function-local `import numpy as np`
   was dropped because the module already imports it at top level).

Everything below is **flagged, not fixed** — decide what (if anything) you want
to do about each.

## Potential bugs / inconsistencies

### 1. README workflow example references a key `finetune()` does not return
The "Interpret" workflow snippet (carried over verbatim from the original
README) passes `cfg=info['cfg']`:

```python
boot = bootstrap_treatment_ranking(model, your_trial_ds, treatment_col=2,
                                   cfg=info['cfg'], n_boot=200)
```

But `finetune()` returns `info = dict(train_trials=…, val_trials=…, best_val_nll=…)`
— there is **no `'cfg'` key**, so this line would raise `KeyError`. The `cfg`
that `interpret.*` functions need is the *model config* dict (the one with
`k_phylo` and `d_samp_in`), which you get from `deploy.load_model(...)` or the
saved checkpoint, not from `finetune()`'s return value. Left as-is in the README;
fix the example (or have `finetune` also return `cfg`) when convenient.

### 2. Dead local variable in `interpret.attention_introspection`
`interpret.py` (~line 344) computes `log_lib = torch.log(...)` but never uses it
— `attention_introspection` runs the trunk manually (`model.embed` → blocks →
`final_ln` → `y_pool`) and never calls the count head, so the library offset is
not needed. Harmless dead computation (this is the single `F841` ruff finding).

### 3. Comment drift in `model.py` count head
The `count_head` comment (~line 187) labels its 3 outputs
`[log_mu_per_sample, log_theta, zi_logit]`, but the forward pass uses output `[0]`
as `log_p_logit` — a per-taxon relative-abundance logit that is `log_softmax`'d
over taxa and then offset by `log_library` to form `log_mu`. The code is
internally consistent; only the comment is slightly misleading.

## Lint suppressions (intentional, not fixed)

`ruff check` flags 21 pre-existing nits in the moved source. Rather than perform
"while I'm here" cleanups, these rule codes are ignored **for `src/microbiomepfn/*`
only** (test files are held to the full rule set) via `[tool.ruff.lint.per-file-ignores]`
in `pyproject.toml`:

- `F401` (17×) — unused imports, e.g. `typing.Union` in `deploy.py`, `import math`
  in `losses.py`, `Tuple` in `model.py`, etc.
- `E702` (4×) — semicolon-joined statements in `prior.sample_yule_tree`.
- `F841` (1×) — the dead `log_lib` above.
- `E741` (2×) — intentional single-letter names: `I` (ISAB inducing-point matrix
  in `model.py`) and `l` (loop variable in the Yule-tree sampler).

If you'd rather lint these strictly, drop the corresponding codes from the
per-file-ignore list and clean them up — they're all low-risk.

## Cosmetic / housekeeping

- The CI badge in `README.md` uses a literal `OWNER` placeholder in the GitHub
  Actions URL — replace it with your org/user after the repo is pushed.
- `LICENSE` copyright line reads `Copyright (c) 2026 Metagen` (inferred from the
  configured author email). Adjust the holder name if that's not right.

## Post-bootstrap changes

### Mixed precision (AMP) in `train.py`
Added on request to make training on a T4-class GPU practical. This *does* touch
the training loop, deliberately:

- `train_step(...)` gained `scaler` and `use_amp` params. When `use_amp=True` the
  model forward runs under `torch.autocast(dtype=float16)` and backward/step go
  through a `GradScaler`. The NB/ZINB loss (lgamma/exp) is unsafe in fp16, so model
  outputs are upcast to fp32 before `compute_loss`. **With `use_amp=False` (the
  default for `train_step`) the path is the original fp32 code, unchanged.**
- `train(...)` gained `use_amp=True` (only active on CUDA — fp32 on CPU) and builds
  the `GradScaler`; the CLI gained `--no_amp`.
- The fp16 AMP branch is **CUDA-only and not unit-tested**: CPU autocast supports
  bfloat16, not fp16, so it can't be exercised on the CI runners. `test_train_smoke.py`
  pins the fp32 (`use_amp=False`) path; the AMP branch is exercised manually via
  `notebooks/train_t4.ipynb` on a real GPU.

See `notebooks/train_t4.ipynb` for a Colab/T4 training + evaluation walkthrough.

### Generated-draw size caps in `train.py`
`train(...)` (and the CLI) gained `max_gen_taxa` / `max_gen_samples`. The prior
otherwise samples up to `n_taxa_range[1]=1500` taxa / `n_samples_range[1]=500`
samples *every step* and then subsamples down to `n_taxa_cap` / `n_samples_cap`
— so the heavy CPU sampling (tree, BM, Dirichlet-multinomial) runs at full size
regardless of the caps. These knobs shrink the *generated* ranges to cut that
cost. Defaults are `None` (validated ranges untouched). `--y_weight` /
`--effect_weight` were also surfaced on the CLI (they already existed on `train()`).

## Findings from training runs (flag, not fixed)

### The outcome (`y`) head cannot learn as currently wired
Observed in a 78k-step run: the **binary `y` loss stays pinned at 0.693 = ln 2
(chance) indefinitely**, and continuous `y` loss hovers at ~1.0 (= variance of the
standardized target, i.e. "predict the mean"). This is a structural limitation,
not undertraining:

- The model's inputs are `cell_feats`, `tax_feats`, `samp_feats`. **The label `y`
  is never fed into the model** — `features.sample_features` includes the
  context/query *visibility flag* but not the label values.
- Each prior draw generates a **fresh random `y` function** (random sparse
  `w_taxa`, `w_cov`). So there is no fixed features→`y` mapping to learn across
  draws, and the model is never shown the draw's context labels to infer the
  draw-specific function (as a TabPFN-style ICL setup would require).
- `deploy.predict_y` has the same gap: it sets `visible_sample = context_mask`
  but never passes `y_context` into the model.

Net: the `y`/few-shot-outcome head optimizes to "predict the marginal," which is
exactly ln 2 for balanced binary. The **count/MLM task is unaffected and sound**
(the model does see visible cells and predict masked ones), and the README's
headline use — differential abundance via `interpret.py` counterfactuals — runs
through the count head, not `y`. But the few-shot `y`-prediction capability needs
a real design change (feed context labels into the input) to work. Left unfixed.

### Held-out cell prediction is conditional-structure-dependent (not a bug)
On a no-treatment 78k checkpoint, aggregate held-out Spearman ρ barely beat the
marginal-mean baseline (+0.013), but stratifying by per-draw covariate structure
showed the model **+0.028 on high-structure draws** and **−0.024 on
near-marginal draws**. So the model does learn covariate→taxon structure; the flat
aggregate is dilution from marginal-dominated draws (where there is little to gain
and the model adds slight harmful variance). Setting `y_weight=0`/`effect_weight=0`
(count-focused) and training treatment-aware are the recommended next levers — no
code change required for the loss weights.
