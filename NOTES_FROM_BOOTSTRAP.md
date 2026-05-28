# Notes from bootstrap

This file records things noticed while packaging the existing source into a
proper Python project. **No scientific/numerical logic was changed.** The only
edits to source files were:

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
