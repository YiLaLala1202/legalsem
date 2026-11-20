# analysis

Statistical inference and figure generation for the paper's mechanism sections
(paired tests, extraction-source diagnostics, candidate-space analysis,
class-level effects). No script in this directory calls a model — everything is
recomputed from stored prediction files.

## Prerequisite

Produce predictions first with `run.py` from the repository root, for example:

```bash
python run.py extract  --dataset lcr-cn --backbone qwen3-8b --split test
python run.py classify --dataset lcr-cn --backbone qwen3-8b \
    --elements outputs/lcr-cn_qwen3-8b_extract.jsonl --mode dynamic
```

Scripts read from `outputs/lcr-cn/` and
`outputs/slard-31/`, resolved relative to this package, so the
repository can live anywhere on disk.

## Scripts

| script | produces |
| --- | --- |
| `run_stats.py` | six preset contrasts with paired bootstrap CI and Holm-corrected p-values |
| `run_candidate_space.py` | candidate-space diagnostics, error decomposition, per-class coverage |
| `run_class_effects.py` | per-class Direct vs Full effect tables (input to the figures) |
| `run_class_ci.py` | per-class paired bootstrap CI for Full − Direct on LCR-CN |
| `run_lcr_summary.py` | one-page Markdown summary of the LCR-CN mechanism analysis |
| `ci_stage_backbone.py` | stage / backbone confidence intervals |
| `plot_class_effect_lcr.py` | three standalone LCR-CN figures (PDF + PNG) |
| `plot_class_effect_slard.py` | SLARD-31 three-panel figure |
| `plot_stage_backbone.py` | stage-by-backbone figure |
| `plot_fig_motivation.py` | motivation figure |
| `plot_demo_variants.py` | worked-example variants |

All scripts are self-contained: they import only the standard library,
`numpy`, and `matplotlib`. Run them directly or as a module:

```bash
python scripts/run_stats.py --quick          # 1000 resamples, ~30 s
python scripts/run_stats.py --dataset LCR-CN
python scripts/run_candidate_space.py
python scripts/run_class_effects.py --dataset SLARD-31
python -m analysis.plot_class_effect_slard    # from the repository root
```

Outputs go to `scripts/outputs/` (CSV, LaTeX fragments, Markdown) and
`scripts/figures/` (PDF + PNG). Both directories are created on demand and are
excluded from version control.

## Candidate reconstruction

Some stored SLARD-31 prediction files record only `{id, category}` and omit
`candidate_categories`. Treating those as "no restriction" would silently report
`Cov = 100`, `ACS = 31`, `ρ = 0`. `run_candidate_space.py` therefore flags such
rows with `no_restriction=1` in the CSV and rebuilds the candidate set
deterministically from the sibling `*_test_extract.jsonl` (non-empty category
fields activate the category; an all-empty extraction falls back to the full
label space).

LCR-CN baseline files store the prediction under `conflict_type` rather than a
dedicated prediction field; `run_class_ci.py` documents and handles this.

## Multiple testing

`run_stats.py` applies Holm step-down across every pre-specified task-by-contrast
pair that has the required files present, and writes both raw and adjusted
p-values. Contrasts that could not be computed are listed as `skipped` at the
end of the run, and the Holm family size counts only the contrasts actually
evaluated.

## Figure conventions

- Panel (a): per-class F1 for Direct (hollow) vs Full (filled), sorted by Δ.
  Pale markers indicate changes within ±5 pp and should not be read as effects.
- Panel (b): the same classes across all six backbones; the colour scale is
  symmetric with fixed limits (±50 for SLARD-31, ±30 for LCR-CN).
- Panel (c): per-class candidate coverage.

Δ F1 is signed, so the diverging colour map `delta_soft` is built explicitly in
`plot_class_effect_lcr.py` (brick red for degradation → off-white at zero →
ink blue for improvement). Swap `CMAP_DELTA` to change it. All `savefig` calls
use `bbox_inches="tight"` with a fixed pad so long class names and rotated
backbone labels are not clipped.
