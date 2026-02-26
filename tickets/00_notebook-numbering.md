# Clean up notebook numbering to match phase plans

**Type:** tech-debt
**Priority:** medium
**Noticed:** 2026-02-26
**Status:** resolved
**Phase:** general
**Resolved:** 2026-02-26 — renamed all notebooks to `{phase}-{seq}_{name}` scheme

## Context

Notebooks were created organically during Phase 2–4 work. Their numbering conflicted with phase plans and didn't indicate which phase each notebook belonged to.

## Resolution

**Decision: adopt `{phase}-{seq}_{name}` naming so notebooks sort by phase.**

All notebooks renamed and all references updated (phase plans, cross-references in notebooks, HTML reports).

Renames applied:
- `00_data_pipeline_test.ipynb` → `02-01_data_pipeline_test.ipynb`
- `01_data_exploration.ipynb` → `03-01_data_exploration.ipynb`
- `02_umap_exploration.ipynb` → `03-02_umap_exploration.ipynb`
- `03_rasterization.ipynb` → `04-01_rasterization.ipynb`
- `04_baseline_lgbm.ipynb` → `05-01_baseline_lgbm.ipynb`

Reports also renamed:
- `reports/eda-findings.ipynb` → `reports/03-01_eda-findings.ipynb` (+ `.html`)
- `reports/rasterization-verification.html` → `reports/04-01_rasterization-verification.html`

References updated in:
- `plans/03-eda.md`, `plans/04-rasterization.md`, `plans/05-baseline-model.md`, `plans/06-cnn-model.md`
- `notebooks/03-01_data_exploration.ipynb` (UMAP cross-ref, report links)
- `notebooks/03-02_umap_exploration.ipynb` (title)
- `notebooks/04-01_rasterization.ipynb` (UMAP cross-ref)
- `reports/03-01_eda-findings.ipynb` and `reports/03-01_eda-findings.html`
- `reports/04-01_rasterization-verification.html`
- `CLAUDE.md` (reports table, EDA findings reference)

Final notebook numbering:

| Notebook | Phase |
|----------|-------|
| `02-01_data_pipeline_test.ipynb` | 2 (Data Pipeline) |
| `03-01_data_exploration.ipynb` | 3 (EDA) |
| `03-02_umap_exploration.ipynb` | 3 (EDA) |
| `04-01_rasterization.ipynb` | 4 (Rasterization) |
| `05-01_baseline_lgbm.ipynb` | 5 (Baseline Model) |
| `06-01_cnn_training.ipynb` | 6 (CNN Model) — future |

## Acceptance Criteria

- [x] All existing notebooks renamed to `{phase}-{seq}_{name}` scheme
- [x] All plan references point to correct notebook names
- [x] No numbering collisions between existing and planned notebooks
