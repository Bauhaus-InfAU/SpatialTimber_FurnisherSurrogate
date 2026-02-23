# Phase 5: Baseline Model (LightGBM)

[← Overview](../PLAN.md) | Depends on: [Data Pipeline](02-data-pipeline.md), [EDA](03-eda.md)

## Goal

Train a simple tabular model as sanity check. Answer: "Can you predict score from area + shape + room type?"

**Notebook**: `notebooks/02_baseline_tabular.ipynb`

## Tasks

- [ ] Train LightGBM regressor with W&B logging
- [ ] Evaluate on validation set (MAE, RMSE, R², per-room-type MAE)
- [ ] Feature importance analysis
- [ ] Establish performance floor (record baseline MAE in Decisions Log)

## Model: LightGBM Regressor

- **Input**: 13 features (area, aspect_ratio, n_vertices, room_type one-hot)
- **Target**: score (0–100)
- **Split**: Same apartment-level 80/10/10 as CNN

**Starting hyperparameters**:

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| learning_rate | 0.05 |
| max_depth | 6 |
| num_leaves | 31 |

## Hyperparameter Tuning

Minimal tuning — this is a sanity check model.

| Parameter | Range | Impact |
|-----------|-------|--------|
| `n_estimators` | 100–2000 | High (coupled with lr) |
| `learning_rate` | 0.01–0.2 | High |
| `num_leaves` | 15–63 | Medium |
| `min_child_samples` | 10–100 | Medium |

## Evaluation

| Metric | Purpose |
|--------|---------|
| MAE | Primary — average error in score points |
| RMSE | Penalizes large errors |
| R² | Overall explained variance |
| Per-room-type MAE | Struggles with specific rooms? |

## Interpretation

- **MAE < 15**: Geometry clearly determines score. Baseline is strong.
- **MAE > 25**: Spatial detail matters more than summary stats. CNN should help.

## Decisions Log

*(Record decisions here as they're made)*
