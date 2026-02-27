# Phase 5: Baseline Model (LightGBM)

[← Overview](../PLAN.md) | Depends on: [Data Pipeline](02-data-pipeline.md), [EDA](03-eda.md)

## Goal

Train a simple tabular model as sanity check. Answer: "Can you predict score from area + shape + room type?"

**Notebook**: `notebooks/05-01_baseline_lgbm.ipynb`

## Tasks

- [x] Train LightGBM regressor with W&B logging
- [x] Evaluate on validation set (MAE, RMSE, R², per-room-type MAE)
- [x] Feature importance analysis
- [x] Establish performance floor (record baseline MAE in Decisions Log)
- [x] Retrain baseline with apartment_type feature (14→21 features)

## Model: LightGBM Regressor

- **Input**: 14 features (area, aspect_ratio, n_vertices, door_rel_x, door_rel_y, 9 room_type one-hot)
- **Target**: score (0–100)
- **Split**: Same apartment-level 80/10/10 as CNN
- **Optimal rounds**: 457 (from 5-fold CV with early stopping, patience=50)

**Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| n_estimators | 457 (CV-selected from max 2000) |
| learning_rate | 0.05 |
| max_depth | 6 |
| num_leaves | 31 |
| min_child_samples | 20 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |

## Results

| Metric | Val | Test |
|--------|-----|------|
| MAE | 10.87 | 11.02 |
| RMSE | 18.95 | 19.15 |
| R² | 0.7949 | 0.8000 |
| Fail/Pass Acc | 0.8833 | 0.8744 |
| Fail/Pass F1 | 0.9260 | 0.9182 |

**Naive MAE**: 37.48 → **71% improvement** over predicting room-type mean.

### Per-Room-Type MAE (test)

| Room type | n | MAE | Naive MAE |
|-----------|---|-----|-----------|
| Bathroom | 833 | 3.14 | 30.22 |
| Children 1 | 494 | 6.61 | 34.21 |
| Children 2 | 286 | 7.48 | 33.42 |
| Children 4 | 63 | 8.97 | 36.08 |
| Children 3 | 162 | 9.48 | 32.61 |
| Bedroom | 744 | 10.29 | 40.29 |
| WC | 429 | 10.87 | 42.62 |
| Kitchen | 833 | 16.89 | 43.15 |
| Living room | 750 | 18.84 | 39.13 |

### Feature Importance (by gain)

Top 3: **area** (dominant), **n_vertices**, **room_type_Bathroom**. Door position (rel_x, rel_y) is low — spatial layout matters more than tabular summary.

### Score Bucket Analysis (test)

| Bucket | n | MAE |
|--------|---|-----|
| Failed (=0) | 1349 | 13.11 |
| Low (1-39) | 139 | 24.09 |
| Mid (40-69) | 250 | 17.75 |
| Good (70-89) | 952 | 9.98 |
| Excellent (90+) | 1897 | 8.20 |

## Interpretation

**MAE < 15** — geometry clearly determines score. Strong baseline. Area + room type + shape complexity capture most of the scoring signal. However:
- **Kitchen** (MAE 16.89) and **Living room** (MAE 18.84) remain hard — their scoring depends more on spatial layout (door-furniture clearance, wall segment lengths) that tabular features cannot express.
- **Low (1-39)** bucket MAE is 24.09 — model struggles with the transition zone between failure and good scores.
- The CNN should improve on rooms where spatial detail matters most.

## Decisions Log

- **CV for n_estimators only**: 5-fold KFold CV (not StratifiedKFold — continuous target), early stopping patience=50. Found 457 optimal rounds. No grid search — baseline should be simple.
- **Fail/pass threshold at 5**: Since LightGBM predicts continuous values, used threshold=5 for binary classification (fail vs pass). Accounts for regression imprecision near zero.
- **Model saved as joblib**: `models/baseline_lgbm.joblib` (1.3 MB). Also uploaded as W&B artifact.
- **apartment_type added as feature** (2026-02-27): 7 apartment types one-hot encoded (21 total features, was 14). EDA showed large effect for Living room (eta-sq=0.19, +56 pt median delta) and Kitchen (eta-sq=0.11, +25 pt delta). Test MAE improved from 11.02 to 8.24 (−25%). Kitchen MAE: 16.89→11.14, Living room: 18.84→8.39. apt_type features ranked 7th-10th by importance. Best iteration rose from 457 to 1370 (more signal to learn). Model retrained with same hyperparameters except n_estimators cap raised to 2000.
