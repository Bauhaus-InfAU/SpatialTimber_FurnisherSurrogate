# FurnisherSurrogate — ML Strategy

Surrogate model to predict furniture placement scores (0–100) for residential rooms. Two models: tabular baseline (LightGBM) and CNN on rasterized images.

**Data**: 8,322 apartments, 45,880 active rooms | Bimodal scores: 13k zeros + 19k excellent (90–100) | Mean 62.2, median 85.0
**System**: Python 3.12, uv, RTX 4060 8GB, Rhino 8 (CPython)
**Data location**: `../SpatialTimber_DesignExplorer/Furnisher/Apartment Quality Evaluation/apartments.jsonl`

---

## Progress

| # | Phase | Tasks | Status | Plan |
|---|-------|-------|--------|------|
| 1 | **Setup** | 6/6 | `done` | [details](plans/01-setup.md) |
| 2 | **Data Pipeline** | 5/5 | `done` | [details](plans/02-data-pipeline.md) |
| 3 | **EDA** | 13/13 | `done` | [details](plans/03-eda.md) |
| 4 | **Rasterization** | 4/4 | `done` | [details](plans/04-rasterization.md) |
| 5 | **Baseline Model** | 5/5 | `done` | [details](plans/05-baseline-model.md) |
| 6 | **CNN Model** | 6/6 | `done` | [details](plans/06-cnn-model.md) |
| 7 | **Grasshopper** | 9/9 | `done` | [details](plans/07-grasshopper.md) |
| | **Total** | **48/48** | | |

## Documentation Strategy

Each fact lives in **one place**. See [CLAUDE.md](CLAUDE.md) for full protocol. Use `/document` to trigger updates.

| File | Owns | Updated |
|------|------|---------|
| README.md | Project description, data format, setup | At milestones |
| CLAUDE.md | Current state, findings, conventions | Each session end |
| PLAN.md + plans/ | Strategy, decisions, progress | As work progresses |
| W&B | Experiment metrics, curves, artifacts | During training |
| Notebooks | Self-contained analyses | During analysis |

## Project Structure

```
SpatialTimber_FurnisherSurrogate/
├── PLAN.md                        # This overview
├── plans/                         # Detailed phase plans
├── src/furnisher_surrogate/       # Python package
│   ├── data.py                    # JSONL loading, splitting
│   ├── features.py                # Numeric feature extraction
│   ├── rasterize.py               # Polygon → 64x64 image
│   ├── models.py                  # CNN architecture
│   ├── train.py                   # Training loop
│   └── predict.py                 # Inference API
├── notebooks/                     # Jupyter notebooks (analysis, exploration)
├── reports/                       # Findings reports (narrative notebooks + HTML)
├── grasshopper/                   # GhPython components
├── models/                        # Saved artifacts (.pt, .joblib)
├── tests/                         # Pytest suite (fixtures, test_predict.py)
└── tickets/                       # Deferred features, bugs, improvements
```

## Known Limitations

1. **Axis-aligned rooms only** — current data is orthogonal. Real plans are often non-orthogonal. Future: new training data + vertex-level rotation augmentation. [details](plans/06-cnn-model.md#known-limitations)
2. **Fixed 9 room types, 7 apartment types** — adding types requires retraining
3. **Algorithm-specific** — approximates this furnisher's scoring, not objective quality

## Verification Checklist

- [x] Data loads correctly, all 45,880 active rooms parsed
- [x] Feature extraction produces valid values (no NaN, area > 0)
- [x] Rasterized images visually match original room shapes
- [x] Train/val/test split has no apartment leakage
- [x] W&B dashboard shows metrics and artifacts
- [x] Baseline MAE reported and reasonable (8.24 with apt_type, 11.02 without)
- [x] CNN MAE improves over baseline (or we understand why not) — v4 slightly beats LightGBM (8.07 vs 8.24)
- [x] predict_score() API matches manual pipeline (bit-exact, 7 fixture rooms verified)
- [x] apartment_type added across full stack (data, features, models, inference, GH component)
- [x] Grasshopper component returns predictions in Rhino 8
