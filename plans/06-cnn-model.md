# Phase 6: CNN Model

[← Overview](../PLAN.md) | Depends on: [Rasterization](04-rasterization.md), [Baseline](05-baseline-model.md)

## Goal

Train a CNN on rasterized room images. Compare against baseline.

**Notebook**: `notebooks/06-01_cnn_training.ipynb`

## Tasks

- [x] `src/furnisher_surrogate/models.py` — CNN architecture
- [x] `src/furnisher_surrogate/train.py` — training loop with W&B
- [x] Train CNN v1 with default hyperparameters
- [x] Compare CNN vs baseline on same test set (MAE, RMSE, R², per-room-type MAE, scatter plots)
- [x] Diagnostic tuning (v2: balanced branches, v3: +geometry+skip)

## Architecture (`models.py`)

```
Input: 3x64x64 image + room_type (int) + area (float) + door_rel_x (float) + door_rel_y (float)

Conv block 1: Conv2d(3→32, 3, pad=1) → BN → ReLU → MaxPool(2)    → 32x32x32
Conv block 2: Conv2d(32→64, 3, pad=1) → BN → ReLU → MaxPool(2)   → 64x16x16
Conv block 3: Conv2d(64→128, 3, pad=1) → BN → ReLU → MaxPool(2)  → 128x8x8
Conv block 4: Conv2d(128→256, 3, pad=1) → BN → ReLU → MaxPool(2) → 256x4x4

GlobalAvgPool → 256-dim vector
Concat with room_type_embedding(9→16) + [area, door_rel_x, door_rel_y] → 275-dim
FC(275→128) → ReLU → Dropout(0.3)
FC(128→1) → output (predicted score)
```

~500k parameters. Fits comfortably on RTX 4060 (8GB).

## Training (`train.py`)

| Setting | Value |
|---------|-------|
| Loss | MSE |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR, 50 epochs |
| Batch size | 128 |
| Early stopping | Patience 10 epochs on val loss |

## Data Augmentation

Axis-aligned rooms: flips and 90° rotations produce valid rooms with identical scores.

| Transform | How |
|-----------|-----|
| Horizontal flip | `torch.flip(img, dims=[-1])`, p=0.5 |
| Vertical flip | `torch.flip(img, dims=[-2])`, p=0.5 |
| 90° rotation | `torch.rot90(img, k, dims=[-2,-1])`, k random 0-3 |

Applied on-the-fly in `Dataset.__getitem__`. ~8 unique variants per sample. **No augmentation on val/test.**

Tabular baseline doesn't need augmentation (features are invariant to these transforms).

## Hyperparameter Tuning

Diagnostic approach — read W&B curves and adjust:

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Learning rate** | 1e-3 | 1e-4 – 5e-3 | Highest |
| **Weight decay** | 1e-4 | 1e-5 – 1e-3 | High |
| **Dropout** | 0.3 | 0.1–0.5 | Medium |
| Batch size | 128 | 64–256 | Medium |
| Door gaussian sigma | 2px | 1–4px | Low |

**Strategy**:
1. Train v1 with defaults
2. Val loss plateaus high → lower learning rate
3. Overfitting → increase dropout or weight decay
4. Underfitting → increase channel widths or add conv block

## Evaluation

Same metrics as baseline (MAE, RMSE, R², per-room-type MAE) on same test set for fair comparison. Plus score-bucket F1 and prediction scatter plots.

## Known Limitations

- **Axis-aligned only**: Current data is orthogonal. Real plans often aren't. Future needs: new training data + vertex-level rotation + possibly higher resolution.
- **Fixed 9 room types**: New types require retraining.
- **Algorithm-specific**: Approximates this furnisher, not objective quality.

## Results

| Version | Architecture | Test MAE | vs Baseline | W&B |
|---------|-------------|----------|-------------|-----|
| v1 | Raw concat (256+16+3=275), FC(275→128→1) | 17.90 | +6.88 | `3wcevehy` |
| v2 | Image bottleneck 256→64, tabular FC 19→32 | 12.40 | +1.38 | `qutd7leh` |
| v3 | +n_vertices, +aspect_ratio, tabular skip | 11.23 | +0.21 | `ld6iz2h4` |
| Baseline | LightGBM on 14 tabular features | 11.02 | — | `3t4hiefb` |

**Conclusion:** CNN at best matches LightGBM. Spatial image features provide negligible improvement over tabular features. LightGBM remains the production model for Phase 7.

**Report:** [`reports/06-01_cnn-model-comparison.ipynb`](../reports/06-01_cnn-model-comparison.ipynb) | [HTML preview](https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/06-01_cnn-model-comparison.html)

## Decisions Log

- **Area added as numeric FC input** (decided during Phase 4 planning): Rasterization uses per-room normalization (longest side → 60px) to maximize shape detail, which discards absolute size. Since area is the strongest predictor (r=+0.37 from EDA), it's fed as a numeric scalar into the FC head alongside room_type and door position. This changes the concat from 274-dim to 275-dim.

- **Image bottleneck (v2)**: Compressed image features from 256→64 dims to prevent the noisy image branch from dominating the tabular signal. Combined with tabular FC (19→32), this rebalanced the branches and dropped MAE from 17.90 to 12.40.

- **n_vertices + aspect_ratio + tabular skip (v3)**: Added two geometry features the baseline already had, plus a skip connection letting tabular features bypass the image branch. These were the baseline's key features that v1/v2 lacked. Dropped MAE from 12.40 to 11.23.

- **Conclusion — spatial image features negligible**: Each improvement came from strengthening tabular input or weakening image input. The CNN never extracted useful spatial information that tabular features couldn't capture. LightGBM remains production model.
