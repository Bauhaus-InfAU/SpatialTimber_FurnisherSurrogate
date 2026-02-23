# Phase 6: CNN Model

[← Overview](../PLAN.md) | Depends on: [Rasterization](04-rasterization.md), [Baseline](05-baseline-model.md)

## Goal

Train a CNN on rasterized room images. Compare against baseline.

**Notebook**: `notebooks/03_raster_pipeline.ipynb`

## Tasks

- [ ] `src/furnisher_surrogate/models.py` — CNN architecture
- [ ] `src/furnisher_surrogate/train.py` — training loop with W&B
- [ ] Train CNN v1 with default hyperparameters
- [ ] Compare CNN vs baseline on same test set (MAE, RMSE, R², per-room-type MAE, scatter plots)
- [ ] Diagnostic tuning if needed (adjust based on W&B curves)

## Architecture (`models.py`)

```
Input: 3x64x64 image + room_type (int) + door_rel_x (float) + door_rel_y (float)

Conv block 1: Conv2d(3→32, 3, pad=1) → BN → ReLU → MaxPool(2)    → 32x32x32
Conv block 2: Conv2d(32→64, 3, pad=1) → BN → ReLU → MaxPool(2)   → 64x16x16
Conv block 3: Conv2d(64→128, 3, pad=1) → BN → ReLU → MaxPool(2)  → 128x8x8
Conv block 4: Conv2d(128→256, 3, pad=1) → BN → ReLU → MaxPool(2) → 256x4x4

GlobalAvgPool → 256-dim vector
Concat with room_type_embedding(9→16) + [door_rel_x, door_rel_y] → 274-dim
FC(274→128) → ReLU → Dropout(0.3)
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

## Decisions Log

*(Record decisions here as they're made)*
