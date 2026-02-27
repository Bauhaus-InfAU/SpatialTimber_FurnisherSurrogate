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
- [x] Retrain CNN v4 with apartment_type embedding

## Architecture (`models.py`)

Shared CNN backbone across all versions:

```
Conv block 1: Conv2d(3→32, 3, pad=1) → BN → ReLU → MaxPool(2)    → 32x32x32
Conv block 2: Conv2d(32→64, 3, pad=1) → BN → ReLU → MaxPool(2)   → 64x16x16
Conv block 3: Conv2d(64→128, 3, pad=1) → BN → ReLU → MaxPool(2)  → 128x8x8
Conv block 4: Conv2d(128→256, 3, pad=1) → BN → ReLU → MaxPool(2) → 256x4x4
GlobalAvgPool → 256-dim image vector
```

### Version configs

Each version adds to the previous. The `config` dict is saved inside the `.pt` checkpoint so `predict.py` reconstructs the correct architecture automatically.

| Parameter | v1 | v2 | v3 | v4 |
|-----------|----|----|----|----|
| `image_bottleneck` | — | 64 | 64 | 64 |
| `tabular_hidden` | — | 32 | 32 | 32 |
| `tabular_skip` | no | no | **yes** | yes |
| `n_tabular` | 3 | 3 | **5** | 5 |
| `apt_embed_dim` | — | — | — | **4** |
| `n_apt_types` | — | — | — | **7** |
| `fc_hidden` | 128 | 64 | 64 | 128 |
| `dropout` | 0.3 | 0.3 | 0.3 | 0.3 |
| `embed_dim` (room type) | 16 | 16 | 16 | 16 |
| Tabular inputs | area, door_x, door_y | area, door_x, door_y | + n_vertices, aspect_ratio | + n_vertices, aspect_ratio |
| Checkpoint | `cnn_v1.pt` | (not saved) | `cnn_v3.pt` | `cnn_v4.pt` |

~400–425k parameters. Fits comfortably on RTX 4060 (8GB).

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
| v4 | +apt_type embedding (4-dim) | 8.07 | −2.95 | — |
| Baseline (14f) | LightGBM on 14 tabular features | 11.02 | — | `3t4hiefb` |
| Baseline (21f) | LightGBM on 21 features (+apt_type) | 8.24 | — | — |

**Conclusion:** apartment_type improved both models substantially (v3→v4: −28%, baseline: −25%). CNN v4 slightly beats LightGBM 21f (8.07 vs 8.24). However, the difference is small (0.17 MAE) — LightGBM remains the production model for simplicity (no PyTorch inference needed).

**Report:** [`reports/06-01_cnn-model-comparison.ipynb`](../reports/06-01_cnn-model-comparison.ipynb) | [HTML preview](https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/06-01_cnn-model-comparison.html)

## Decisions Log

- **Area added as numeric FC input** (decided during Phase 4 planning): Rasterization uses per-room normalization (longest side → 60px) to maximize shape detail, which discards absolute size. Since area is the strongest predictor (r=+0.37 from EDA), it's fed as a numeric scalar into the FC head alongside room_type and door position. This changes the concat from 274-dim to 275-dim.

- **Image bottleneck (v2)**: Compressed image features from 256→64 dims to prevent the noisy image branch from dominating the tabular signal. Combined with tabular FC (19→32), this rebalanced the branches and dropped MAE from 17.90 to 12.40.

- **n_vertices + aspect_ratio + tabular skip (v3)**: Added two geometry features the baseline already had, plus a skip connection letting tabular features bypass the image branch. These were the baseline's key features that v1/v2 lacked. Dropped MAE from 12.40 to 11.23.

- **Conclusion — spatial image features negligible**: Each improvement came from strengthening tabular input or weakening image input. The CNN never extracted useful spatial information that tabular features couldn't capture. LightGBM remains production model.

- **apartment_type embedding added (v4)** (2026-02-27): Added 4-dim embedding for 7 apartment types to the tabular branch. Test MAE improved from 11.23 (v3) to 8.07 (v4), now slightly beating LightGBM 21f (8.24). CNN v4 wins on 6/9 room types (Bedroom −1.26, Kitchen −1.27, Living room −0.31), loses on WC (+2.54) and Children 3 (+1.71). Practical difference is small — LightGBM remains production model for simplicity. Saved as `models/cnn_v4.pt`.
