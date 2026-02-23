# Phase 4: Rasterization Pipeline

[← Overview](../PLAN.md) | Depends on: [Data Pipeline](02-data-pipeline.md)

## Goal

Convert room polygons + door positions into 64x64 multi-channel images for CNN input.

## Tasks

- [ ] `src/furnisher_surrogate/rasterize.py` — polygon → 64x64 3-channel image
- [ ] Visual verification in notebook (compare rasterized vs original)
- [ ] UMAP on rasterized images (compare to tabular UMAP from [EDA](03-eda.md))
- [ ] Pre-rasterize full dataset to `.npz`

## Rasterization (`rasterize.py`)

**Normalization**: Scale longest side to fit 60 pixels, **center** the polygon in the 64x64 grid (equal padding on all sides), maintain aspect ratio. Centering ensures the room stays in the same image region after flip/rotation augmentations.

**Channels** (3 channels):

| Channel | Content | Type |
|---------|---------|------|
| 1 | **Room mask** — 1 inside polygon, 0 outside | binary |
| 2 | **Wall edges** — 1 on polygon boundary pixels | binary |
| 3 | **Door marker** — gaussian blob (sigma ~2px) at door position | float |

## Door Position — Dual Encoding

Door position is encoded **twice**:
1. **In the image** — gaussian blob in channel 3 (spatial relationship to room shape)
2. **As numeric inputs** — `door_rel_x`, `door_rel_y` in [0,1] relative to bounding box, fed into the CNN's FC head

This lets the CNN learn from whichever signal is easier.

## Pre-rasterization

Pre-rasterize all 45,880 rooms to a single `.npz` file:
- ~45k x 3 x 64 x 64 x uint8 ≈ 700 MB
- Load once into memory for training

## Decisions Log

*(Record decisions here as they're made)*
