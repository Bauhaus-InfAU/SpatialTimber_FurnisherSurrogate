# Phase 4: Rasterization Pipeline

[← Overview](../PLAN.md) | Depends on: [Data Pipeline](02-data-pipeline.md)

## Goal

Convert room polygons + door positions into 64x64 multi-channel images for CNN input.

**Notebook**: `notebooks/04-01_rasterization.ipynb`

## Tasks

- [x] `src/furnisher_surrogate/rasterize.py` — polygon → 64x64 3-channel image
- [x] Visual verification in notebook (compare rasterized vs original)
- [x] UMAP on rasterized images (compare to tabular UMAP from [EDA](03-eda.md))
- [x] Pre-rasterize full dataset to `.npz`

## Rasterization (`rasterize.py`)

**Normalization**: Per-room scaling — longest polygon side fits 60 pixels, **centered** in the 64x64 grid (equal padding on all sides), aspect ratio preserved, y-axis flipped to match image convention. Absolute room size is lost in the image; `area` is passed as a separate numeric input to the CNN's FC head (see [Phase 6 decisions](06-cnn-model.md#decisions-log)).

**Channels** (3 channels):

| Channel | Content | Type |
|---------|---------|------|
| 0 | **Room mask** — 255 inside polygon, 0 outside | uint8 binary |
| 1 | **Wall edges** — 255 on polygon boundary pixels (1px width) | uint8 binary |
| 2 | **Door marker** — gaussian blob (sigma=2px) at door position | uint8 0–255 |

**Rendering**: PIL `ImageDraw.polygon(fill=255)` for mask (handles concave L-shapes/notched rooms), `ImageDraw.line()` per segment for edges, numpy meshgrid gaussian for door blob.

## Door Position — Dual Encoding

Door position is encoded **twice**:
1. **In the image** — gaussian blob in channel 2 (spatial relationship to room shape)
2. **As numeric inputs** — `door_rel_x`, `door_rel_y` in [0,1] relative to bounding box, fed into the CNN's FC head

This lets the CNN learn from whichever signal is easier.

## Pre-rasterized Dataset (`data/rooms_rasterized.npz`)

All 45,880 rooms pre-rasterized via `uv run python -m furnisher_surrogate.rasterize`. Stored as a single compressed numpy archive (gitignored).

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `images` | `(45880, 3, 64, 64)` | `uint8` | Rasterized room images (mask, edges, door) |
| `scores` | `(45880,)` | `float32` | Ground-truth furnisher scores (0–100) |
| `room_type_idx` | `(45880,)` | `int8` | Room type index (0–8, maps to `ROOM_TYPES`) |
| `area` | `(45880,)` | `float32` | Room area in m² (from shoelace formula) |
| `door_rel_x` | `(45880,)` | `float32` | Door x position normalized to [0,1] within bbox |
| `door_rel_y` | `(45880,)` | `float32` | Door y position normalized to [0,1] within bbox |
| `apartment_seeds` | `(45880,)` | `int64` | Apartment ID for split assignment |

All arrays share the same row index — `images[i]`, `scores[i]`, etc. all refer to the same room. Uses `np.savez_compressed` (~8.7 MB on disk vs ~563 MB uncompressed).

**At training time** (Phase 6): `images[idx].float() / 255.0` normalizes to [0,1]. Split assignment uses `apartment_seeds` with `assign_splits()`.

## Decisions Log

- **Per-room normalization** (not global scale): Each room's longest side → 60px. Maximizes shape detail for all rooms. Absolute size lost in image but recovered via `area` as CNN numeric input. A global scale would shrink small rooms to ~3–4px (unreadable), defeating the purpose of rasterization.
- **Y-axis flipped** in coordinate transform so `imshow` default (origin='upper') matches visual expectation. CNN doesn't care about orientation (augmentation includes flips).
- **`savez_compressed`** instead of `savez`: binary channels compress extremely well — 8.7 MB vs ~563 MB uncompressed.
- **`area` added to `.npz`**: Needed because per-room normalization discards absolute size, and area is the strongest predictor (r=+0.37 from EDA). Fed to CNN FC head.
