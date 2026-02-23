# Phase 2: Data Pipeline

[← Overview](../PLAN.md) | Depends on: [Setup](01-setup.md)

## Goal

Build data loading, feature extraction, and train/val/test splitting infrastructure.

## Tasks

- [ ] `src/furnisher_surrogate/data.py` — JSONL loader, parse apartments and rooms
- [ ] Data integrity check — SHA-256 manifest for `apartments.jsonl`
- [ ] `src/furnisher_surrogate/features.py` — numeric feature extraction (area, aspect_ratio, n_vertices, room_type, door_rel_x, door_rel_y)
- [ ] Apartment-level train/val/test split (80/10/10) stratified by apartment_type
- [ ] Verify split has no apartment leakage + correct proportions

## Data Integrity (`data.py`)

The training data lives in a sibling repo and may change if the furnisher pipeline is re-run. To ensure reproducibility, `data.py` implements a hash-based integrity check.

**Manifest file**: `data/data_manifest.json` (git-tracked)
```json
{
  "source": "../SpatialTimber_DesignExplorer/Furnisher/Apartment Quality Evaluation/apartments.jsonl",
  "sha256": "a1b2c3...",
  "rows": 8322,
  "active_rooms": 45880,
  "snapshot_date": "2026-02-23"
}
```

**Behavior**:
- **First load** (no manifest exists): compute SHA-256, create manifest, proceed
- **Subsequent loads** (manifest exists): compare hash — if mismatch, **refuse to load** with message: *"Data has changed since last snapshot. Run `python -m furnisher_surrogate.data --update` to accept."*
- **`--update` flag**: recompute hash, update manifest, confirm change
- **W&B integration**: log `data_sha256` as a run config parameter for traceability

**What's NOT stored in git**: the JSONL file itself. Only the manifest (5 lines) is tracked.

## Data Loading (`data.py`)

- Load `apartments.jsonl` line by line (after integrity check passes)
- Parse each apartment into rooms, filter to `active == true`
- Return structured data: polygon points, door position, room type, score

## Feature Extraction (`features.py`)

Minimal numeric features for the baseline model:

| Feature | Extraction |
|---------|-----------|
| `area` | Shoelace formula on polygon vertices |
| `aspect_ratio` | bbox_width / bbox_height (always >= 1) |
| `n_vertices` | len(points) - 1 (4, 6, or 8) |
| `room_type` | One-hot encode 9 room types |

Total: 4 numeric + 9 one-hot = **13 features**

## Train/Val/Test Split

```
Total active rooms: 45,880
├── Train: 36,704 (80%)
├── Val:    4,588 (10%)
└── Test:   4,588 (10%)
```

**Critical**: Split at **apartment level** (not room level) to prevent data leakage — rooms from the same apartment are correlated. Stratify by `apartment_type`.

## Decisions Log

*(Record decisions here as they're made)*
