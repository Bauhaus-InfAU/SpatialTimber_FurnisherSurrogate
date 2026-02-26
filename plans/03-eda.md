# Phase 3: Exploratory Data Analysis

[← Overview](../PLAN.md) | Depends on: [Data Pipeline](02-data-pipeline.md)

## Goal

Understand the training data before modeling. Establish reference metrics. Identify potential issues.

**Notebooks**: `notebooks/01_data_exploration.ipynb` (main EDA), `notebooks/02_umap_exploration.ipynb` (UMAP)

## Tasks

- [x] Sanity checks (record counts, missing values, score range, polygon validity, door on wall, vertex counts)
- [x] Score distribution analysis (overall + per-room-type + per-apartment-type + per-shape)
- [x] Geometry distribution analysis (area, aspect ratio, bbox)
- [x] Correlation matrix (numeric features vs score)
- [x] Visual room gallery (3x3 per room type, color by score)
- [x] Score=0 failure analysis (visualize failed rooms, compare distributions)
- [x] Door position analysis (normalized positions, correlation with score)
- [x] UMAP — all 14 features (interactive plotly, expect room-type-dominated clusters)
- [x] UMAP — 5 numeric features only (color by score + color by room type)
- [x] UMAP — per room type on 5 numeric features (isolate geometry→score per type)
- [x] Establish reference metrics (naive MAE, feature-score correlations)
- [x] **Checkpoint**: Review findings before proceeding — may inform feature choices

**Outcome:** EDA findings report at `reports/eda-findings.ipynb` ([HTML preview](https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/eda-findings.html)). Key findings: bimodal score (28.6% fail), area strongest predictor (r=+0.37), door position has zero signal, naive MAE=37.48, inter-room correlation near zero (r=0.006).

## Sanity Checks

| Check | What to verify |
|-------|---------------|
| Record counts | 8,322 apartments, 45,880 active rooms |
| Missing values | No unexpected nulls in active rooms |
| Score range | All scores in [0, 100] |
| Polygon validity | Closed (first == last point), z always 0 |
| Door on wall | Every door lies on a polygon edge (within tolerance) |
| Vertex counts | Only 4, 6, or 8 unique vertices |

## Statistical Distributions

**Score** (critical — bimodal target):
- Overall histogram: 0-spike (13k) + 90-100 cluster (19k)
- Per-room-type histograms (3x3 grid)
- Score by apartment type
- Score by shape complexity (4/6/8 vertices)

**Geometry**:
- Area distribution per room type
- Aspect ratio distribution
- Bounding box width vs height

**Correlations**: area, aspect_ratio, n_vertices vs score. Pairplot colored by score bucket.

## Visual Room Inspection

**Room gallery** (matplotlib):
- 3x3 grid per room type with polygon outlines + door position
- Color by score (green → red)
- One high, one mid, one failed per type
- Room dimensions as annotation

**Score=0 analysis** — 13k failed rooms:
- Visualize sample of failed rooms
- Compare area distributions: score=0 vs score>0
- Room type breakdown of failures

**Door position**: normalized door positions, correlation with score per wall.

## UMAP

**Library**: `umap-learn` + `plotly`

**Pass 1 — tabular features** (area, aspect_ratio, n_vertices, room_type one-hot):
- 13 features → 2D embedding
- Interactive plotly scatter: color by score, hover shows details
- Key question: visible score gradient?

**Pass 2 — rasterized images** (after [Phase 4](04-rasterization.md)):
- 64x64x3 flattened → 2D embedding
- Compare to tabular UMAP: tighter clusters = CNN has more to learn

## Optional: Self-Organizing Map (SOM)

Start with UMAP. Only add SOM (via `minisom`) if grid-based topology view is needed.

## Key Metrics to Establish

| Metric | Why it matters |
|--------|---------------|
| Score mean/median/std per room type | Naive MAE baseline |
| % of score=0 per type | Classification vs regression decision |
| Area threshold for failure | Below X m², rooms always fail |
| Feature-score correlations | Signal strength for baseline |
| Inter-room correlation | Validates apartment-level splitting |

## Decisions Log

- **UMAP split into separate notebook** (`02_umap_exploration.ipynb`) to keep main EDA notebook fast and lightweight. Three variants: all 14 features, 5 numeric only, per room type.
- **EDA findings report** created as narrative notebook (`reports/eda-findings.ipynb`) with HTML export for Notion embedding. Source of truth for EDA conclusions.
