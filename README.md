# FurnisherSurrogate

Surrogate model that predicts furniture placement quality scores for residential rooms, replacing a slow procedural scoring pipeline with a fast learned approximation.

## Motivation

The SpatialTimber project includes a procedural furnisher algorithm (implemented as a Grasshopper pipeline) that evaluates how well furniture can be placed in a given room. This score is essential for guiding generative architectural layout design via reinforcement learning.

The problem: the procedural furnisher is far too slow to call inside an RL training loop. A single apartment evaluation takes seconds, and RL requires millions of evaluations. A surrogate model that approximates the furnisher score enables practical RL-based design exploration.

## Training Data

Training data is stored in `apartments.jsonl` in the sibling repository (`SpatialTimber_FurnisherData`).

### Format

- **JSONL** — one JSON object per line, each representing a single apartment
- **~8,000 apartments**, each containing up to **9 rooms** in a fixed order
- ~80% standard-sized rooms, ~20% deliberately undersized to provide training diversity

### Per-Room Fields

| Field | Description |
|-------|-------------|
| `name` | Room type identifier |
| `active` | Whether the room is present in this apartment |
| `polygon` | Closed polyline outline in meters, axis-aligned, counter-clockwise winding |
| `door` | Door position (point on wall) |
| `score` | Furniture placement quality score (0–100), or `null` if room is absent |

### Room Types

Bedroom, Living room, Bathroom, WC, Kitchen, Children 1–4

### Apartment Types

Studio (combined living/bedroom), 1-Bedroom through 5-Bedroom

### Room Shapes

- **Rectangles** — most common
- **L-shapes** — single corner cut
- **Double-cuts** — U, S, or C shapes (two corner cuts)

## Score

The furnisher score quantifies how well furniture can be placed in a room on a 0–100 scale.

### How It Works

The procedural furnisher builds a tree of placement attempts. Each leaf node scores based on variant counts:
- 0 variants → 0.0
- 1 variant → 0.75
- 2+ variants → 1.0

Node scores aggregate upward through weighted averages, using option weights and level weights, to produce the final room score.

### Score Ranges

| Range | Meaning |
|-------|---------|
| 90–100 | Excellent — furniture fits comfortably |
| 70–89 | Good — acceptable placement |
| 40–69 | Problematic — tight or compromised |
| 1–39 | Poor — barely functional |
| 0 | Failed — no valid placement found |
| `null` | Room absent from apartment |

For the full scoring formula, see the [Furnisher Score documentation](https://www.notion.so/spatialtimber/Furnisher-Score-1d6b1023b22680a9a0c5c7ad80ac8df0).

## Surrogate Model Task

**Input:** room outline (polygon), door position, room type

**Output:** predicted score (0–100)

The model predicts per-room scores, not per-apartment. Each room is scored independently.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # creates .venv, installs all deps (incl. PyTorch CUDA)
uv run wandb login           # paste API key from https://wandb.ai/authorize
```

Verify GPU access:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data Location

Training data lives in the sibling repository:

```
../SpatialTimber_FurnisherData/apartments.jsonl
```
