# Grasshopper Surrogate Score Component

Predicts furniture placement quality scores (0–100) for room geometries using a trained CNN model.

## Setup (Rhino 8)

### 1. Install dependencies

Open **Rhino 8 → Script Editor → Terminal** and run:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy Pillow
pip install git+https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate.git
```

### 2. Download model

Download the `.pt` checkpoint from W&B:

- Go to [wandb.ai/infau/furnisher-surrogate](https://wandb.ai/infau/furnisher-surrogate) → Artifacts
- Download the latest `cnn-*` model artifact
- Place the `.pt` file somewhere accessible (e.g. `C:\Models\cnn_v1.pt`)

Or via CLI (if wandb is installed):
```
wandb artifact get infau/furnisher-surrogate/cnn-v1:latest --root ./models
```

### 3. Set up Grasshopper component

1. Open `test_surrogate.gh` in Grasshopper
2. Add a **GhPython** component (or use the one in the test file)
3. Paste the contents of `surrogate_score.py` into the component
4. Set the component inputs:
   - `polygon` — Polyline (room boundary)
   - `door` — Point3d (door position on wall)
   - `room_type` — String (one of: Bedroom, Living room, Bathroom, WC, Kitchen, Children 1–4)
   - `model_path` — String (path to .pt file, optional if model is in default location)
5. Output `score` is a float (0–100)

## Model swapping

To use a newer model, simply change the `model_path` input to point to the new `.pt` file. No code changes needed — the checkpoint contains all architecture and normalization parameters.

## Test rooms

The `test_surrogate.gh` file includes predefined room geometries with known expected scores. After setup, verify the displayed scores match these expected values:

| Room | Type | Expected score (cnn_v1) |
|------|------|------------------------|
| See `tests/fixtures/test_rooms.json` for exact values | | |

## Supported room types

Bedroom, Living room, Bathroom, WC, Kitchen, Children 1, Children 2, Children 3, Children 4

## Troubleshooting

- **ModuleNotFoundError: furnisher_surrogate** — package not installed in Rhino's Python. Re-run the pip install commands above.
- **FileNotFoundError: No model found** — set the `model_path` input to the full path of your `.pt` file.
- **Score = 0 for all rooms** — check that the polygon and door coordinates are in meters (not millimeters).
