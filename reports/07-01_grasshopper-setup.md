# Grasshopper Integration — Setup & Verification Guide

Step-by-step instructions for deploying the furnisher surrogate model as a Grasshopper component in Rhino 8.

## Prerequisites

Before starting, confirm these are in place:

- Rhino 8 installed (with CPython scripting support)
- Internet access (to install packages and download model)
- The `furnisher_surrogate` Python package is published at `https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate`
- Trained model checkpoint (`.pt` file) available on W&B at `infau/furnisher-surrogate`

All inference code (`predict.py`, `rasterize.py`, `models.py`) and the GhPython component script (`surrogate_score.py`) are already written and tested via pytest (7 fixture rooms, 7 test cases, all passing).

---

## Step 1: Install Dependencies in Rhino 8

Open **Rhino 8 → Tools → Script Editor → Terminal** (the CPython terminal, not RhinoScript).

Run these two commands in order:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This installs PyTorch CPU-only (~200 MB). Must be done separately because it uses a special index URL.

```
pip install "git+https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate.git[inference]"
```

This installs the `furnisher_surrogate` package with the `[inference]` extra (numpy, Pillow). The quotes are needed because of the brackets.

**Verify the installation:**

```python
from furnisher_surrogate.predict import predict_score
print("OK")
```

If this prints `OK` without errors, the package is installed correctly.

---

## Step 2: Download the Model

The `.pt` checkpoint files are stored as W&B Artifacts (not in the git repo).

### Available models

| Model | Features | Test MAE | Recommended? |
|-------|----------|----------|-------------|
| `cnn_v4.pt` | geometry + room type + apartment type | 8.07 | **Yes** — best accuracy |
| `cnn_v1.pt` | geometry + room type only | 17.90 | For verification only (test room expected scores use v1) |

### Option A: Web UI

1. Go to [wandb.ai/infau/furnisher-surrogate](https://wandb.ai/infau/furnisher-surrogate)
2. Click **Artifacts** in the left sidebar
3. Download the desired model (e.g., `cnn_v4.pt` for production, `cnn_v1.pt` for verification)
4. Place it in a known location, e.g., `C:\Models\cnn_v4.pt`

### Option B: CLI (if wandb is installed)

```
wandb artifact get infau/furnisher-surrogate/cnn-v4:latest --root C:\Models
```

Remember the full path — you will need it in Step 4.

---

## Step 3: Create the GhPython Component

In Grasshopper:

1. Drag a **GhPython Script** component onto the canvas
2. Right-click the component → **Edit Source** (or double-click)
3. **Delete** the default script content
4. **Paste** the following script in its entirety:

```python
"""Surrogate score predictor — GhPython component for Rhino 8.

Inputs (set in Grasshopper component):
    polygon : Polyline  — room boundary (closed polyline)
    door    : Point3d   — door position on wall
    room_type : str     — one of: Bedroom, Living room, Bathroom, WC,
                          Kitchen, Children 1-4
    apartment_type : str — one of: Studio (bedroom), Studio (living),
                           1-Bedroom, 2-Bedroom, 3-Bedroom, 4-Bedroom,
                           5-Bedroom
    model_path : str    — (optional) path to .pt checkpoint

Output:
    score : float — predicted furniture placement score (0-100)
"""

import numpy as np
from furnisher_surrogate.predict import predict_score

# Convert Rhino Polyline → numpy (N, 2) array
# polygon is a Rhino Polyline; iterate its points
poly_np = np.array(
    [[pt.X, pt.Y] for pt in polygon],
    dtype=np.float64,
)

# Ensure closed polyline
if not np.allclose(poly_np[0], poly_np[-1]):
    poly_np = np.vstack([poly_np, poly_np[0:1]])

# Convert Rhino Point3d → numpy (2,) array
door_np = np.array([door.X, door.Y], dtype=np.float64)

# Predict
score = predict_score(
    poly_np,
    door_np,
    room_type,
    apartment_type=apartment_type if apartment_type else None,
    model_path=model_path if model_path else None,
)
```

5. **Set up the component inputs** (right-click each input → Rename / Type Hint):

| Input | Name | Type Hint | Description |
|-------|------|-----------|-------------|
| 1st | `polygon` | Polyline | Room boundary (closed polyline, in meters) |
| 2nd | `door` | Point3d | Door position (point on wall, in meters) |
| 3rd | `room_type` | str | Room type name (see list below) |
| 4th | `apartment_type` | str | Apartment type name (see list below, optional — defaults to "2-Bedroom") |
| 5th | `model_path` | str | Full path to `.pt` file (optional) |

6. **Set up the output:**

| Output | Name | Description |
|--------|------|-------------|
| 1st | `score` | Predicted score (float, 0–100) |

### Valid room types

```
Bedroom
Living room
Bathroom
WC
Kitchen
Children 1
Children 2
Children 3
Children 4
```

### Valid apartment types

```
Studio (bedroom)
Studio (living)
1-Bedroom
2-Bedroom
3-Bedroom
4-Bedroom
5-Bedroom
```

If `apartment_type` is left empty, the model defaults to "2-Bedroom" (the most common type).

---

## Step 4: Wire Up Test Rooms

Create the following test rooms in Grasshopper to verify the component works. For each room:
- Draw the **Polyline** with exact vertex coordinates (in meters)
- Place a **Point** at the door position
- Connect a **Value List** or **Panel** with the room type string
- Leave **apartment_type** empty (defaults to "2-Bedroom") — the expected scores below assume this default
- Connect the **model_path** (Panel with the full path to your `.pt` file)
- Read the **score** output

> **Note:** The expected scores below are for `cnn_v1.pt`, which does not use apartment type. If you use `cnn_v4.pt` or later, scores will differ because those models incorporate apartment type. To verify against the expected values, use `cnn_v1.pt`.

### Room 1: rect_high_bedroom

**Type:** Bedroom

**Polygon vertices** (closed rectangle, 3.52 x 4.46 m):
```
(0.0, 0.0)
(3.5193, 0.0)
(3.5193, 4.4561)
(0.0, 4.4561)
(0.0, 0.0)
```

**Door position:** `(0.0, 3.5475)`

**Expected score (cnn_v1):** 71.94

---

### Room 2: rect_zero_kitchen

**Type:** Kitchen

**Polygon vertices** (closed rectangle, 1.31 x 2.03 m):
```
(19.6169, 0.0)
(20.9247, 0.0)
(20.9247, 2.0264)
(19.6169, 2.0264)
(19.6169, 0.0)
```

**Door position:** `(20.5896, 0.0)`

**Expected score (cnn_v1):** 0.00

This room is deliberately too small for a kitchen — the model correctly predicts failure.

---

### Room 3: lshape_living

**Type:** Living room

**Polygon vertices** (L-shape, 6 unique vertices):
```
(6.162, 0.0)
(9.4097, 0.0)
(9.4097, 1.5189)
(11.2577, 1.5189)
(11.2577, 4.9371)
(6.162, 4.9371)
(6.162, 0.0)
```

**Door position:** `(11.2577, 1.9545)`

**Expected score (cnn_v1):** 58.76

---

### Room 4: complex_children

**Type:** Children 2

**Polygon vertices** (complex 8-vertex shape):
```
(29.9175, 0.0)
(33.7534, 0.0)
(33.7534, 0.6481)
(34.3236, 0.6481)
(34.3236, 1.6277)
(32.6282, 1.6277)
(32.6282, 2.8063)
(29.9175, 2.8063)
(29.9175, 0.0)
```

**Door position:** `(32.2947, 0.0)`

**Expected score (cnn_v1):** 32.85

---

### Room 5: small_bathroom

**Type:** Bathroom

**Polygon vertices** (closed rectangle, 2.46 x 1.85 m):
```
(14.1215, 0.0)
(16.5784, 0.0)
(16.5784, 1.8506)
(14.1215, 1.8506)
(14.1215, 0.0)
```

**Door position:** `(15.5249, 1.8506)`

**Expected score (cnn_v1):** 60.00

---

### Room 6: large_living_high

**Type:** Living room

**Polygon vertices** (L-shape, 6 unique vertices):
```
(7.7005, 0.0)
(10.9188, 0.0)
(10.9188, 5.2762)
(6.2591, 5.2762)
(6.2591, 0.8392)
(7.7005, 0.8392)
(7.7005, 0.0)
```

**Door position:** `(7.1404, 0.8392)`

**Expected score (cnn_v1):** 72.33

---

### Room 7: wc_mid

**Type:** WC

**Polygon vertices** (closed rectangle, 1.64 x 1.33 m):
```
(16.0242, 0.0)
(17.6667, 0.0)
(17.6667, 1.3265)
(16.0242, 1.3265)
(16.0242, 0.0)
```

**Door position:** `(17.0513, 0.0)`

**Expected score (cnn_v1):** 60.08

---

## Step 5: Verification Checklist

After wiring up all test rooms, compare each output score against the expected values.

| # | Room | Type | Expected | GH Output | Pass? |
|---|------|------|----------|-----------|-------|
| 1 | rect_high_bedroom | Bedroom | 71.94 | | |
| 2 | rect_zero_kitchen | Kitchen | 0.00 | | |
| 3 | lshape_living | Living room | 58.76 | | |
| 4 | complex_children | Children 2 | 32.85 | | |
| 5 | small_bathroom | Bathroom | 60.00 | | |
| 6 | large_living_high | Living room | 72.33 | | |
| 7 | wc_mid | WC | 60.08 | | |

**Tolerance:** Scores should match within **0.01** of the expected values. If any score differs by more than 0.01, something is wrong — see Troubleshooting below.

**All 7 must pass** before considering the Grasshopper integration verified.

Save the Grasshopper definition as `grasshopper/test_surrogate.gh` once all rooms are wired up and verified.

---

## Step 6: Compare with Actual Furnisher Scores

For reference, the actual procedural furnisher scores for these rooms are:

| Room | Surrogate (cnn_v1) | Actual Furnisher | Delta |
|------|---------------------|------------------|-------|
| rect_high_bedroom | 71.94 | 87.8 | -15.9 |
| rect_zero_kitchen | 0.00 | 0.0 | 0.0 |
| lshape_living | 58.76 | 50.0 | +8.8 |
| complex_children | 32.85 | 0.0 | +32.9 |
| small_bathroom | 60.00 | 100.0 | -40.0 |
| large_living_high | 72.33 | 100.0 | -27.7 |
| wc_mid | 60.08 | 75.0 | -14.9 |

The surrogate is an approximation — it will not match the procedural furnisher exactly. The purpose of this comparison is to confirm the model produces reasonable scores in the right ballpark, not bit-exact matches.

**Note:** These comparison scores are for `cnn_v1`. The production model (`cnn_v4`) uses apartment type context and will produce different (generally more accurate) scores, especially for Living room and Kitchen.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: furnisher_surrogate` | Package not installed in Rhino's Python | Re-run `pip install` commands from Step 1 in Rhino Script Editor Terminal |
| `ModuleNotFoundError: torch` | PyTorch not installed or wrong index | Re-run `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `FileNotFoundError: No model found` | `.pt` file not at expected path | Set the `model_path` input to the full path of your `.pt` file (e.g., `C:\Models\cnn_v4.pt`) |
| Score = 0 for all rooms | Coordinates in millimeters instead of meters | All coordinates must be in **meters**. If your Rhino model is in mm, divide by 1000. |
| Score differs from expected by > 0.01 | Wrong model file or corrupted download | The expected test scores in Step 4 are for `cnn_v1.pt`. If using `cnn_v4.pt`, scores will differ — that's expected. To verify bit-exact, use `cnn_v1.pt`. |
| Very slow first prediction (~5-10 sec) | Model loading on first call | Normal — the model is cached after the first call. Subsequent predictions are fast (~100-200 ms). |

---

## Model Swapping

To use a newer or different model:

1. Download the new `.pt` file from W&B (e.g., `cnn_v3.pt`, `cnn_v4.pt`)
2. Place it in your models folder
3. Update the `model_path` input on the GhPython component to point to the new file
4. Re-run — the component will load the new model automatically

No code changes are needed. Each `.pt` checkpoint contains all architecture parameters and normalization statistics, so the inference code reconstructs the correct model variant (v1–v4) on the fly. Older checkpoints (v1–v3) that lack apartment type support will simply ignore the `apartment_type` input.

**If `model_path` is left empty**, the code searches for the latest `cnn_*.pt` file in the `models/` directory relative to the installed package location. For most Grasshopper setups, it is simpler to always provide an explicit path.
