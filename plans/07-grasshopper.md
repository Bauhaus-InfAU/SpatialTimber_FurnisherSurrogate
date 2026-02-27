# Phase 7: Grasshopper Integration

[← Overview](../PLAN.md) | Depends on: [Baseline](05-baseline-model.md) or [CNN](06-cnn-model.md)

## Goal

Deploy the best model as a Grasshopper component for interactive use in Rhino 8. Compare surrogate predictions against the procedural furnisher in real-time.

## Tasks

- [x] Decouple `rasterize.py` from `data.py` (TYPE_CHECKING guard + `rasterize_arrays()`)
- [x] `src/furnisher_surrogate/predict.py` — inference API (`predict_score()`)
- [x] Update packaging (`pyproject.toml`: `[inference]` extra, Python >=3.10)
- [x] `grasshopper/surrogate_score.py` — GhPython component
- [x] `grasshopper/README.md` — Rhino 8 setup instructions
- [x] Test fixtures (`tests/fixtures/test_rooms.json`) + pytest suite (`tests/test_predict.py`)
- [x] Add apartment_type input to predict_score() and GH component
- [ ] `grasshopper/test_surrogate.gh` — predefined test rooms in Grasshopper
- [ ] End-to-end test in Rhino 8 (install, load model, run component, compare scores)

## Approach: PyTorch CPU

Use PyTorch (CPU-only, ~200MB) directly instead of ONNX export. Rationale:
- Same `.pt` checkpoints from training — no export step
- Model swap = replace one file, no re-export
- Handles all architecture variants (v1–v4) via checkpoint `config` dict
- ONNX export is non-trivial for variable architecture (bottleneck, skip, n_tabular)

## Inference API (`predict.py`)

Single entry point for all consumers (Grasshopper, scripts, tests):

```python
from furnisher_surrogate.predict import predict_score

score = predict_score(
    polygon=np.array([[0,0], [4,0], [4,3], [0,3], [0,0]]),
    door=np.array([2.0, 0.0]),
    room_type="Bedroom",
    model_path="models/cnn_v1.pt",  # optional
)
```

**Internals:** Rasterizes polygon → 3x64x64 image, computes tabular features (area, door_rel), standardizes using checkpoint stats, runs inference, clamps to [0,100].

**Model resolution:** explicit `model_path` → `FURNISHER_MODEL_PATH` env var → latest `models/cnn_*.pt` → error with download instructions.

**Dependency chain:** `predict.py` imports only from `rasterize.py` (decoupled from `data.py` via TYPE_CHECKING) and `models.py` (torch only). No sklearn needed.

## GhPython Component (`grasshopper/surrogate_score.py`)

~6 lines: converts Rhino Polyline/Point3d → numpy arrays, calls `predict_score()`.

## Packaging

```toml
[project.optional-dependencies]
inference = ["numpy", "Pillow", "torch>=2.0.0"]
```

Rhino 8 install:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate.git[inference]
```

## Test Suite

7 fixture rooms from the test set covering diverse geometries (rect/L-shape/complex, high/zero/mid scores, 5 room types). Pytest verifies bit-exact match with expected scores.

## Decisions Log

- **PyTorch over ONNX** (2026-02-26): ONNX export is fragile for this architecture (3 input tensors, optional bottleneck/skip branches, variable n_tabular). PyTorch CPU wheel is ~200MB (not 2GB as initially estimated). Model swap is trivial with PyTorch — just drop in a new `.pt` file.
- **Python >=3.10** (2026-02-26): Lowered from >=3.12. Needed for `slots=True` in `data.py` dataclasses (3.10 feature). Inference code works on 3.9+ but full package needs 3.10.
- **Model distribution via W&B Artifacts** (2026-02-26): `.pt` files are gitignored. Stored as W&B artifacts, downloadable via web UI or `wandb artifact get`.
- **Decoupled rasterize.py** (2026-02-26): Moved `Room` import behind `TYPE_CHECKING` guard. Added `rasterize_arrays(polygon, door)` that accepts raw numpy arrays. Existing `rasterize_room(room)` delegates to it. This breaks the sklearn import chain for inference consumers.
