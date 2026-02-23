# Phase 7: Grasshopper Integration

[← Overview](../PLAN.md) | Depends on: [Baseline](05-baseline-model.md) or [CNN](06-cnn-model.md)

## Goal

Deploy the best model as a Grasshopper component for interactive use in Rhino 8. Compare surrogate predictions against the procedural furnisher in real-time.

## Tasks

- [ ] ONNX export for best model
- [ ] `grasshopper/surrogate_predictor.py` — GhPython component
- [ ] Install dependencies in Rhino 8's CPython
- [ ] Test against procedural furnisher

## Approach: ONNX Runtime

Export to ONNX, load with `onnxruntime` in Grasshopper. Preferred over PyTorch because:
- `onnxruntime` is lightweight (~50MB vs ~2GB for torch)
- CPU inference is fast enough (single room = microseconds)
- No CUDA dependency in Rhino

## ONNX Export

```python
# LightGBM → use native format or sklearn2onnx
# CNN → torch.onnx.export
torch.onnx.export(model, (dummy_image, dummy_type, dummy_door_x, dummy_door_y),
                  "models/cnn_surrogate.onnx")
```

## GhPython Component (`surrogate_predictor.py`)

Inputs:
1. Room polygon (Rhino Polyline)
2. Door point (Rhino Point3d)
3. Room type (string)

Processing:
1. Convert polygon to feature vector (tabular) or rasterized image (CNN)
2. Run ONNX inference

Output:
- Predicted score (float, 0–100)

## Rhino 8 Setup

Install into Rhino 8's CPython environment:
```
pip install onnxruntime numpy
```
(via Rhino's Script Editor terminal)

## Decisions Log

*(Record decisions here as they're made)*
