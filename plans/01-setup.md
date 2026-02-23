# Phase 1: Setup

[← Overview](../PLAN.md)

## Goal

Set up the development environment, install dependencies, and create the project structure.

## Tasks

- [ ] Create `pyproject.toml` with all dependencies
- [ ] Install PyTorch with CUDA 12.x support
- [ ] Set up W&B account + `wandb login`
- [x] Create `src/furnisher_surrogate/` package structure with `__init__.py`
- [x] Create project directories (`notebooks/`, `grasshopper/`, `models/`, `data/`) + `.gitignore`
- [ ] Verify GPU is accessible from PyTorch

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "furnisher-surrogate"
requires-python = ">=3.12"
dependencies = [
    "torch",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "matplotlib",
    "jupyter",
    "wandb",
    "Pillow",
    "tqdm",
    "joblib",
    "umap-learn",
    "plotly",
    "minisom",
]
```

**PyTorch CUDA** (current install is CPU-only):
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## W&B Setup

1. Create account at https://wandb.ai (sign up with GitHub)
2. Get API key from https://wandb.ai/authorize
3. Run `wandb login` in terminal, paste API key (one-time)
4. Create project `furnisher-surrogate` on wandb.ai

**Integration pattern** (used in every training script):
```python
import wandb
run = wandb.init(project="furnisher-surrogate", name="run-name", config={...})
wandb.log({"metric": value})
wandb.finish()
```

**What to log**: train/val loss curves, prediction scatter plots, per-room-type metrics, feature importance, model artifacts.

## Decisions Log

- **2026-02-23**: Created folder structure early (before `pyproject.toml`) to establish project layout. Empty dirs use `.gitkeep`. Data artifacts (`.npz`, `.pt`, `.onnx`, `.joblib`) and `data/apartments.jsonl` are gitignored — only `data/data_manifest.json` will be tracked.
