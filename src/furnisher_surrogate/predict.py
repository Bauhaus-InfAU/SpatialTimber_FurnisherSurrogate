"""Inference API for furniture placement score prediction.

Single entry point for all consumers (Grasshopper, scripts, tests).
Depends only on numpy, Pillow, and torch — no sklearn or training deps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .models import RoomCNN
from .rasterize import rasterize_arrays

# ── Constants (duplicated from data.py to avoid sklearn import chain) ──

ROOM_TYPES: list[str] = [
    "Bedroom",
    "Living room",
    "Bathroom",
    "WC",
    "Kitchen",
    "Children 1",
    "Children 2",
    "Children 3",
    "Children 4",
]

ROOM_TYPE_TO_IDX: dict[str, int] = {rt: i for i, rt in enumerate(ROOM_TYPES)}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Geometry helpers (inlined from features.py to avoid data.py) ──────


def _area(polygon: np.ndarray) -> float:
    """Polygon area via the shoelace formula (always positive)."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def _aspect_ratio(polygon: np.ndarray) -> float:
    """Bounding-box width / height, always >= 1.0."""
    mins = polygon.min(axis=0)
    maxs = polygon.max(axis=0)
    w, h = maxs[0] - mins[0], maxs[1] - mins[1]
    if h == 0 or w == 0:
        return 1.0
    ratio = w / h
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _n_vertices(polygon: np.ndarray) -> int:
    """Unique vertex count (polygon length - 1 for the closing repeat)."""
    return len(polygon) - 1


def _door_rel_position(polygon: np.ndarray, door: np.ndarray) -> tuple[float, float]:
    """Door position normalised to [0, 1] within the bounding box."""
    mins = polygon.min(axis=0)
    maxs = polygon.max(axis=0)
    extent = maxs - mins
    extent = np.where(extent == 0, 1.0, extent)
    rel = (door - mins) / extent
    return float(rel[0]), float(rel[1])


# ── Model cache ──────────────────────────────────────────────────────

_model_cache: dict[str, tuple[RoomCNN, dict]] = {}


def _resolve_model_path(model_path: str | Path | None) -> Path:
    """Resolve model path from argument, env var, or default location."""
    if model_path is not None:
        return Path(model_path)

    env_path = os.environ.get("FURNISHER_MODEL_PATH")
    if env_path:
        return Path(env_path)

    # Default: look for any .pt file in models/
    models_dir = _PROJECT_ROOT / "models"
    if models_dir.is_dir():
        pt_files = sorted(models_dir.glob("cnn_*.pt"))
        if pt_files:
            return pt_files[-1]  # latest by name (v1, v2, ...)

    raise FileNotFoundError(
        "No model found. Either:\n"
        "  1. Pass model_path= to predict_score()\n"
        "  2. Set FURNISHER_MODEL_PATH env var\n"
        "  3. Place a .pt checkpoint in models/\n"
        "  4. Download from W&B: wandb artifact get infau/furnisher-surrogate/cnn-v1:latest"
    )


def _load_model(model_path: str | Path) -> tuple[RoomCNN, dict]:
    """Load model from checkpoint, caching by path."""
    model_path = Path(model_path)
    key = str(model_path.resolve())
    if key not in _model_cache:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {})

        model = RoomCNN(
            n_room_types=cfg.get("n_room_types", 9),
            embed_dim=cfg.get("embed_dim", 16),
            n_tabular=cfg.get("n_tabular", 3),
            channels=tuple(cfg.get("channels", (32, 64, 128, 256))),
            fc_hidden=cfg.get("fc_hidden", 128),
            dropout=cfg.get("dropout", 0.3),
            image_bottleneck=cfg.get("image_bottleneck"),
            tabular_hidden=cfg.get("tabular_hidden"),
            tabular_skip=cfg.get("tabular_skip", False),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _model_cache[key] = (model, ckpt)

    return _model_cache[key]


# ── Public API ───────────────────────────────────────────────────────


def predict_score(
    polygon: np.ndarray,
    door: np.ndarray,
    room_type: str,
    model_path: str | Path | None = None,
) -> float:
    """Predict furniture placement score for a single room.

    Parameters
    ----------
    polygon : (N, 2) float64
        Closed polyline in meters (first vertex == last vertex).
        If not closed, it will be auto-closed.
    door : (2,) float64
        Door position as a point on the room's wall, in meters.
    room_type : str
        One of: Bedroom, Living room, Bathroom, WC, Kitchen,
        Children 1, Children 2, Children 3, Children 4.
    model_path : str or Path, optional
        Path to a .pt checkpoint. Defaults to latest model in models/.

    Returns
    -------
    float
        Predicted score clamped to [0, 100].

    Raises
    ------
    ValueError
        If room_type is not one of the 9 known types.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    door = np.asarray(door, dtype=np.float64)

    # Auto-close polygon if needed
    if not np.allclose(polygon[0], polygon[-1]):
        polygon = np.vstack([polygon, polygon[0:1]])

    # Validate room type
    if room_type not in ROOM_TYPE_TO_IDX:
        raise ValueError(
            f"Unknown room_type '{room_type}'. Must be one of: {ROOM_TYPES}"
        )
    room_type_idx = ROOM_TYPE_TO_IDX[room_type]

    # Load model
    path = _resolve_model_path(model_path)
    model, ckpt = _load_model(path)
    cfg = ckpt.get("config", {})
    n_tabular = cfg.get("n_tabular", 3)

    # Rasterize
    image = rasterize_arrays(polygon, door)  # (3, 64, 64) uint8
    image_t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

    # Tabular features
    area_mean = ckpt.get("area_mean", cfg.get("area_mean", 0.0))
    area_std = ckpt.get("area_std", cfg.get("area_std", 1.0))

    area_raw = _area(polygon)
    area_norm = (area_raw - area_mean) / (area_std + 1e-8)

    door_rx, door_ry = _door_rel_position(polygon, door)

    if n_tabular == 3:
        tabular = [area_norm, door_rx, door_ry]
    elif n_tabular == 5:
        # v3-style: area, door_rel_x, door_rel_y, aspect_ratio, n_vertices
        ar = _aspect_ratio(polygon)
        nv = float(_n_vertices(polygon))
        # Standardize extra features if stats available
        ar_mean = ckpt.get("aspect_mean", cfg.get("aspect_mean", 0.0))
        ar_std = ckpt.get("aspect_std", cfg.get("aspect_std", 1.0))
        nv_mean = ckpt.get("n_verts_mean", cfg.get("n_verts_mean", 0.0))
        nv_std = ckpt.get("n_verts_std", cfg.get("n_verts_std", 1.0))
        ar_norm = (ar - ar_mean) / (ar_std + 1e-8)
        nv_norm = (nv - nv_mean) / (nv_std + 1e-8)
        tabular = [area_norm, door_rx, door_ry, ar_norm, nv_norm]
    else:
        raise ValueError(f"Unsupported n_tabular={n_tabular} in checkpoint config")

    tabular_t = torch.tensor([tabular], dtype=torch.float32)
    room_type_t = torch.tensor([room_type_idx], dtype=torch.long)

    # Inference
    with torch.no_grad():
        score = model(image_t, room_type_t, tabular_t).squeeze().item()

    return float(np.clip(score, 0.0, 100.0))
