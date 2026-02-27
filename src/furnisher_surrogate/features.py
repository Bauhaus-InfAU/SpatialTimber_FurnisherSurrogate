"""Numeric feature extraction from Room geometry.

All functions take a Room and return numeric values.
Pure numpy — no torch dependency — so Grasshopper can vendor these later.
"""

from __future__ import annotations

import numpy as np

from .data import APT_TYPES, ROOM_TYPES, Room

# ── Per-room scalar features ─────────────────────────────────


def area(room: Room) -> float:
    """Polygon area via the shoelace formula (always positive)."""
    x = room.polygon[:, 0]
    y = room.polygon[:, 1]
    return 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def aspect_ratio(room: Room) -> float:
    """Bounding-box width / height, always >= 1.0."""
    mins = room.polygon.min(axis=0)
    maxs = room.polygon.max(axis=0)
    w, h = maxs[0] - mins[0], maxs[1] - mins[1]
    if h == 0 or w == 0:
        return 1.0
    ratio = w / h
    return ratio if ratio >= 1.0 else 1.0 / ratio


def n_vertices(room: Room) -> int:
    """Unique vertex count (polygon length - 1 for the closing repeat)."""
    return len(room.polygon) - 1


def door_rel_position(room: Room) -> tuple[float, float]:
    """Door position normalised to [0, 1] within the bounding box."""
    mins = room.polygon.min(axis=0)
    maxs = room.polygon.max(axis=0)
    extent = maxs - mins
    extent = np.where(extent == 0, 1.0, extent)
    rel = (room.door - mins) / extent
    return float(rel[0]), float(rel[1])


def room_type_onehot(room: Room) -> np.ndarray:
    """One-hot encoding of room type. Shape: (9,)."""
    vec = np.zeros(len(ROOM_TYPES), dtype=np.float32)
    vec[room.room_type_idx] = 1.0
    return vec


def apt_type_onehot(room: Room) -> np.ndarray:
    """One-hot encoding of apartment type. Shape: (7,)."""
    vec = np.zeros(len(APT_TYPES), dtype=np.float32)
    if room.apartment_type_idx is not None:
        vec[room.apartment_type_idx] = 1.0
    return vec


# ── Feature vector assembly ───────────────────────────────────

FEATURE_NAMES: list[str] = (
    ["area", "aspect_ratio", "n_vertices", "door_rel_x", "door_rel_y"]
    + [f"room_type_{rt}" for rt in ROOM_TYPES]
    + [f"apt_type_{at}" for at in APT_TYPES]
)


def extract_features(room: Room) -> np.ndarray:
    """Full feature vector for the baseline model. Shape: (21,).

    Layout: [area, aspect_ratio, n_vertices, door_rel_x, door_rel_y,
             room_type_Bedroom, ..., room_type_Children 4,
             apt_type_Studio (bedroom), ..., apt_type_5-Bedroom]

    5 numeric + 9 room_type one-hot + 7 apt_type one-hot = 21 features.
    """
    dx, dy = door_rel_position(room)
    return np.concatenate(
        [
            np.array(
                [area(room), aspect_ratio(room), n_vertices(room), dx, dy],
                dtype=np.float32,
            ),
            room_type_onehot(room),
            apt_type_onehot(room),
        ]
    )


def extract_feature_matrix(rooms: list[Room]) -> np.ndarray:
    """Feature matrix for a list of rooms. Shape: (N, 21)."""
    return np.stack([extract_features(r) for r in rooms])


def extract_scores(rooms: list[Room]) -> np.ndarray:
    """Score vector. Shape: (N,). Raises if any score is None."""
    scores: list[float] = []
    for r in rooms:
        if r.score is None:
            raise ValueError(
                f"Room '{r.room_type}' (apt seed={r.apartment_seed}) "
                f"has no score — is this an inference-time room?"
            )
        scores.append(r.score)
    return np.array(scores, dtype=np.float32)
