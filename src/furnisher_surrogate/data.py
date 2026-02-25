"""Data loading, integrity checking, and train/val/test splitting.

This module is the single entry point for reading apartments.jsonl.
All downstream code (EDA, features, rasterization, training) should
import from here rather than parsing the JSONL directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ── Constants ─────────────────────────────────────────────────

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

ROOM_TYPE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ROOM_TYPES)}

Split = Literal["train", "val", "test"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_PATH = (
    _PROJECT_ROOT
    / ".."
    / "SpatialTimber_DesignExplorer"
    / "Furnisher"
    / "Apartment Quality Evaluation"
    / "apartments.jsonl"
).resolve()
_MANIFEST_PATH = _PROJECT_ROOT / "data" / "data_manifest.json"


# ── Domain types ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Room:
    """A single active room — the fundamental prediction unit.

    Constructed from JSONL loading (bulk) or from Rhino geometry
    (Grasshopper inference, single room at a time).
    """

    polygon: np.ndarray  # (N, 2) float64, closed polyline in meters
    door: np.ndarray  # (2,) float64, point on wall in meters
    room_type: str  # one of ROOM_TYPES
    room_type_idx: int  # index into ROOM_TYPES
    score: float | None  # 0–100, or None at inference time

    # Provenance — set when loaded from JSONL, None at inference
    apartment_seed: int | None = None
    apartment_type: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Room):
            return NotImplemented
        return (
            np.array_equal(self.polygon, other.polygon)
            and np.array_equal(self.door, other.door)
            and self.room_type == other.room_type
            and self.score == other.score
            and self.apartment_seed == other.apartment_seed
        )

    def __hash__(self) -> int:
        return hash((self.room_type, self.score, self.apartment_seed))


@dataclass(frozen=True, slots=True)
class Apartment:
    """One apartment's active rooms. Used for apartment-level splitting."""

    seed: int
    apt_type: str
    rooms: tuple[Room, ...]


# ── Internal parsing ──────────────────────────────────────────


def _parse_room(
    raw: dict, seed: int, apt_type: str
) -> Room | None:
    """Parse one room dict from JSONL. Returns None if inactive."""
    if not raw.get("active", False):
        return None

    pts_3d = raw["points"]
    polygon = np.array([[p[0], p[1]] for p in pts_3d], dtype=np.float64)
    door_3d = raw["door"]
    door = np.array([door_3d[0], door_3d[1]], dtype=np.float64)
    name = raw["name"]

    return Room(
        polygon=polygon,
        door=door,
        room_type=name,
        room_type_idx=ROOM_TYPE_TO_IDX[name],
        score=raw["score"],
        apartment_seed=seed,
        apartment_type=apt_type,
    )


# ── Integrity ─────────────────────────────────────────────────


def _compute_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(
    data_path: Path = _DEFAULT_DATA_PATH,
    manifest_path: Path = _MANIFEST_PATH,
    *,
    update: bool = False,
) -> str:
    """Verify or create the data manifest. Returns the SHA-256 hash.

    Raises RuntimeError if hash mismatch and *update* is False.
    """
    current_hash = _compute_sha256(data_path)

    if manifest_path.exists() and not update:
        manifest = json.loads(manifest_path.read_text())
        if manifest["sha256"] != current_hash:
            raise RuntimeError(
                f"Data has changed since last snapshot.\n"
                f"  Expected: {manifest['sha256']}\n"
                f"  Got:      {current_hash}\n"
                f"Run: python -m furnisher_surrogate.data --update"
            )

    return current_hash


# ── Public loading API ────────────────────────────────────────


def load_apartments(
    data_path: Path | None = None,
    *,
    update_manifest: bool = False,
) -> list[Apartment]:
    """Load all apartments from JSONL. Verifies data integrity.

    This is the only function that reads the JSONL file.
    """
    path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
    sha = check_integrity(path, update=update_manifest)

    apartments: list[Apartment] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            seed = raw["seed"]
            apt_type = raw["apt_type"]
            rooms: list[Room] = []
            for raw_room in raw["rooms"]:
                room = _parse_room(raw_room, seed, apt_type)
                if room is not None:
                    rooms.append(room)
            apartments.append(
                Apartment(seed=seed, apt_type=apt_type, rooms=tuple(rooms))
            )

    # Write / update manifest with counts
    active_rooms = sum(len(a.rooms) for a in apartments)
    manifest = {
        "source": str(path),
        "sha256": sha,
        "rows": len(apartments),
        "active_rooms": active_rooms,
        "snapshot_date": __import__("datetime").date.today().isoformat(),
    }
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    return apartments


def load_rooms(data_path: Path | None = None) -> list[Room]:
    """Flat list of all active rooms. Delegates to load_apartments."""
    return [room for apt in load_apartments(data_path) for room in apt.rooms]


# ── Splitting ─────────────────────────────────────────────────


def assign_splits(
    apartments: list[Apartment],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[int, Split]:
    """Deterministic apartment-level stratified split.

    Returns ``{apartment_seed: "train" | "val" | "test"}``.
    Stratified by *apt_type* so each split has similar apartment-type
    proportions.
    """
    apt_seeds = [a.seed for a in apartments]
    apt_types = [a.apt_type for a in apartments]

    # Step 1: train vs (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=ratios[1] + ratios[2],
        random_state=seed,
    )
    train_idx, rest_idx = next(sss1.split(apt_seeds, apt_types))

    # Step 2: val vs test from the remainder
    rest_types = [apt_types[i] for i in rest_idx]
    val_frac = ratios[1] / (ratios[1] + ratios[2])
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac,
        random_state=seed,
    )
    val_sub, test_sub = next(sss2.split(rest_idx, rest_types))

    split_map: dict[int, Split] = {}
    for i in train_idx:
        split_map[apt_seeds[i]] = "train"
    for i in val_sub:
        split_map[apt_seeds[rest_idx[i]]] = "val"
    for i in test_sub:
        split_map[apt_seeds[rest_idx[i]]] = "test"

    return split_map


def get_rooms_by_split(
    apartments: list[Apartment],
    split_map: dict[int, Split],
) -> dict[Split, list[Room]]:
    """Partition rooms into train/val/test using the apartment-level split."""
    result: dict[Split, list[Room]] = {"train": [], "val": [], "test": []}
    for apt in apartments:
        split = split_map[apt.seed]
        result[split].extend(apt.rooms)
    return result


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load apartments.jsonl and print summary statistics."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute and accept a new data hash",
    )
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()

    apts = load_apartments(data_path=args.data_path, update_manifest=args.update)
    splits = assign_splits(apts)
    rooms = get_rooms_by_split(apts, splits)

    total_rooms = sum(len(a.rooms) for a in apts)
    print(f"Loaded {len(apts)} apartments, {total_rooms} active rooms")
    for s in ("train", "val", "test"):
        n = len(rooms[s])
        print(f"  {s}: {n} rooms ({n / total_rooms:.1%})")
