"""Floor plan dataclasses and scoring-rule loader for the WP2 evaluation module.

This module defines the shared input format for all reward functions
(furnishability, daylight, circulation). All scoring rules live in
``src/evaluation/rules/*.json`` — not hardcoded here — so rule adjustments
require no code changes.

Frozen dataclasses
------------------
``np.ndarray`` is not hashable, so the frozen dataclasses that contain arrays
set ``__hash__ = None`` explicitly to raise a clear ``TypeError`` if you attempt
to hash them, rather than silently succeeding with a wrong result. Use the ``id``
field (fixture ID string) or ``apartment_type`` for dict keys instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Type enumerations ──────────────────────────────────────────────────────────
#
# Canonical source for surrogate (indices 0-8) + evaluation-only Hallway (9).
# These lists must stay in sync with ``src/furnisher_surrogate/data.py:ROOM_TYPES``
# for the 9 scored types (Hallway is evaluation-only; the surrogate never trains on it).

ROOM_TYPES: list[str] = [
    "Bedroom",      # 0
    "Living room",  # 1
    "Bathroom",     # 2
    "WC",           # 3
    "Kitchen",      # 4
    "Children 1",   # 5
    "Children 2",   # 6
    "Children 3",   # 7
    "Children 4",   # 8
    "Hallway",      # 9 — evaluation only; not in the surrogate's training set
]

APT_TYPES: list[str] = [
    "Studio (bedroom)",  # 0
    "Studio (living)",   # 1
    "1-Bedroom",         # 2
    "2-Bedroom",         # 3
    "3-Bedroom",         # 4
    "4-Bedroom",         # 5
    "5-Bedroom",         # 6
]

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoomLayout:
    """One room in the floor plan — geometry only, no scores.

    Parameters
    ----------
    room_type : str
        One of ``ROOM_TYPES``.
    polygon : np.ndarray
        Shape ``(N, 2)``, float64. Closed polyline in meters
        (first point == last point). Winding order is not enforced —
        algorithms must use orientation-agnostic methods (e.g. ray casting).
    """

    room_type: str
    polygon: np.ndarray  # (N, 2) closed polyline, meters

    # Frozen dataclasses with ndarray fields cannot be hashed.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ApartmentLayout:
    """Complete apartment geometry — the input to every reward function.

    Parameters
    ----------
    id : str
        Unique identifier for the apartment (e.g. "H01", "D03").
    apartment_type : str
        One of ``APT_TYPES``.
    entrance : np.ndarray
        Shape ``(2,)``, float64. Point on the outer wall that marks the
        apartment entrance (front door location), in meters.
    outer_polygon : np.ndarray
        Shape ``(N, 2)``, float64. Closed polyline of the apartment
        boundary, in meters. Winding order is not enforced.
    outer_is_exterior : tuple[bool, ...]
        Length ``N - 1``. ``outer_is_exterior[i]`` is ``True`` if the edge
        from ``outer_polygon[i]`` to ``outer_polygon[i+1]`` faces outside
        (facade or exposed gable). ``False`` for party walls.
    doors : tuple[tuple[float, float], ...]
        All interior door openings. Each door is an ``(x, y)`` point on the
        shared wall between two rooms. Two rooms are adjacent (BFS) if any
        door point lies on both their polygon boundaries. The surrogate uses
        the door point on a room's boundary as its furniture-placement door.
    rooms : tuple[RoomLayout, ...]
        All rooms. Order is not significant.
    """

    id: str
    apartment_type: str
    entrance: np.ndarray              # (2,) meters
    outer_polygon: np.ndarray         # (N, 2) meters, closed
    outer_is_exterior: tuple[bool, ...]  # length N-1, one flag per outer edge
    doors: tuple[tuple[float, float], ...]  # interior door openings (x, y)
    rooms: tuple[RoomLayout, ...]

    # Frozen dataclasses with ndarray fields cannot be hashed.
    __hash__ = None  # type: ignore[assignment]


# ── Rule loader ───────────────────────────────────────────────────────────────

_RULES_DIR = Path(__file__).parent / "rules"


def load_rules(domain: str, rules_path: Path | None = None) -> dict:
    """Load scoring rules for a domain from JSON.

    Rules are loaded lazily (at function call time), so swapping a rules file
    before calling a scoring function takes effect immediately.

    Parameters
    ----------
    domain : str
        One of: ``'circulation'``, ``'daylight'``, ``'furnishability'``,
        ``'composite'``.
    rules_path : Path, optional
        Override the default rules file. Useful for per-experiment configs or
        unit tests that need isolated rule sets.

    Returns
    -------
    dict
        The parsed JSON rules document (``_doc`` fields are included; callers
        may ignore them).

    Raises
    ------
    FileNotFoundError
        If the rules file does not exist at the resolved path.
    """
    path = rules_path or (_RULES_DIR / f"{domain}.json")
    with open(path) as f:
        return json.load(f)
