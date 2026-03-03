"""WP2 Evaluation module — reward functions for furnishability, daylight, circulation.

All reward functions share the ``ApartmentLayout`` / ``RoomLayout`` dataclasses
defined in ``apartment.py``. Scoring rules live in ``rules/*.json`` and are
loaded via ``load_rules()``.
"""

from .apartment import (
    APT_TYPES,
    ROOM_TYPES,
    ApartmentLayout,
    RoomLayout,
    load_rules,
)

__all__ = [
    "APT_TYPES",
    "ApartmentLayout",
    "load_rules",
    "ROOM_TYPES",
    "RoomLayout",
]
