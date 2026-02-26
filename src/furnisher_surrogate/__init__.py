"""Furnisher Surrogate — predict furniture placement scores from room geometry."""

from .data import (
    ROOM_TYPES,
    ROOM_TYPE_TO_IDX,
    Apartment,
    Room,
    load_apartments,
    load_rooms,
)
from .rasterize import IMG_SIZE, rasterize_room

__all__ = [
    "ROOM_TYPES",
    "ROOM_TYPE_TO_IDX",
    "Apartment",
    "Room",
    "load_apartments",
    "load_rooms",
    "IMG_SIZE",
    "rasterize_room",
]
