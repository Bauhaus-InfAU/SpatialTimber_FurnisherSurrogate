"""Furnisher Surrogate — predict furniture placement scores from room geometry."""

from .rasterize import IMG_SIZE, rasterize_arrays, rasterize_room

# Training-time imports need sklearn, which may not be installed in
# lightweight inference environments (e.g. Grasshopper / Rhino 8).
try:
    from .data import (
        APT_TYPES,
        APT_TYPE_TO_IDX,
        ROOM_TYPES,
        ROOM_TYPE_TO_IDX,
        Apartment,
        Room,
        load_apartments,
        load_rooms,
    )
except ImportError:
    pass

try:
    from .models import RoomCNN
except ImportError:
    pass

try:
    from .predict import predict_score
except ImportError:
    pass

__all__ = [
    "APT_TYPES",
    "APT_TYPE_TO_IDX",
    "IMG_SIZE",
    "predict_score",
    "rasterize_arrays",
    "rasterize_room",
    "ROOM_TYPES",
    "ROOM_TYPE_TO_IDX",
    "Apartment",
    "Room",
    "load_apartments",
    "load_rooms",
    "RoomCNN",
]
