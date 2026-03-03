#! python 3
"""Apartment writer — GhPython component for Rhino 8.

Takes geometry drawn in Rhino and exports a valid apartment fixture JSON.
Primary use: Luyang exports her floor plans from Rhino to the fixture format
expected by the evaluation functions.

No external package imports — only json (stdlib) and Rhino.Geometry.

Inputs:
    apt_id               [Item] str        — unique ID (e.g. "L01")
    apartment_type       [Item] str        — must be one of the 7 APT_TYPES
    outer_polygon        [Item] Polyline   — CCW closed apartment boundary
    entrance             [Item] Point3d    — point on outer wall
    outer_is_exterior    [List] bool       — one flag per outer polygon edge (len == len(outer_polygon) - 1)
    doors                [List] Point3d    — all interior door openings, one Point3d per door
    room_polygons        [List] Polyline   — one per room, CCW closed
    room_types           [List] str        — parallel to room_polygons
    expected_daylight    [Item] float|None — 0–100 or leave empty
    expected_circulation [Item] float|None — 0–100 or leave empty
    output_path          [Item] str        — where to write the JSON
    write                [Item] bool       — toggle/button trigger

Outputs:
    json_str [Item] str — pretty-printed JSON (for preview panel)
    status   [Item] str — "Written to /path/to/file.json" or error message
"""

import json
import Rhino.Geometry as rg

APT_TYPES = [
    "Studio (bedroom)", "Studio (living)", "1-Bedroom",
    "2-Bedroom", "3-Bedroom", "4-Bedroom", "5-Bedroom",
]

ROOM_TYPES = [
    "Bedroom", "Living room", "Bathroom", "WC", "Kitchen",
    "Children 1", "Children 2", "Children 3", "Children 4", "Hallway",
]

# --- reset outputs ---
json_str = ""
status = ""

try:
    # --- validate ---
    errors = []

    if not apt_id:
        errors.append("apt_id is required")

    if apartment_type not in APT_TYPES:
        errors.append("apartment_type '{}' not in APT_TYPES ({})".format(
            apartment_type, ", ".join(APT_TYPES)
        ))

    if outer_polygon is None:
        errors.append("outer_polygon is required")

    if entrance is None:
        errors.append("entrance is required")

    n_outer_edges = (outer_polygon.Count - 1) if outer_polygon is not None else 0
    n_ext_flags = len(outer_is_exterior) if outer_is_exterior else 0
    if outer_polygon is not None and n_ext_flags != n_outer_edges:
        errors.append(
            "outer_is_exterior length ({}) must equal len(outer_polygon) - 1 ({})".format(
                n_ext_flags, n_outer_edges
            )
        )

    if not doors:
        errors.append("doors list is required (at least one door point)")

    n_rooms = len(room_polygons) if room_polygons else 0
    n_types = len(room_types) if room_types else 0
    if n_rooms != n_types:
        errors.append(
            "room_polygons ({}) and room_types ({}) must have equal length".format(
                n_rooms, n_types
            )
        )

    for i, rt in enumerate(room_types or []):
        if rt not in ROOM_TYPES:
            errors.append("room_types[{}] '{}' not in ROOM_TYPES".format(i, rt))

    if errors:
        status = "ERROR: " + "; ".join(errors)
    else:
        def poly_to_list(poly):
            return [[round(pt.X, 4), round(pt.Y, 4)] for pt in poly]

        rooms_data = []
        for poly, rt in zip(room_polygons, room_types):
            rooms_data.append({
                "room_type": rt,
                "polygon": poly_to_list(poly),
            })

        apt_dict = {
            "id": apt_id,
            "apartment_type": apartment_type,
            "entrance": [round(entrance.X, 4), round(entrance.Y, 4)],
            "outer_polygon": poly_to_list(outer_polygon),
            "outer_is_exterior": [bool(f) for f in outer_is_exterior],
            "doors": [[round(d.X, 4), round(d.Y, 4)] for d in doors],
            "rooms": rooms_data,
            "expected_scores": {
                "daylight": expected_daylight if expected_daylight is not None else None,
                "circulation": expected_circulation if expected_circulation is not None else None,
                "furnitability": None,
            },
        }

        json_str = json.dumps([apt_dict], indent=2)

        if write:
            with open(output_path, "w") as f:
                f.write(json_str)
            status = "Written to {}".format(output_path)
        else:
            status = "Preview only (set write=True to save)"

except Exception as e:
    status = "ERROR: {}".format(e)
