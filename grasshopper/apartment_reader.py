#! python 3
"""Apartment reader — GhPython component for Rhino 8.

Loads an apartment fixture JSON and reconstructs it as Rhino geometry.
Primary uses:
  - Visually verify hand-crafted fixtures are geometrically correct
  - Inspect Luyang's labeled test apartments before running evaluation

No external package imports — only json (stdlib) and Rhino.Geometry.

Inputs:
    json_path  [Item] str — path to .json file (array of apartments, or single object)
    index      [Item] int — which apartment to load (0-based, default 0)

Outputs:
    apt_id               [Item] str        — the `id` field
    apartment_type       [Item] str        — e.g. "2-Bedroom"
    outer_polygon        [Item] Polyline   — apartment boundary
    entrance             [Item] Point3d    — entrance point on outer wall
    outer_is_exterior    [List] bool       — True = facade, one flag per outer polygon edge
    doors                [List] Point3d    — all interior door openings
    room_polygons        [List] Polyline   — one per room
    room_types           [List] str        — parallel to room_polygons
    expected_daylight    [Item] float|None — from expected_scores.daylight
    expected_circulation [Item] float|None — from expected_scores.circulation
    status               [Item] str        — e.g. "Loaded H01 (2-Bedroom) — 3 rooms, 2 doors"
"""

import json
import Rhino.Geometry as rg

# --- reset outputs ---
apt_id = None
apartment_type = None
outer_polygon = None
entrance = None
outer_is_exterior = []
doors = []
room_polygons = []
room_types = []
expected_daylight = None
expected_circulation = None
status = ""

try:
    # --- load file ---
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except IOError:
        status = "ERROR: file not found — {}".format(json_path)
        raise StopIteration

    # normalise to list
    if isinstance(data, dict):
        apartments = [data]
    else:
        apartments = data

    idx = index if index is not None else 0
    if idx < 0 or idx >= len(apartments):
        status = "ERROR: index {} out of range (file has {} apartments)".format(
            idx, len(apartments)
        )
        raise StopIteration

    apt = apartments[idx]

    # --- required top-level fields ---
    for field in ("id", "apartment_type", "entrance", "outer_polygon",
                  "outer_is_exterior", "doors", "rooms"):
        if field not in apt:
            status = "ERROR: missing field '{}'".format(field)
            raise StopIteration

    # --- scalars ---
    apt_id = apt["id"]
    apartment_type = apt["apartment_type"]

    # --- entrance ---
    ex, ey = apt["entrance"]
    entrance = rg.Point3d(ex, ey, 0)

    # --- outer polygon ---
    outer_polygon = rg.Polyline(
        [rg.Point3d(x, y, 0) for x, y in apt["outer_polygon"]]
    )

    # --- outer_is_exterior ---
    outer_is_exterior = list(apt["outer_is_exterior"])

    # --- doors ---
    for dx, dy in apt["doors"]:
        doors.append(rg.Point3d(dx, dy, 0))

    # --- rooms ---
    for room in apt["rooms"]:
        room_types.append(room["room_type"])
        pts = [rg.Point3d(x, y, 0) for x, y in room["polygon"]]
        room_polygons.append(rg.Polyline(pts))

    # --- expected scores ---
    es = apt.get("expected_scores", {})
    expected_daylight = es.get("daylight")
    expected_circulation = es.get("circulation")

    status = "Loaded {} ({}) — {} rooms, {} doors".format(
        apt_id, apartment_type, len(room_polygons), len(doors)
    )

except StopIteration:
    pass
except Exception as e:
    status = "ERROR: {}".format(e)
