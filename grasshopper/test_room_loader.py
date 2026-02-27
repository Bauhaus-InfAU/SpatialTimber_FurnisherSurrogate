#! python 3
# r: numpy
"""Test room loader — GhPython component for Rhino 8.

Reads test_rooms.json and outputs geometry + metadata as parallel lists,
ready to wire into the surrogate_score component.

Inputs:
    json_path : str — full path to test_rooms.json

Outputs:
    polygons        — room boundary polylines
    doors           — door positions
    room_types      — room type strings
    apartment_types — apartment type strings
    names           — room names for labelling
    expected_scores — expected cnn_v4 scores (apartment_type defaults to 2-Bedroom)
"""

import json
import Rhino.Geometry as rg

with open(json_path, "r") as f:
    data = json.load(f)

polygons = []
doors = []
room_types = []
apartment_types = []
names = []
expected_scores = []

for room in data["rooms"]:
    pts = [rg.Point3d(x, y, 0) for x, y in room["polygon"]]
    polygons.append(rg.Polyline(pts))

    dx, dy = room["door"]
    doors.append(rg.Point3d(dx, dy, 0))

    room_types.append(room["room_type"])
    apartment_types.append(room["apartment_type"])
    names.append(room["name"])
    expected_scores.append(room["expected_score"])
