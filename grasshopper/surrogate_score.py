"""Surrogate score predictor — GhPython component for Rhino 8.

Inputs (set in Grasshopper component):
    polygon : Polyline  — room boundary (closed polyline)
    door    : Point3d   — door position on wall
    room_type : str     — one of: Bedroom, Living room, Bathroom, WC,
                          Kitchen, Children 1-4
    apartment_type : str — one of: Studio (bedroom), Studio (living),
                           1-Bedroom, 2-Bedroom, 3-Bedroom, 4-Bedroom,
                           5-Bedroom
    model_path : str    — (optional) path to .pt checkpoint

Output:
    score : float — predicted furniture placement score (0-100)
"""

import numpy as np
from furnisher_surrogate.predict import predict_score

# Convert Rhino Polyline → numpy (N, 2) array
# polygon is a Rhino Polyline; iterate its points
poly_np = np.array(
    [[pt.X, pt.Y] for pt in polygon],
    dtype=np.float64,
)

# Ensure closed polyline
if not np.allclose(poly_np[0], poly_np[-1]):
    poly_np = np.vstack([poly_np, poly_np[0:1]])

# Convert Rhino Point3d → numpy (2,) array
door_np = np.array([door.X, door.Y], dtype=np.float64)

# Predict
score = predict_score(
    poly_np,
    door_np,
    room_type,
    apartment_type=apartment_type if apartment_type else None,
    model_path=model_path if model_path else None,
)
