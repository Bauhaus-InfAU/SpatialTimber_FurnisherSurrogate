"""Rasterize room polygons into 64x64 3-channel images for CNN input.

Each room becomes a uint8 array of shape (3, 64, 64):
  Channel 0: Room interior mask (0 or 255)
  Channel 1: Wall edges (0 or 255)
  Channel 2: Door marker — gaussian blob (0-255, peak=255)

Pure numpy + Pillow. No torch dependency (Grasshopper can reuse this).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from .data import Room

# ── Constants ─────────────────────────────────────────────────

IMG_SIZE: int = 64  # output image dimension (square)
FIT_SIZE: int = 60  # polygon longest side fits this many pixels
DOOR_SIGMA: float = 2.0  # gaussian blob sigma in pixels


# ── Coordinate transform ─────────────────────────────────────


def polygon_to_pixel_coords(
    polygon: np.ndarray,
    img_size: int = IMG_SIZE,
    fit_size: int = FIT_SIZE,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Transform polygon from meter-space to pixel-space.

    The longest polygon side is scaled to *fit_size* pixels and centered
    in the *img_size* x *img_size* grid.  Y-axis is flipped so that the
    image top-left origin matches visual expectation.

    Returns
    -------
    pixel_polygon : (N, 2) float64 — polygon vertices in pixel coords (closed)
    scale : float — meters-to-pixels scale factor
    offset : (2,) float64 — translation applied after scaling
    """
    pts = polygon[:-1]  # strip closing vertex
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = maxs - mins

    longest_side = extent.max()
    if longest_side < 1e-6:
        # Degenerate polygon — return center point
        scale = 1.0
        offset = np.array([img_size / 2.0, img_size / 2.0])
        pixel_pts = np.full_like(pts, img_size / 2.0)
    else:
        scale = fit_size / longest_side
        scaled = (pts - mins) * scale
        padding = (img_size - extent * scale) / 2.0
        pixel_pts = scaled + padding

    # Flip y so image origin (top-left) matches visual expectation
    pixel_pts[:, 1] = (img_size - 1) - pixel_pts[:, 1]

    # Re-close polygon
    pixel_polygon = np.vstack([pixel_pts, pixel_pts[0:1]])

    # Combined offset for point_to_pixel: p_pixel = (p - mins) * scale + padding, then y-flip
    if longest_side >= 1e-6:
        offset = -mins * scale + padding
    return pixel_polygon, scale, offset


def point_to_pixel(
    point: np.ndarray,
    scale: float,
    offset: np.ndarray,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Transform a single point using the same scale+offset as the polygon."""
    px = point * scale + offset
    px[1] = (img_size - 1) - px[1]  # y-flip
    return px


# ── Channel renderers ────────────────────────────────────────


def _render_mask(pixel_polygon: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Render filled polygon mask. Returns (img_size, img_size) uint8, {0, 255}."""
    img = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    pts = pixel_polygon[:-1]  # strip closing vertex for PIL
    xy = [(round(float(p[0])), round(float(p[1]))) for p in pts]
    draw.polygon(xy, fill=255)
    return np.array(img, dtype=np.uint8)


def _render_edges(pixel_polygon: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Render polygon boundary edges. Returns (img_size, img_size) uint8, {0, 255}."""
    img = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    for i in range(len(pixel_polygon) - 1):
        x0, y0 = float(pixel_polygon[i, 0]), float(pixel_polygon[i, 1])
        x1, y1 = float(pixel_polygon[i + 1, 0]), float(pixel_polygon[i + 1, 1])
        draw.line(
            [(round(x0), round(y0)), (round(x1), round(y1))],
            fill=255,
            width=1,
        )
    return np.array(img, dtype=np.uint8)


def _render_door(
    door_pixel: np.ndarray,
    img_size: int = IMG_SIZE,
    sigma: float = DOOR_SIGMA,
) -> np.ndarray:
    """Render gaussian blob at door position. Returns (img_size, img_size) uint8, 0-255."""
    y_coords = np.arange(img_size, dtype=np.float64)
    x_coords = np.arange(img_size, dtype=np.float64)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    cx, cy = float(door_pixel[0]), float(door_pixel[1])
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    gauss = np.exp(-dist_sq / (2.0 * sigma**2))

    return np.clip(gauss * 255, 0, 255).astype(np.uint8)


# ── Public API ────────────────────────────────────────────────


def rasterize_arrays(
    polygon: np.ndarray,
    door: np.ndarray,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """Rasterize from raw numpy arrays. No Room dependency.

    Parameters
    ----------
    polygon : (N, 2) float64 — closed polyline in meters
    door : (2,) float64 — point on wall in meters

    Returns
    -------
    np.ndarray of shape (3, img_size, img_size), dtype uint8.
    """
    pixel_poly, scale, offset = polygon_to_pixel_coords(polygon, img_size)
    door_px = point_to_pixel(door.copy(), scale, offset, img_size)

    ch0 = _render_mask(pixel_poly, img_size)
    ch1 = _render_edges(pixel_poly, img_size)
    ch2 = _render_door(door_px, img_size)

    return np.stack([ch0, ch1, ch2], axis=0)


def rasterize_room(room: Room, img_size: int = IMG_SIZE) -> np.ndarray:
    """Convert a Room to a 3-channel image.

    Convenience wrapper around :func:`rasterize_arrays`.

    Returns
    -------
    np.ndarray of shape (3, img_size, img_size), dtype uint8.
    """
    return rasterize_arrays(room.polygon, room.door, img_size)


# ── Batch pre-rasterization ──────────────────────────────────


def precompute_dataset(
    rooms: list[Room],
    output_path: str | None = None,
) -> np.ndarray:
    """Rasterize all rooms and optionally save to .npz.

    Saved arrays: images, scores, room_type_idx, apartment_type_idx, area,
    door_rel_x, door_rel_y, apartment_seeds.
    """
    from tqdm import tqdm

    from .features import area as compute_area
    from .features import door_rel_position

    n = len(rooms)
    images = np.empty((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    scores = np.empty(n, dtype=np.float32)
    room_type_idx = np.empty(n, dtype=np.int8)
    apartment_type_idx = np.empty(n, dtype=np.int8)
    areas = np.empty(n, dtype=np.float32)
    door_rel_x = np.empty(n, dtype=np.float32)
    door_rel_y = np.empty(n, dtype=np.float32)
    apartment_seeds = np.empty(n, dtype=np.int64)

    for i, room in enumerate(tqdm(rooms, desc="Rasterizing")):
        images[i] = rasterize_room(room)
        scores[i] = room.score if room.score is not None else -1.0
        room_type_idx[i] = room.room_type_idx
        apartment_type_idx[i] = room.apartment_type_idx if room.apartment_type_idx is not None else -1
        areas[i] = compute_area(room)
        dx, dy = door_rel_position(room)
        door_rel_x[i] = dx
        door_rel_y[i] = dy
        apartment_seeds[i] = room.apartment_seed if room.apartment_seed is not None else -1

    if output_path is not None:
        np.savez_compressed(
            output_path,
            images=images,
            scores=scores,
            room_type_idx=room_type_idx,
            apartment_type_idx=apartment_type_idx,
            area=areas,
            door_rel_x=door_rel_x,
            door_rel_y=door_rel_y,
            apartment_seeds=apartment_seeds,
        )

    return images


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Pre-rasterize all rooms to .npz for CNN training."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "rooms_rasterized.npz",
    )
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()

    from .data import load_rooms

    rooms = load_rooms(data_path=args.data_path)
    print(f"Loaded {len(rooms)} rooms")

    precompute_dataset(rooms, output_path=str(args.output))
    file_size = args.output.stat().st_size / 1e6
    print(f"Saved to {args.output} ({file_size:.1f} MB)")
