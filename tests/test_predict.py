"""Tests for the predict_score inference API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from furnisher_surrogate.predict import predict_score, _model_cache

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "test_rooms.json"
MODEL_PATH = Path(__file__).parent.parent / "models" / "cnn_v1.pt"


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear model cache between tests to avoid cross-test interference."""
    _model_cache.clear()
    yield
    _model_cache.clear()


@pytest.fixture
def fixtures():
    with open(FIXTURES_PATH) as f:
        return json.load(f)


# ── Fixture-based score tests ──────────────────────────────────────


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="cnn_v1.pt not found")
def test_fixture_scores(fixtures):
    """predict_score matches expected scores for all fixture rooms."""
    for room in fixtures:
        polygon = np.array(room["polygon"], dtype=np.float64)
        door = np.array(room["door"], dtype=np.float64)
        expected = room["expected_score_cnn_v1"]

        score = predict_score(polygon, door, room["room_type"], model_path=MODEL_PATH)

        assert abs(score - expected) < 0.01, (
            f"{room['name']}: expected {expected}, got {score}"
        )


# ── Input validation ───────────────────────────────────────────────


def test_invalid_room_type():
    """Unknown room type raises ValueError."""
    polygon = np.array([[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]], dtype=np.float64)
    door = np.array([2.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Unknown room_type"):
        predict_score(polygon, door, "Garage")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="cnn_v1.pt not found")
def test_auto_close_polygon():
    """Unclosed polygon is auto-closed and produces same result as closed."""
    closed = np.array([[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]], dtype=np.float64)
    unclosed = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float64)
    door = np.array([2.0, 0.0], dtype=np.float64)

    score_closed = predict_score(closed, door, "Bedroom", model_path=MODEL_PATH)
    score_unclosed = predict_score(unclosed, door, "Bedroom", model_path=MODEL_PATH)

    assert abs(score_closed - score_unclosed) < 0.001


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="cnn_v1.pt not found")
def test_score_clamped_to_0_100():
    """Output is always in [0, 100]."""
    polygon = np.array([[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]], dtype=np.float64)
    door = np.array([2.0, 0.0], dtype=np.float64)

    for rt in ["Bedroom", "Kitchen", "Bathroom", "WC", "Living room"]:
        score = predict_score(polygon, door, rt, model_path=MODEL_PATH)
        assert 0.0 <= score <= 100.0, f"{rt}: score {score} out of range"


# ── Model caching ─────────────────────────────────────────────────


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="cnn_v1.pt not found")
def test_model_caching():
    """Model is loaded once and cached on subsequent calls."""
    polygon = np.array([[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]], dtype=np.float64)
    door = np.array([2.0, 0.0], dtype=np.float64)

    assert len(_model_cache) == 0
    predict_score(polygon, door, "Bedroom", model_path=MODEL_PATH)
    assert len(_model_cache) == 1

    # Second call should not add a new cache entry
    predict_score(polygon, door, "Kitchen", model_path=MODEL_PATH)
    assert len(_model_cache) == 1


# ── Rasterize decoupling ──────────────────────────────────────────


def test_rasterize_arrays_matches_rasterize_room():
    """rasterize_arrays produces identical output to rasterize_room."""
    from furnisher_surrogate.data import Room, ROOM_TYPE_TO_IDX
    from furnisher_surrogate.rasterize import rasterize_arrays, rasterize_room

    polygon = np.array([[0, 0], [5, 0], [5, 4], [0, 4], [0, 0]], dtype=np.float64)
    door = np.array([2.5, 0.0], dtype=np.float64)

    room = Room(
        polygon=polygon,
        door=door,
        room_type="Bedroom",
        room_type_idx=ROOM_TYPE_TO_IDX["Bedroom"],
        score=None,
        apartment_seed=None,
        apartment_type=None,
    )

    img_room = rasterize_room(room)
    img_arrays = rasterize_arrays(polygon, door)

    assert np.array_equal(img_room, img_arrays)
