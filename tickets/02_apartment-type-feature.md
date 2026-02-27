# Apartment type as model feature

**ID:** 02
**Type:** improvement
**Priority:** high
**Noticed:** 2026-02-26
**Status:** resolved
**Phase:** `plans/07-grasshopper.md` (affects baseline, CNN, and deployment)

## Context

The furnisher scores rooms differently depending on the apartment type — the same room geometry + room type + door position can receive different scores in different apartment contexts (e.g. a bedroom in a 1-room apartment vs a 4-room apartment requires different furniture). The `apartment_type` field exists in the data (`Apartment.apt_type`, `Room.apartment_type`) but is currently used **only** for stratifying the train/val/test split, never as a model input.

**Current state across the pipeline:**

| Component | `apartment_type` present? | Used as feature? |
|-----------|:---:|:---:|
| `data.py` — `Apartment.apt_type` | Yes | No (stratification only) |
| `data.py` — `Room.apartment_type` | Yes | No (marked "provenance") |
| `features.py` — `extract_features()` | No | No |
| `models.py` — `RoomCNN` | No | No |
| `train.py` — `RoomDataset` | No | No |
| `predict.py` — `predict_score()` | No | No (not a parameter) |
| `grasshopper/surrogate_score.py` | No | No (not a component input) |

## Description

Investigate whether adding `apartment_type` as a model feature improves prediction accuracy:

1. **EDA**: Check how many distinct apartment types exist, their distribution, and whether score distributions differ significantly per (room_type, apartment_type) pair. If scores are identical across apartment types for the same room geometry, this feature adds no value.

2. **Baseline model**: Add `apartment_type` as a categorical feature in LightGBM (one-hot or native categorical). Compare MAE to current baseline (11.02).

3. **CNN model**: Add `apartment_type` as an additional embedding input alongside `room_type`. Compare MAE to current best (11.23).

4. **Inference API**: If the feature helps, add `apartment_type` as a required parameter to `predict_score()`.

5. **Grasshopper component**: Add `apartment_type` as a component input.

## Acceptance Criteria

- [x] EDA confirms whether apartment type affects scores for identical room configurations
- [x] Baseline retrained with apartment_type; MAE compared to 11.02
- [x] If beneficial: `predict_score()` accepts `apartment_type` parameter
- [x] If beneficial: Grasshopper component accepts `apartment_type` input
- [x] Decision documented in relevant phase plan's Decisions Log
