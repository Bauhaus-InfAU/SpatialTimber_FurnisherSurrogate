# Scoring Specification — WP2 Reward Functions

> **Single source of truth.** This document defines all score ranges, applicable room types, aggregation rules, and the composite formula for the three WP2 reward functions. Implementation lives in `src/evaluation/`; rule parameters live in `src/evaluation/rules/*.json`.

---

## Overview

Each apartment produces three domain scores and one composite score, all in the range **0–100**:

| Score | Domain | Source |
|-------|--------|--------|
| `furnitability` | Can furniture be placed in each room? | CNN/LightGBM surrogate model |
| `daylight` | Do habitable rooms have exterior wall access? | Geometric check (topological) |
| `circulation` | Are all rooms reachable from the entrance within threshold? | BFS graph check (topological) |
| `composite` | Weighted average of the three domain scores | Weighted formula |

---

## Shared Input Format

All reward functions take an `ApartmentLayout` object (defined in `src/evaluation/apartment.py`):

```
ApartmentLayout
├── id: str                          # unique identifier
├── apartment_type: str              # one of APT_TYPES
├── entrance: (x, y)                 # point on outer wall — apartment front door
├── outer_polygon: [(x,y), ...]      # closed CCW apartment boundary, meters
├── walls: [WallSegment, ...]        # all walls tagged exterior/interior
│   └── WallSegment
│       ├── start, end: (x, y)
│       └── is_exterior: bool
└── rooms: [RoomLayout, ...]
    └── RoomLayout
        ├── room_type: str           # one of ROOM_TYPES
        ├── polygon: [(x,y), ...]    # closed CCW room boundary, meters
        └── door: (x, y)            # single door point on room wall
```

**Coordinate conventions:**
- All units in **meters**
- Polygon winding: **counter-clockwise (CCW)**
- Polygons are **closed**: `polygon[0] == polygon[-1]`
- `entrance` and `door` are points on wall boundaries (within 5 cm tolerance)

**Room types** (canonical list in `src/evaluation/apartment.py`):

| Index | Type | Surrogate | Daylight | Circulation max dist |
|-------|------|-----------|----------|----------------------|
| 0 | Bedroom | scored | habitable | 1 |
| 1 | Living room | scored | habitable | 1 |
| 2 | Bathroom | scored | non-habitable | 1 |
| 3 | WC | scored | non-habitable | 1 |
| 4 | Kitchen | scored | habitable | 1 |
| 5 | Children 1 | scored | habitable | 2 |
| 6 | Children 2 | scored | habitable | 2 |
| 7 | Children 3 | scored | habitable | 2 |
| 8 | Children 4 | scored | habitable | 2 |
| 9 | Hallway | **not scored** | non-habitable | **0** (must be entry room) |

---

## 1. Furnishability Score

**Question:** Can standard furniture be placed in each room?

**Method:** Per-room prediction from the surrogate model (CNN v4 or LightGBM baseline). The model takes room polygon + door position + room type + apartment type and outputs a score 0–100.

**Per-room score:**
- Scored room types: all except Hallway (see `src/evaluation/rules/furnishability.json`)
- Hallway: score = `null` (excluded from aggregate)

**Apartment aggregate:**
```
furnitability_score = mean(score for room in rooms if room is scored)
```

**Score semantics:**
| Range | Interpretation |
|-------|---------------|
| 90–100 | Excellent — standard layout works |
| 70–89 | Good — minor constraints |
| 40–69 | Problematic — significant constraints |
| 1–39 | Poor — very difficult to furnish |
| 0 | Failed — furniture cannot be placed |

**Rules file:** `src/evaluation/rules/furnishability.json`

---

## 2. Daylight Score

**Question:** Do habitable rooms have access to exterior walls (natural light)?

**Method:** For each habitable room, check whether any edge of the room polygon collinearly overlaps with at least one exterior `WallSegment` (`is_exterior=true`) within a 1 cm perpendicular tolerance.

**Habitable rooms** (must have exterior wall): Bedroom, Living room, Kitchen, Children 1–4.
**Non-habitable** (excluded): Bathroom, WC, Hallway.

**Per-room score:**
- `100` — at least one polygon edge overlaps an exterior wall
- `0` — no polygon edge overlaps any exterior wall
- non-habitable rooms: excluded from aggregate (`null`)

**Apartment aggregate:**
```
daylight_score = (habitable rooms with exterior wall / total habitable rooms) × 100
```

**Edge case — no habitable rooms:**
If the apartment contains zero habitable rooms (e.g. only WC + Bathroom), `daylight_score = 100` (vacuous pass — no habitable rooms to fail). This prevents utility-room-only apartments from being penalised. The value can be changed in `rules/daylight.json → no_habitable_rooms_score`.

**Rules file:** `src/evaluation/rules/daylight.json`

---

## 3. Circulation Score

**Question:** Is every room accessible from the apartment entrance within an acceptable number of steps?

**Method:** Build a room adjacency graph, then BFS from the entrance room. Each room receives a binary pass/fail based on whether its BFS distance is within the type-specific maximum.

### Step 1 — Entrance room detection

The **entry room** is the room whose polygon boundary contains `apartment.entrance` within `entry_detection.tolerance_m = 0.05` m (5 cm).

- If the apartment has a Hallway that contains the entrance → Hallway is the entry room (distance 0).
- If no Hallway exists → whichever room contains the entrance is the entry room (distance 0). This supports studio layouts where the entrance opens directly into the living room.

### Step 2 — Room adjacency graph

Two rooms **A** and **B** are adjacent if:
- the door point of A lies on a polygon edge of B (within `adjacency.door_tolerance_m = 0.05` m), **OR**
- the door point of B lies on a polygon edge of A

Adjacency is undirected. The graph is used for BFS.

### Step 3 — BFS distance scoring

For each room, compute BFS distance from the entry room. Then:

```
room_score = 100  if distance ≤ max_distance[room_type]
             0    otherwise
```

**Maximum distances** (from `src/evaluation/rules/circulation.json`):

| Room type | Max BFS distance | Notes |
|-----------|-----------------|-------|
| Hallway | 0 | Must BE the entry room; if at distance 1+, layout is inverted |
| Bedroom | 1 | Directly off hallway / entry room |
| Living room | 1 | Directly off hallway / entry room |
| Kitchen | 1 | Directly off hallway / entry room |
| Bathroom | 1 | Directly off hallway / entry room |
| WC | 1 | Directly off hallway / entry room |
| Children 1–4 | 2 | Can be accessed via another bedroom or directly off hallway |

**Apartment aggregate:**
```
circulation_score = (rooms with score=100 / total rooms) × 100
```

**Rules file:** `src/evaluation/rules/circulation.json`

---

## 4. Composite Score

**Formula:**
```
composite_score = w_f × furnitability_score
               + w_d × daylight_score
               + w_c × circulation_score
```

**Default weights** (from `src/evaluation/rules/composite.json`):

| Domain | Weight |
|--------|--------|
| furnitability | 0.333 |
| daylight | 0.333 |
| circulation | 0.334 |
| **Total** | **1.000** |

**Missing domain handling:** If a domain score is `null` (edge case — e.g. model unavailable), that domain is excluded and the remaining weights are renormalised. Controlled by `composite.json → missing_score_handling.method = "skip"`.

**Rules file:** `src/evaluation/rules/composite.json`

---

## Rule Files

### Design rationale

The scoring system separates **logic** from **policy**:

- **Logic** (Python) — *how* to compute a score: BFS traversal, polygon edge overlap, weighted average. This is stable and changes only if the algorithm changes.
- **Policy** (JSON) — *what* counts: which rooms are habitable, what distance thresholds to use, how to weight domains. These are research decisions that change between experiments.

Keeping policy in JSON files has four concrete benefits:

1. **No code changes for parameter tuning.** Changing the composite weight from ⅓ to 0.5 or relaxing a BFS threshold for Children rooms requires editing one JSON file — no Python, no tests, no PR.
2. **Per-experiment configurability.** Each RL training run can point to its own rule file via `load_rules(domain, rules_path=...)`. You can compare "strict circulation rules" vs "relaxed circulation rules" without touching the implementation.
3. **Self-documenting parameters.** Every value has a `_doc` field explaining what it means and the effect of changing it. Luyang or a future researcher can open the JSON and understand the rules without reading Python source code.
4. **Clear audit trail.** Rule files are version-controlled. A git diff on `circulation.json` shows exactly which thresholds changed between experiments, which is easier to review than a diff on a Python constant buried in a module.

The one trade-off: rule files must be kept consistent with SCORING.md (this document is the human-readable source of truth; JSON is the machine-readable one). When changing a rule, update both.

### Rule file reference

All scoring parameters are encoded in JSON config files — no hardcoded Python dicts or sets in the implementation.

| File | Domain | Key parameters |
|------|--------|---------------|
| `src/evaluation/rules/furnishability.json` | Furnishability | Which room types the surrogate scores |
| `src/evaluation/rules/daylight.json` | Daylight | Habitable room types; overlap tolerance; vacuous pass value |
| `src/evaluation/rules/circulation.json` | Circulation | Max BFS distances per room type; adjacency tolerance; entry detection method |
| `src/evaluation/rules/composite.json` | Composite | Domain weights; missing-domain strategy |

To run an experiment with modified rules, copy the relevant JSON file, edit the values, and pass the new path to `load_rules(domain, rules_path=Path("my_rules.json"))`.

---

## Output Structure

```python
# Per-room result
RoomScore(
    room_type: str,
    furnitability: float | None,   # null for Hallway
    daylight: float | None,        # null for non-habitable rooms
    circulation: float,            # 0 or 100
)

# Apartment-level result
ApartmentScore(
    rooms: tuple[RoomScore, ...],
    furnitability: float,          # 0–100 aggregate
    daylight: float,               # 0–100 aggregate
    circulation: float,            # 0–100 aggregate
    composite: float,              # 0–100 weighted
)
```

---

## Test Fixtures

Five hand-crafted fixtures in `tests/fixtures/apartments/hand_crafted.json` cover the key edge cases:

| ID | Description | daylight | circulation |
|----|-------------|----------|-------------|
| H01 | 2-bed, 2 rooms on facade, Hallway at entrance | 100 | 100 |
| H02 | 2-bed, no rooms on facade, Hallway at entrance | 0 | 100 |
| H03 | Studio: entrance in Living room, no Hallway | 0 | 100 |
| H04 | Bedroom at BFS distance 2 (max=1), 2 habitable rooms on facade | 100 | 66.7 |
| H05 | Only WC + Bathroom (no habitable rooms) | 100 | 100 |

Additional labeled apartments from Luyang's database go in the same directory. See `tests/fixtures/apartments/README.md` for the full test set specification.

---

## Implementation Status

| Phase | Module | Status |
|-------|--------|--------|
| 8 | `src/evaluation/apartment.py` — dataclasses + `load_rules()` | ✅ Done |
| 9 | Accuracy benchmark | Planned |
| 10 | Simple surrogate (decision tree) | Planned |
| 11 | `src/evaluation/daylight.py` — daylight check | Planned |
| 12 | `src/evaluation/circulation.py` — circulation check | Planned |
| 13 | `src/evaluation/composite.py` + GH integration | Planned |
