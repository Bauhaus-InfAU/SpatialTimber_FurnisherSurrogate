# Phase 12: Circulation Accessibility

## Goal

Check whether every room in an apartment is accessible from the apartment entrance via the hallway. Hallway IS a room type in the RL representation. Uses a room adjacency graph derived from door positions on shared walls.

**Why topological rules are sufficient:** Circulation accessibility is a graph reachability problem — no ML needed. The RL floor plan generator produces a pixel-grid layout (0.6m grid) where rooms are polygons and doors are points on walls. A simple graph traversal fully captures the design intent.

**Depends on:** Phase 8 — `ApartmentLayout`, `RoomLayout`, `WallSegment` dataclasses and test apartments must exist before implementation.

---

## Tasks

- [ ] 12.1 Implement room adjacency graph builder — door-in-shared-wall detection (door of room A lies on a wall edge of room B, within tolerance)
- [ ] 12.2 Implement entrance-hallway connection check — entrance point lies on a wall of the hallway room
- [ ] 12.3 Implement `circulation_score(apartment: ApartmentLayout) → dict` — per-room reachability + aggregate 0–100
- [ ] 12.4 Write pytest tests — ≥6 cases: fully connected, isolated room, no hallway, multiple hallways
- [ ] 12.5 Build verification notebook `notebooks/12-01_circulation_verification.ipynb`
- [ ] 12.6 Write report `reports/12-01_circulation.ipynb` + HTML; update PLAN.md + Notion

---

## Check Logic

Rules are loaded via `load_rules('circulation')` from `src/evaluation/rules/circulation.json`. No thresholds are hardcoded in Python.

**Step 1 — Entrance room detection:**
- The **entry room** is the room whose polygon boundary contains `apartment.entrance` within `entry_detection.tolerance_m = 0.05` m.
- If the apartment has a Hallway containing the entrance → Hallway is the entry room (distance 0).
- If no Hallway exists → whichever room contains the entrance is the entry room (BFS starts there). This supports studio layouts (entrance into Living room).

**Step 2 — Room adjacency graph:**
- Two rooms A and B are adjacent if: the door of A lies on a wall edge of B (within `adjacency.door_tolerance_m = 0.05` m), OR the door of B lies on a wall edge of A.
- Edge in graph: (room_A, room_B) — undirected.

**Step 3 — BFS distance scoring:**
- Run BFS from the entry room. Compute each room's BFS distance.
- For each room, look up its max allowed distance from `circulation.json → max_distance_from_entrance[room_type].max` (fallback: `max_distance_default.max = 1`).
- `room_score = 100` if `distance ≤ max`; `0` otherwise.

**Score output:**
```python
{
    "rooms": {
        "Hallway": 100,       # dist=0, max=0 → PASS
        "Living room": 100,   # dist=1, max=1 → PASS
        "Bedroom": 0,         # dist=2, max=1 → FAIL
        "Kitchen": 100,       # dist=1, max=1 → PASS
    },
    "total_rooms": 4,
    "rooms_passing": 3,
    "circulation_score": 75.0   # (3/4) × 100
}
```

**Tolerances** (from `circulation.json`):
- Entry detection: 0.05 m
- Door-on-wall adjacency: 0.05 m
- Both are larger than daylight tolerance (0.01 m) to handle small numerical errors from the RL generator.

---

## Edge Cases

| Case | Handling |
|------|---------|
| No Hallway room | Entry room = whichever room contains `apartment.entrance`. BFS starts from there. Rooms are scored by distance from that entry room vs. their type max. |
| Multiple Hallway rooms | Treat each Hallway as a graph node. BFS explores all of them. A Hallway at dist=0 passes (max=0); a Hallway at dist>0 fails (inverted layout). |
| Room with door but no shared wall with any other room | Isolated, BFS cannot reach it → dist = ∞ → score = 0. |
| Hallway not connected to entrance (wrong room contains entrance) | Hallway is a regular node in the graph. If entrance is in Living room, Living room is dist=0; Hallway may be dist=1, which exceeds max=0 → Hallway scores 0 (inverted layout). |

---

## Test Cases (≥6)

Use fixtures from `tests/fixtures/apartments/hand_crafted.json` plus Luyang's test set.

| ID | Description | Expected |
|----|-------------|---------|
| T1 (H01) | All rooms within distance threshold (Hallway + 2 Bedrooms) | circulation_score=100 |
| T2 | One room isolated (door not on any adjacent room wall) | circulation_score < 100 |
| T3 (H03) | No Hallway room; entrance in Living room; Bedroom at dist=1 | circulation_score=100 |
| T4 (H04) | Bedroom at BFS dist=2, max=1 | circulation_score=66.7 (2/3 pass) |
| T5 | Two Hallway rooms forming a corridor; both reachable | both Hallways scored by dist vs max=0; only the entry Hallway at dist=0 passes |
| T6 (H05) | Only WC + Bathroom; both within dist threshold | circulation_score=100 |

See also `SCORING.md` for expected scores on all H01–H05 hand-crafted cases.

---

## Deliverables

| Type | Artifact | Path |
|------|----------|------|
| Tool | Circulation evaluation function | `src/evaluation/circulation.py` |
| Notebook | Verification + graph visualization | `notebooks/12-01_circulation_verification.ipynb` |
| Report | Findings | `reports/12-01_circulation.ipynb` + `.html` |

---

## Decisions Log

- **Phase renamed 11 → 12** (2026-03-01): New Phase 8 (Floor Plan Representation) inserted. Renumbered accordingly.
- **Distance-based model replaces BFS reachability** (2026-03-02): Original plan used binary reachability from Hallway (reachable=100, not=0). Phase 8 implementation replaced this with a BFS-distance model: each room type has a maximum allowed distance from the entry room (configured in `circulation.json`). This rewards layouts where rooms are appropriately close to the entrance, not just connected. Hallway max=0, most rooms max=1, Children rooms max=2. Score = (rooms_within_threshold / total_rooms) × 100.
- **Entrance-room fallback for no-Hallway apartments** (2026-03-02): When no Hallway exists, BFS starts from the room containing `apartment.entrance` (e.g. Living room in a studio). This avoids blanket circulation_score=0 for valid studio layouts. Algorithm in `circulation.json → entry_detection.method = "entrance_point_on_wall"`.
- **Rules in JSON, not Python** (2026-03-02): All thresholds (max distances, tolerances) live in `src/evaluation/rules/circulation.json`. Implementation calls `load_rules('circulation')` at function invocation time. No hardcoded dicts in Python.

---

## Key Files

| File | Role |
|------|------|
| `src/evaluation/apartment.py` | Phase 8 deliverable — `ApartmentLayout`, `RoomLayout` |
| `src/evaluation/daylight.py` | Phase 11 deliverable — edge-overlap utility (reuse for door-on-wall detection) |
| `tests/fixtures/apartments/hand_crafted.json` | Phase 8 hand-crafted fixtures |
| `tests/test_daylight.py` | Test pattern to follow |
