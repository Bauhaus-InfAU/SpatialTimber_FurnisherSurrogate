# Apartment Test Set — Specification for Luyang

This folder contains floor plan fixtures for testing the WP2 reward functions (daylight, circulation, furnishability). It currently holds 5 hand-crafted developer fixtures (`H01`–`H05`). Luyang's contribution is ~25 labeled apartments from the cleaned floor plan database, covering the edge cases listed below.

## Format

Each apartment is one JSON object conforming to [`schema.json`](schema.json). All apartments are collected in a single JSON array file (e.g. `luyang_test_set.json`).

```json
{
  "id": "D01",
  "apartment_type": "2-Bedroom",
  "entrance": [4.0, 0.0],
  "outer_polygon": [[0,0], [8,0], [8,6], [0,6], [0,0]],
  "outer_is_exterior": [true, true, true, true],
  "doors": [[2.0, 2.0], [6.0, 2.0]],
  "rooms": [
    {
      "room_type": "Hallway",
      "polygon": [[0,0], [8,0], [8,2], [0,2], [0,0]]
    },
    {
      "room_type": "Bedroom",
      "polygon": [[0,2], [4,2], [4,6], [0,6], [0,2]]
    }
  ],
  "expected_scores": {
    "daylight": 100,
    "circulation": 100,
    "furnitability": null
  }
}
```

**Coordinate conventions:**
- All units in **meters**
- Polygon winding: **any direction** — winding order is not enforced; algorithms use orientation-agnostic methods
- Polygons are **closed**: first point == last point

**`outer_is_exterior`:** one bool per outer polygon edge (`outer_polygon[i]` → `outer_polygon[i+1]`).
- `true` — edge faces outside (facade, gable)
- `false` — party wall (shared with neighbouring apartment)

**`doors`:** flat list of interior door opening points. Each point lies on the shared wall between two rooms. Two rooms are adjacent (for BFS circulation) if any door point lies on both their polygon boundaries. A Hallway with 3 neighbours has 3 door points in this list. Rooms have no `door` field — use the apartment-level `doors` list instead.

**`expected_scores.furnitability`:** always `null` — computed by the surrogate model, never manually labeled.

---

## Cases Needed

We need **~25 apartments** covering:

### Daylight (10 apartments)

| ID   | Description | Expected `daylight` |
|------|-------------|---------------------|
| D01  | All habitable rooms on facade | 100 |
| D02  | No habitable rooms on facade | 0 |
| D03  | 1 of 3 habitable rooms on facade | 33.3 |
| D04  | Only WC + Bathroom (no habitable rooms) | 100 |
| D05  | Bedroom polygon touches facade corner only — NOT a full shared edge | 0 |
| D06–D10 | Typical 1–5 bedroom apartments, natural facade distribution | varies |

**Habitable rooms** (must have exterior wall access): Bedroom, Living room, Kitchen, Children 1–4.
**Non-habitable** (ignored in aggregate): Bathroom, WC, Hallway.

### Circulation (10 apartments)

| ID   | Description | Expected `circulation` |
|------|-------------|------------------------|
| C01  | All rooms connected via Hallway, all within distance threshold | 100 |
| C02  | One Bedroom isolated from Hallway (no shared door) | < 100 |
| C03  | No Hallway room; entrance opens into Living room; all other rooms reachable from Living room within threshold | 100 |
| C04  | Bedroom at BFS distance 2 from entrance (max for Bedroom = 1) | < 100 |
| C05  | Two Hallway segments connected via door | 100 |
| C06–C10 | Typical 1–5 bedroom apartments, fully connected within distance thresholds | 100 |

**Circulation algorithm:** BFS from the entrance room (the room whose polygon contains `entrance` within 5 cm). A room passes if its BFS distance ≤ the type's max distance from [`circulation.json`](../../../src/evaluation/rules/circulation.json).

### Labeling notes

- `entrance` must be a point on the `outer_polygon` boundary (within 5 cm tolerance).
- Each point in `doors` must lie on the shared wall between exactly two room polygons.
- When uncertain about the `expected_scores`, set to `null` — uncertain fixtures are still useful for smoke-testing data loading.
- For `expected_scores.circulation` on partial cases (some rooms fail): compute `(rooms_passing / total_rooms) * 100`.

---

## Existing Hand-Crafted Fixtures (`hand_crafted.json`)

| ID  | Description | daylight | circulation | Notes |
|-----|-------------|----------|-------------|-------|
| H01 | 2-bed, 2 rooms on facade, Hallway at entrance | 100 | 100 | All rooms at BFS dist ≤ max |
| H02 | 2-bed, no rooms on facade, Hallway at entrance | 0 | 100 | Interior bedrooms |
| H03 | Studio: entrance in Living room, no Hallway | 0 | 100 | Living room at dist=0; Bedroom at dist=1 |
| H04 | Bedroom at BFS distance 2 (max=1), 2 habitable rooms on facade | 100 | 66.7 | Hallway+Living room pass, Bedroom fails |
| H05 | Only WC + Bathroom (no habitable rooms) | 100 | 100 | Vacuous daylight pass |

---

## Validation

All fixtures must validate against `schema.json`. Use:

```bash
uv run python -c "
import json, jsonschema, pathlib
schema = json.loads(pathlib.Path('tests/fixtures/apartments/schema.json').read_text())
data = json.loads(pathlib.Path('tests/fixtures/apartments/hand_crafted.json').read_text())
for apt in data:
    jsonschema.validate(apt, schema)
print(f'All {len(data)} fixtures valid.')
"
```
