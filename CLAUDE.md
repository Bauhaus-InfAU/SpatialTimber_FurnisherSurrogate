# CLAUDE.md

## Project

FurnisherSurrogate — a surrogate model to approximate slow procedural furniture placement scores for use in RL training.

## Data

Training data: `../SpatialTimber_DesignExplorer/Furnisher/Apartment Quality Evaluation/apartments.jsonl` (~8k apartments, 46k active rooms, JSONL format).

## Domain Concepts

- **Room types:** Bedroom, Living room, Bathroom, WC, Kitchen, Children 1–4
- **Score (0–100):** furniture placement quality — 90+ excellent, 70–89 good, 40–69 problematic, <40 poor, 0 failed, null = room absent
- **Polygon format:** closed polyline in meters, axis-aligned, counter-clockwise winding order
- **Door:** position as a point on the room's wall

## Key Constraint

The surrogate predicts **per-room** scores, not per-apartment. Each room is an independent prediction.

## Status

Phase 1 (Setup) in progress — project directories created, `__init__.py` in place, `.gitignore` configured. Remaining: `pyproject.toml`, PyTorch CUDA, W&B login, GPU verify. See `PLAN.md` for progress (2/38 tasks).

## Documentation Protocol

This project uses a strict "single source of truth" documentation strategy. When the user says **"document"** (or invokes `/document`), follow these rules:

### File roles — each fact lives in ONE place

| File | Contains | Update frequency |
|------|----------|-----------------|
| `README.md` | Project description, data format, setup instructions | At milestones only |
| `CLAUDE.md` (this file) | Current project state, conventions, key findings | End of each session |
| `PLAN.md` | Strategy, checkboxes, decisions with rationale | As work progresses |
| W&B | All experiment metrics, loss curves, model artifacts | Automatic during training |
| Notebooks | Self-contained analyses (EDA, training) | During analysis work |

### What to update when documenting

1. **CLAUDE.md** — Update the Status section and add any new findings/conventions:
   - What was implemented or changed this session
   - Key findings that affect future work
   - New conventions or gotchas discovered
   - Keep total file under ~50 lines

2. **Phase plans** (`plans/*.md`) — Update task checkboxes:
   - Mark `- [ ]` → `- [x]` for completed tasks
   - Add/remove tasks if scope changed
   - Add brief decision notes in Decisions Log sections

3. **PLAN.md** — Sync the global progress table:
   - Count done/total from each phase plan's `## Tasks` section
   - Update `Tasks` column (e.g., `3/5`), `Status` column (`pending` / `in progress` / `done`)
   - Update the `Total` row
   - Phase plans are source of truth; PLAN.md progress table is derived

4. **README.md** — Only update if project scope or setup changed:
   - New dependencies added
   - New setup steps required
   - Project description evolved

### What NOT to duplicate

- Experiment metrics → W&B only
- Analysis results → notebooks only
- Data format details → README.md only (this file just links to it)
- Code explanations → code comments only
