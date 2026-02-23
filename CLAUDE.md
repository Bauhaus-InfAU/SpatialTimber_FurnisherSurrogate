# CLAUDE.md

## Project

FurnisherSurrogate — a surrogate model to approximate slow procedural furniture placement scores for use in RL training.

**Repo:** `https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate` (**PUBLIC** — never commit secrets, keys, or credentials)

**Secrets:** Store API keys in `.env` (gitignored). Never hardcode keys in code or tracked files.

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

Phase 1 (Setup) complete — PyTorch 2.10.0+cu128, RTX 4060 8GB verified, W&B logged in (`infau`), hatchling editable install. Phase 2 (Data Pipeline) is next. See `PLAN.md` for progress (6/38 tasks).

## Notion

Workspace: **Spatial Timber** | Hub page: `12d02b874c6880269a34eca3dd867edf`

| Database | Data source ID |
|----------|---------------|
| Projects | `collection://12d02b87-4c68-81da-9612-000bebce533d` |
| Tasks | `collection://12d02b87-4c68-8126-a0dc-000bc9955625` |
| Sprints | `collection://2fc02b87-4c68-8029-a5b5-000bfc4f15a2` |

- **WP2 page:** `2f802b874c688070985bfa3f34938c50`
- **Martin (user):** `user://4e65cb83-7da9-47b5-aa9b-76a0c47a4b48`
- Use Notion MCP tools to read/update. Tasks are created in the Tasks data source with `Project` relation pointing to WP2.
- **Linking convention:** When referencing repo files in Notion (task descriptions, project pages), always use full GitHub URLs so readers can click through — e.g. `[plans/03-eda.md](https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/plans/03-eda.md)`, not bare backtick paths.

## Documentation Protocol

This project uses a strict "single source of truth" documentation strategy. When the user says **"document"** (or invokes `/document`), follow these rules:

### File roles — each fact lives in ONE place

| File / System | Contains | Update frequency |
|---------------|----------|-----------------|
| `README.md` | Project description, data format, setup instructions | At milestones only |
| `CLAUDE.md` (this file) | Current project state, conventions, key findings | End of each session |
| `PLAN.md` | Strategy, checkboxes, decisions with rationale | As work progresses |
| W&B | All experiment metrics, loss curves, model artifacts | Automatic during training |
| Notebooks | Self-contained analyses (EDA, training) | During analysis work |
| Notion (WP2 + Tasks) | Project scope, high-level task status & descriptions | When documenting |
| `tickets/*.md` | Deferred features, bugs, improvements — backlog parking lot | As noticed |

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

5. **Notion WP2 project page** (`2f802b874c688070985bfa3f34938c50`) — Sync with repo state:
   - Update the Approach / Outcome sections if scope or strategy changed
   - Update the Summary property to reflect current project state
   - Keep content concise — Notion is for team-facing overview, not implementation detail

6. **Notion Tasks** (in Tasks data source) — Sync status with `plans/*.md`:
   - Fetch each WP2 task; update `Status` property to match phase progress (Not Started → In Progress → Done)
   - If a task's scope changed, update its page content (description + links)
   - All repo file references must be clickable GitHub links (see Notion linking convention above)
   - Phase plans (`plans/*.md`) are source of truth; Notion task status is derived

7. **Tickets** (`tickets/*.md`) — Review open tickets:
   - Mark resolved tickets as `Status: resolved` if the issue was fixed during this session
   - If a ticket was addressed as part of a phase task, note which one
   - Do NOT create tickets during `/document` — tickets are created ad-hoc when issues are noticed

### Tickets (`tickets/`)

Lightweight backlog for features, bugs, and improvements noticed mid-session that should not interrupt current work. Template: `tickets/_TEMPLATE.md`.

- **Naming:** `tickets/{slug}.md` — short kebab-case slug, e.g. `tickets/non-orthogonal-rooms.md`
- **When to create:** User says "ticket this", "note this for later", "park this", or you encounter a non-blocking issue during implementation
- **When NOT to create:** If the issue blocks current work — fix it now instead
- **Fields:** Type (feature/bug/improvement/tech-debt), Priority (low/medium/high), Status (open/in-progress/resolved), Phase link, Context, Description, Acceptance Criteria
- **Lifecycle:** open → in-progress → resolved. Resolved tickets stay in the folder (git history) but get marked

### What NOT to duplicate

- Experiment metrics → W&B only
- Analysis results → notebooks only
- Data format details → README.md only (this file just links to it)
- Code explanations → code comments only
