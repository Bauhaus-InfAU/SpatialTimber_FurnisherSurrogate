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

## Naming Convention

Notebooks and reports are prefixed by phase number: `{phase}-{seq}_{name}`. This groups artifacts by phase and sorts them naturally.

- **Notebooks:** `notebooks/{phase:02d}-{seq:02d}_{name}.ipynb` — e.g. `03-01_data_exploration.ipynb`, `03-02_umap_exploration.ipynb`
- **Reports:** `reports/{phase:02d}-{seq:02d}_{name}.{ext}` — e.g. `03-01_eda-findings.ipynb`, `04-01_rasterization-verification.html`
- **Plans:** `plans/{phase:02d}-{name}.md` — e.g. `plans/03-eda.md` (unchanged, already follows this pattern)

- **Tickets:** `tickets/{ID:02d}_{slug}.md` — e.g. `tickets/00_notebook-numbering.md`, `tickets/01_non-orthogonal-rooms.md`

When creating a new notebook, report, or ticket, use the next available sequence number.

## Status

Phases 1–6 complete. Phase 7 (Grasshopper) in progress (6/8 tasks). See `PLAN.md` for progress (42/46 tasks).

**Data pipeline**: `data.py` loads 8,322 apartments / 45,880 active rooms via `load_apartments()`. Frozen `Room`/`Apartment` dataclasses, SHA-256 integrity manifest, apartment-level stratified split (80/10/10). `features.py` extracts 14 features (5 numeric + 9 one-hot), pure numpy. No processed-data caching — JSONL re-parsed each call (~2-3 sec).

**EDA findings** (see `reports/03-01_eda-findings.ipynb`): bimodal scores (28.6% fail at 0, 41.6% score >=90), area is strongest predictor (r=+0.37), door position has zero linear signal, naive MAE=37.48, inter-room correlation near zero (r=0.006). Children rooms cap at ~76. Vertex count strongly predicts score (8-vertex median=37 vs 4-vertex median=92).

**Baseline model** (LightGBM): Test MAE=11.02 (71% improvement over naive 37.48), R²=0.80. Area dominant feature by gain. Kitchen (16.89) and Living room (18.84) hardest — spatial layout matters. Model saved at `models/baseline_lgbm.joblib`. W&B run: `infau/furnisher-surrogate/runs/3t4hiefb`.

**CNN model** (Phase 6): Three versions trained (v1→v2→v3), MAE improved 17.90→12.40→11.23 but never beat baseline (11.02). Key finding: spatial image features provide negligible value beyond tabular features. Each improvement came from strengthening tabular branch, not from better image understanding. LightGBM remains production model for Phase 7. Best checkpoint at `models/cnn_v3.pt`. W&B runs: v1 `3wcevehy`, v2 `qutd7leh`, v3 `ld6iz2h4`.

**Grasshopper integration** (Phase 7, in progress): `predict_score()` inference API created in `predict.py` — single function, handles rasterization + model loading + caching. Decoupled from sklearn via TYPE_CHECKING guard in `rasterize.py`. GhPython component is ~6 lines. Package has `[inference]` extra for lightweight install (numpy+Pillow+torch-cpu). 6 pytest tests passing. Remaining: `.gh` test file + end-to-end Rhino 8 verification.

## Reports

Reports from completed phases live in `reports/`. Check these before starting new phases — they contain data distribution boundaries, baselines, and known limitations. HTML exports are viewable via `htmlpreview.github.io` — use these preview links in Notion and plan files.

| Report | Phase | Contents | Preview |
|--------|-------|----------|---------|
| `reports/03-01_eda-findings.ipynb` | 3 (EDA) | Score distributions, feature correlations, failure analysis, data boundaries | [HTML](https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/03-01_eda-findings.html) |
| `reports/04-01_rasterization-verification.html` | 4 (Rasterization) | Visual verification, edge cases, fill ratio checks, dataset stats, UMAP | [HTML](https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/04-01_rasterization-verification.html) |
| `reports/06-01_cnn-model-comparison.ipynb` | 6 (CNN Model) | Architecture evolution v1→v3, baseline comparison, per-room-type analysis, negative result | [HTML](https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/06-01_cnn-model-comparison.html) |

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
- **HTML report links:** For HTML reports in `reports/`, use `htmlpreview.github.io` preview URLs instead of raw GitHub links. Pattern: `https://htmlpreview.github.io/?https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/reports/{filename}.html`. Use these in Notion task Deliverables tables and in plan file Outcome sections.

## Notebook Collaboration

When a Jupyter notebook is open in VS Code with a running kernel, Claude can execute code directly in the kernel via `mcp__ide__executeCode`. This means Claude can inspect variables, check shapes, and run follow-up analysis interactively — no need for save-and-read roundtrips. User runs cells normally; Claude reads/writes to the same kernel.

## Documentation Protocol

This project uses a strict "single source of truth" documentation strategy. When the user says **"document"** (or invokes `/document`), follow these rules:

### File roles — each fact lives in ONE place

| File / System | Contains | Update frequency |
|---------------|----------|-----------------|
| `README.md` | Project description, data format, setup instructions | At milestones only |
| `CLAUDE.md` (this file) | Current project state, conventions, key findings | End of each session |
| `PLAN.md` | Strategy, checkboxes, decisions with rationale | As work progresses |
| W&B | All experiment metrics, loss curves, model artifacts (use `wandb.summary` for scalars, `wandb.Table` for breakdowns, `commit=False` to batch) | Automatic during training |
| Notebooks | Self-contained analyses (EDA, training) | During analysis work |
| `reports/` | Phase findings reports (narrative notebooks + HTML) | At phase completion |
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
   - If a task's scope changed, update its page content using the **task description template** (see below)
   - All repo file references must be clickable GitHub links (see Notion linking convention above)
   - Phase plans (`plans/*.md`) are source of truth; Notion task status is derived

### Notion task description template

Every WP2 task page uses this structure:

```
## Goal
{One sentence: what problem this solves and why it matters.}

## Approach
{2-3 sentences: method and key technical decisions. For pending tasks, prefix with "Not started. Planned approach:"}

## Deliverables
| Type | Artifact |
|------|----------|
| {Report / Tool / Model / Dataset / Notebook / Component / Config} | {name + GitHub link} |

## Conclusions
{Bullet list of findings that matter beyond this task — things downstream tasks or other WPs need to know.
For pending tasks: state key open questions + carry forward relevant conclusions from predecessor tasks under "From [Phase] (inputs to this task):"}

## References
- **Plan:** {link to plans/*.md}
- **Depends on:** {Notion link to predecessor task(s)}
- **Feeds into:** {Notion link to successor task(s)}
```

**Deliverable types**: Report, Tool, Model, Dataset, Notebook, Component, Config
**Conclusions are forward-looking** — not a recap of what was done, but what the next person needs to know

7. **Tickets** (`tickets/*.md`) — Review open tickets:
   - Mark resolved tickets as `Status: resolved` if the issue was fixed during this session
   - If a ticket was addressed as part of a phase task, note which one
   - Do NOT create tickets during `/document` — tickets are created ad-hoc when issues are noticed

### Tickets (`tickets/`)

Lightweight backlog for features, bugs, and improvements noticed mid-session that should not interrupt current work. Template: `tickets/_TEMPLATE.md`.

- **Naming:** `tickets/{ID}_{slug}.md` — sequential zero-padded ID + kebab-case slug, e.g. `tickets/01_non-orthogonal-rooms.md`
- **When to create:** User says "ticket this", "note this for later", "park this", or you encounter a non-blocking issue during implementation
- **When NOT to create:** If the issue blocks current work — fix it now instead
- **Fields:** Type (feature/bug/improvement/tech-debt), Priority (low/medium/high), Status (open/in-progress/resolved), Phase link, Context, Description, Acceptance Criteria
- **Lifecycle:** open → in-progress → resolved. Resolved tickets stay in the folder (git history) but get marked

### What NOT to duplicate

- Experiment metrics → W&B only
- Analysis results → notebooks only
- Data format details → README.md only (this file just links to it)
- Code explanations → code comments only
