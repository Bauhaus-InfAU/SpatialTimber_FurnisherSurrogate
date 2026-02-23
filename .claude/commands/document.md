---
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(git diff:*), Bash(git log:*), Bash(git status:*), mcp__claude_ai_Notion__notion-fetch, mcp__claude_ai_Notion__notion-update-page, mcp__claude_ai_Notion__notion-search
description: Update all project documentation (CLAUDE.md, PLAN.md, README.md, Notion WP2 + Tasks) and sync task progress across plan hierarchy
---

# Document Current Work

Follow the documentation protocol defined in CLAUDE.md. Update all relevant project documents based on the current session's work.

## Step 1: Gather Context

### Current documentation state
@CLAUDE.md
@PLAN.md
@README.md

### Phase plans — read ALL to check task status
@plans/01-setup.md
@plans/02-data-pipeline.md
@plans/03-eda.md
@plans/04-rasterization.md
@plans/05-baseline-model.md
@plans/06-cnn-model.md
@plans/07-grasshopper.md

### Recent changes
!`git status --porcelain`
!`git diff --name-only HEAD~5 2>/dev/null || echo "Not enough commits yet"`
!`git log --oneline -5 2>/dev/null || echo "No commits yet"`

## Step 2: Sync Task Lists (CRITICAL — do this FIRST)

For each phase plan (`plans/01-setup.md` through `plans/07-grasshopper.md`):

1. **Check completed work**: Based on git changes, existing files, and session context, determine which `- [ ]` tasks are actually done.
2. **Mark completed tasks**: Change `- [ ]` to `- [x]` in each phase plan.
3. **Add new tasks**: If work was done that isn't captured by an existing task, add a new `- [x]` entry.
4. **Remove stale tasks**: If a task is no longer relevant (approach changed), remove it or replace it.

Then **update the global progress table in PLAN.md**:
- Count `- [x]` and total `- [ ]` + `- [x]` in each phase plan
- Update the `Tasks` column (e.g., `3/5`)
- Update the `Status` column:
  - `pending` — 0 tasks done
  - `in progress` — some tasks done
  - `done` — all tasks done
- Update the **Total** row

### Task list consistency rules
- Every `- [ ]` or `- [x]` item in a phase plan MUST be counted in PLAN.md's progress table
- If a phase plan's task list changes (tasks added/removed/split), update PLAN.md totals immediately
- The PLAN.md progress table is the **derived** view — phase plans are the **source of truth**

## Step 3: Update CLAUDE.md

Update the **Status** section to reflect what is currently implemented and working. Add any new:
- Key findings that affect future development
- Conventions or patterns established
- Known gotchas discovered

Keep the file concise (under 50 lines of content, excluding the Documentation Protocol section which is permanent).

## Step 4: Update PLAN.md

Beyond the progress table (already handled in Step 2):
- Add brief decision notes in the relevant **phase plan's** Decisions Log where applicable
- Do NOT add experiment metrics (those belong in W&B)
- Do NOT add detailed analysis (those belong in notebooks)

## Step 5: Update README.md (only if needed)

Only touch README.md if:
- New dependencies were added that affect setup
- Project scope changed
- New setup steps are required

If none of these apply, skip this step entirely.

## Step 6: Sync Notion WP2 + Tasks

Notion IDs (from CLAUDE.md):
- **WP2 page:** `2f802b874c688070985bfa3f34938c50`
- **Tasks data source:** `collection://12d02b87-4c68-8126-a0dc-000bc9955625`
- **Repo base URL for links:** `https://github.com/Bauhaus-InfAU/SpatialTimber_FurnisherSurrogate/blob/main/`

### 6a. Sync WP2 project page
1. Fetch the WP2 page to see current content.
2. If scope, approach, or outcomes changed this session, update the relevant sections.
3. Update the Summary property if the project state description is stale.

### 6b. Sync Notion task statuses
Map each phase plan to its corresponding Notion task:

| Phase plan | Notion task page ID |
|------------|---------------------|
| `plans/01-setup.md` | `31002b874c68813a8f48d47c63b45864` |
| `plans/02-data-pipeline.md` | `31002b874c688174adbdf35944952594` |
| `plans/03-eda.md` | `31002b874c688192b4a3ede2fa45763f` |
| `plans/04-rasterization.md` | `31002b874c6881689d8fda7a931dc231` |
| `plans/05-baseline-model.md` | `31002b874c6881049426f76a1ce1c73b` |
| `plans/06-cnn-model.md` | `31002b874c68811d8346d349620e5078` |
| `plans/07-grasshopper.md` | `31002b874c6881048d81f61ae8dfb538` |

For each task:
1. Derive status from the phase plan's checkbox progress:
   - 0 tasks done → **Not Started**
   - Some tasks done → **In Progress** (note: may need to be set manually if the status option doesn't exist yet)
   - All tasks done → **Done**
2. Update the Notion task's `Status` property if it differs from current.
3. If the task's scope or description changed, update its page content. All repo file references must be full GitHub links.

**Phase plans are source of truth** — Notion task status is derived from them, not the other way around.

## Step 7: Summary

After updating, provide a brief summary:
- Which tasks were marked complete (list them)
- Updated progress numbers per phase
- Overall project progress (X/Y tasks complete)
- Which files were modified
- Which Notion pages were updated
