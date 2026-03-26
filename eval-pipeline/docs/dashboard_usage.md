# Static Dashboard Usage Guide

This guide explains how to generate and view the static Chart.js dashboard summarizing task & sprint metrics.

## Inputs
- `eval-pipeline/reports/task_timeline.json` (flat task list with status / engineer / target_sprint)
- `eval-pipeline/reports/sprint_summary.json` (aggregated metrics produced by the extraction script)

Regenerate these first if needed (example):
```bash
python3 eval-pipeline/scripts/extract_task_timeline.py \
  --input eval-pipeline/docs/tasks/tasks.md \
  --input-zh eval-pipeline/docs/tasks/tasks.zh.md \
  --inherit-subtasks \
  --sprint-summary-json \
  --no-timestamp
```
Outputs appear in `eval-pipeline/reports/`.

## Generate Dashboard
```bash
python3 eval-pipeline/scripts/generate_dashboard.py \
  --task-json eval-pipeline/reports/task_timeline.json \
  --sprint-json eval-pipeline/reports/sprint_summary.json \
  --out-file eval-pipeline/reports/dashboard.html
```
If arguments omitted, defaults match the above.

## Open Dashboard
Just open `eval-pipeline/reports/dashboard.html` in a browser (no server required). All charts are rendered client-side via CDN Chart.js.

## Charts Overview
Row 1:
- Burndown (Remaining Main Tasks per Sprint)
- Cumulative Done (Main)

Row 2:
- Main vs Sub Tasks per Sprint (stacked)
- Engineer Distribution per Sprint (stacked)

Row 3:
- Status Distribution (all tasks snapshot)
- Completion % per Sprint

KPI Cards:
- Average Velocity (mean of non-zero `done_main` values)
- Forecast Remaining Sprints (Remaining Main / Average Velocity; N/A if velocity=0)
- Remaining Main Tasks (from last sprint record)
- Total Main Tasks (overall)

## Updating Velocity
Currently all tasks are `Planned`; to enable velocity:
1. Update task governance blocks in `tasks.md` / `tasks.zh.md` to set status (`In-Progress`, `Done`).
2. Re-run extraction script with `--inherit-subtasks --sprint-summary-json`.
3. Re-generate dashboard; velocity & completion metrics will reflect progress.

## Extensibility Ideas
- Add historical snapshots (date dimension) for true time-based burndown (store daily copies of `sprint_summary.json`).
- Add per-status burn-up stacked area chart.
- Introduce risk tagging & filter UI (lightweight: add checkboxes + JS hide/show datasets).
- Add parent-child relationship indicator in flat JSON for hierarchical visualization.
- Export PNG via `chart.toBase64Image()` and save automation script.

## Minimal Modification Points
- Add new charts: edit `generate_dashboard.py` -> modify `build_html` data assembly & JS block.
- Theme tweak: adjust CSS in `<style>` section.
- Additional metrics: compute in `compute_velocity_and_forecast` or derive new function.

## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| All completion 0% | No tasks have status `Done` | Update statuses + re-run extraction |
| Velocity = 0 | No non-zero `done_main` counts | Mark some main tasks `Done` |
| Missing JSON file | Extraction not run | Run extraction script first |
| Garbled engineer names | Inconsistent engineer labels in tasks | Normalize labels in governance blocks |

## License / Notice
This dashboard is static; no external data is sent. Chart.js loaded via CDN (replace with pinned version locally if air-gapped needed).

---
Generated: $(date) (update manually if distributing)
