## Task Timeline Extraction (PM Monitoring)

Script: `scripts/extract_task_timeline.py`

Generates a snapshot report of all tasks with their current status, engineer (owner), and target sprint by parsing `docs/tasks/tasks.md` (and optionally the Chinese variant).

### Run Manually
```
python3 scripts/extract_task_timeline.py \
  --input docs/tasks/tasks.md \
  --input-zh docs/tasks/tasks.zh.md \
  --out-dir reports \
  --output-format markdown csv
```
Outputs (timestamped by default) like:
```
reports/
  task_timeline_20250911_093000.md
  task_timeline_20250911_093000.csv
```

### Fields
| Field         | Description                                         |
|---------------|-----------------------------------------------------|
| ID            | Task identifier (e.g., TASK-034b)                   |
| Title         | Parsed from primary table row                       |
| Status        | Governance block `status` (blank if missing)        |
| Engineer      | Governance block `engineer` (blank if missing)      |
| Target Sprint | Governance block `target_sprint` (blank if missing) |

### Schedule (Cron Example)
Add to crontab (daily 08:00 UTC):
```
0 8 * * * cd /opt/domain-specific-llm-eval/eval-pipeline && /usr/bin/python3 scripts/extract_task_timeline.py --input docs/tasks/tasks.md --out-dir reports --output-format markdown csv >> logs/task_timeline_cron.log 2>&1
```

### CI Integration (Optional)
You can add a GitHub Actions workflow step to run the extractor and upload the artifacts for PM consumption.

### Notes
- Missing governance fields remain blank to highlight gaps.
- Use `--no-timestamp` to overwrite a stable filename for dashboards.
- The parser is regex-based; keep the task ID column format `| TASK-### |` for reliable extraction.