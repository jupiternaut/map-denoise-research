# Commands

Run from `/home/grf/Documents/Codex/2026-09-16/pnp-oracle-lab-20260916T112018Z`.

## Recheck existing frozen evidence

```bash
python3 -m unittest discover -v
python3 audit_checks.py
```

The audit command refreshes its own receipt, not primary data. Source changes,
timings and receipts require a new manifest if making a new frozen release.

## Rerun primary computations into a NEW output directory

```bash
experiment_replay_dir=$(mktemp -d /tmp/pnp-oracle-replay.XXXXXX)
python3 learning.py --output-dir "$experiment_replay_dir"
python3 planning.py --output "$experiment_replay_dir"
python3 composition.py --input "$experiment_replay_dir/learning_rows.csv" --output "$experiment_replay_dir"
python3 tie_audit.py --source "$experiment_replay_dir/composition_rows.csv" --output "$experiment_replay_dir"
```

Random seeds and tie rules are fixed in the modules. Predictions, errors, query
policies and support counts reproduce; wall-time fields are expected to differ.
The original `results/` is not overwritten by these replay commands.

## Regenerate display summaries from this workspace's primary results

```bash
python3 analyze.py
```

Outputs reviewed_summary.json, chart_widget_data.json, chart_learning.csv and
NUMERICAL_SUMMARY.md. It executes chart_query.sql over an in-memory SQLite table
populated from the raw CSV and verifies the means against separate Python grouping.

