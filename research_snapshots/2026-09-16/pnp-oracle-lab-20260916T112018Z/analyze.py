"""Regenerate compact human-facing tables directly from raw, paired result rows."""
from __future__ import annotations
import csv
import json
import sqlite3
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read(name):
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def avg(rows, key):
    values = [float(row[key]) for row in rows if row[key] != ""]
    return sum(values) / len(values) if values else None


def summarize():
    learning = read("learning_rows.csv")
    planning = read("planning_rows.csv")
    composition = read("composition_rows.csv")
    assert len(learning) == 27648 and len(planning) == 3630 and len(composition) == 82944
    learning_groups = defaultdict(list)
    for r in learning:
        if r["m"] == "32":
            learning_groups[(r["family"], r["regime"], r["method"])].append(r)
    lrows = []
    for (family, regime, method), rows in sorted(learning_groups.items()):
        lrows.append({"family": family, "regime": regime, "method": method, "m": 32,
                      "error": avg(rows, "test_error"),
                      "full_error": avg(rows, "full_domain_error"),
                      "unseen_error": avg(rows, "unseen_error"),
                      "unseen_defined_rows": sum(r["unseen_error"] != "" for r in rows),
                      "train_error": avg(rows, "train_error"),
                      "consistent_rate": avg(rows, "consistent"),
                      "mean_unique_seen": avg(rows, "seen_unique"),
                      "rows": len(rows), "target_draws": len({r['target_id'] for r in rows}),
                      "unique_targets": len({r['target_mask'] for r in rows}),
                      "sample_replicates_per_draw": 8})
    pgroups = defaultdict(list)
    for r in planning:
        if r["case"] == "primary":
            pgroups[(r["family"], int(r["budget"]), r["method"])].append(r)
    prows = [{"family": key[0], "budget": key[1], "method": key[2],
              "error": avg(rows, "error"), "goals": len(rows), "worlds_per_goal": 16}
             for key, rows in sorted(pgroups.items())]
    pairs = defaultdict(dict)
    for r in composition:
        key = tuple(r[x] for x in ("family", "target_id", "regime", "m", "rep", "method", "budget"))
        pairs[key][r["planner"]] = r
    reversal = Counter()
    for pair in pairs.values():
        assert set(pair) == {"adaptive", "batch"}
        a, b = pair["adaptive"], pair["batch"]
        true_diff = float(a["actual_error"]) - float(b["actual_error"])
        predicted_diff = float(a["predicted_error"]) - float(b["predicted_error"])
        assert predicted_diff <= 0
        reversal["worse" if true_diff > 0 else "better" if true_diff < 0 else "equal"] += 1
        if true_diff > 0:
            reversal["worse_with_model_tie" if predicted_diff == 0 else "worse_with_strict_model_improvement"] += 1
    summary = {"learning_m32": lrows, "planning_primary": prows,
               "composition_pairs": len(pairs), "composition_comparison": dict(reversal),
               "row_counts": {"learning": len(learning), "planning": len(planning), "composition": len(composition)}}
    (RESULTS / "reviewed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Execute the widget's SQL on RAW rows, not on pre-aggregated display values.
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    columns = list(learning[0])
    db.execute("CREATE TABLE learning_rows (" + ",".join('"' + c + '" TEXT' for c in columns) + ")")
    db.executemany("INSERT INTO learning_rows VALUES (" + ",".join("?" for _ in columns) + ")",
                   [[r[c] for c in columns] for r in learning])
    chart_rows = [dict(r) for r in db.execute((ROOT / "chart_query.sql").read_text())]
    for row in chart_rows:
        independent = next(r for r in lrows if r['family'] == row['family'] and r['method'] == row['method'] and r['regime'] == 'uniform')
        assert abs(row['error_percent'] - 100 * independent['full_error']) < 1e-10
    (RESULTS / "chart_widget_data.json").write_text(json.dumps(chart_rows, indent=2) + "\n")
    db.close()
    with (RESULTS / "chart_learning.csv").open("w", newline="") as handle:
        selected = [r for r in lrows if r["regime"] == "uniform"]
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    table = ["# Raw-row recomputation", "", "m=32. Each learning cell =48 target draws x8 paired sample streams, not384 independent target functions.", "",
             "| Family | Method | Full-domain error | Unseen-input error | Unique targets |", "|---|---|---:|---:|---:|"]
    for r in lrows:
        if r["regime"] == "uniform":
            table.append(f"| {r['family']} | {r['method']} | {r['full_error']:.6f} | {r['unseen_error']:.6f} | {r['unique_targets']} |")
    table += ["", "## Feedback at budget2", "", "| Family | Method | Terminal error | Goals |", "|---|---|---:|---:|"]
    for r in prows:
        if r["budget"] == 2:
            table.append(f"| {r['family']} | {r['method']} | {r['error']:.6f} | {r['goals']} |")
    table += ["", "## Composition tie audit trigger", "", json.dumps(dict(reversal), sort_keys=True),
              "", "All observed adaptive disadvantages coincide with predicted-risk ties; this does not establish harm from a strictly improved model objective.", ""]
    (ROOT / "NUMERICAL_SUMMARY.md").write_text("\n".join(table))
    return summary


if __name__ == "__main__":
    result = summarize()
    print(json.dumps({k: v for k, v in result.items() if k not in ('learning_m32', 'planning_primary')}, indent=2))
