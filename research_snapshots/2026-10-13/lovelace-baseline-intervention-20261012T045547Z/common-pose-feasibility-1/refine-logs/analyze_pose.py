"""Read-only descriptive aggregation of completed A7 receipts; no scientific rerun."""
import json, hashlib
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pose-v1"
def load(p):
    return json.loads(p.read_text())
def number(x):
    if isinstance(x, dict): return Q(x["numerator"], x["denominator"])
    return Q(x)
def count(xs):
    return dict(sorted(Counter(xs).items()))
def zero(w):
    return all(number(x) == 0 for x in w["offsets"])
def group(rows, index):
    records = [index[r["task_id"]] for r in rows]
    selected = [r for r in records if r["status"] == "FORWARD_WITNESS"]
    searches = [c["search"] for r in records for c in r["branches"] if c["search"]]
    witnesses = [(r, [c["witness"] for c in r["branches"] if c["witness"]]) for r in selected]
    return {
        "n": len(rows),
        "a6_status": count(r["a6_status"] for r in rows),
        "status": count(r["status"] for r in rows),
        "nominal_status": count(r["nominal_status"] for r in rows),
        "seed_status": count(r["seed_status"] for r in rows),
        "search_only_task_forward": sum(any(c["search"] and c["search"]["status"] == "FORWARD_WITNESS" for c in r["branches"]) for r in records),
        "additional_verified_over_nominal": sum(r["status"] == "FORWARD_WITNESS" and r["nominal_status"] != "FORWARD_WITNESS" for r in rows),
        "any_selected_zero_pose": sum(any(zero(w) for w in ws) for r, ws in witnesses),
        "all_selected_nonzero_pose": sum(all(not zero(w) for w in ws) for r, ws in witnesses),
        "branch_status": count(c["status"] for r in records for c in r["branches"]),
        "seed_branch_status": count(c["seed"]["status"] if c["seed"] else "INHERITED_INFEASIBLE" for r in records for c in r["branches"]),
        "search_branch_status": count(s["status"] for s in searches),
        "search_termination": count(s["termination"] for s in searches),
        "fixed_seed_only_task_forward": sum(r["status"] == "FORWARD_WITNESS" and not any(c["search"] and c["search"]["status"] == "FORWARD_WITNESS" for c in r["branches"]) for r in records),
        "max_depth_gap_mm_with_forward": str(max((Q(r["initial_ae_mm"]) for r in rows if r["status"] == "FORWARD_WITNESS"), default=Q(0)))
    }
def main():
    stage, verify = load(OUT / "STAGE_STATUS.json"), load(ROOT / "diagnostic-audit/SAVED_RECORD_AUDIT.json")
    assert stage["status"] == "INCOMPLETE" and verify["status"] == "SAVED_RECORD_NUMERICAL_PASS"
    rows, records = load(OUT / "results.json"), load(OUT / "diagnoses.json")
    index = {r["task_id"]: r for r in records}
    assert len(rows) == len(records) == len(index) == 324
    assert sum(len(r["branches"]) for r in records) == 972
    groups = {
        "all": rows,
        "correct": [r for r in rows if Q(r["initial_ae_mm"]) == 0],
        "shifted": [r for r in rows if Q(r["initial_ae_mm"]) != 0],
        "primary40": [r for r in rows if r["variation_index"] == 1 and r["truth_mm"] == "9007/14" and r["case"] != "equal_full_negative" and Q(r["initial_ae_mm"]) != 0],
    }
    assert len(groups["primary40"]) == 40
    for eta in (0, 1):
        for label, subset in list(groups.items())[:3]:
            groups[f"eta{eta}_{label}"] = [r for r in subset if r["variation_index"] == eta]
        for label in ("informative_shifted", "negative_shifted"):
            groups[f"eta{eta}_{label}"] = [r for r in groups[f"eta{eta}_shifted"] if (r["case"] == "equal_full_negative") == (label == "negative_shifted")]
        for depth in sorted({r["truth_mm"] for r in rows}):
            groups[f"eta{eta}_depth{depth}_informative_shifted"] = [r for r in groups[f"eta{eta}_informative_shifted"] if r["truth_mm"] == depth]
    searches = [c["search"] for r in records for c in r["branches"] if c["search"]]
    cost = {key: sum(sum(s[key]) if isinstance(s[key],list) else s[key] for s in searches) for key in ("numeric_evaluations", "classifier_lp_calls", "numeric_lp_calls", "exact_attempts", "numeric_failures", "powell_starts", "powell_evaluations", "seconds")}
    cost["searches"] = len(searches)
    cost["maxima"] = {key: max(max(s[key],default=0) if isinstance(s[key],list) else s[key] for s in searches) for key in ("numeric_lp_calls", "exact_attempts", "powell_starts", "powell_evaluations", "seconds")}
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/"results.json", OUT/"diagnoses.json", OUT/"summary.json", OUT/"STAGE_STATUS.json", ROOT/"diagnostic-audit/SAVED_RECORD_AUDIT.json")}
    result = {
        "scope": "saved-record diagnostic with numerical audit only; original formal stage INCOMPLETE; not independent-scene inference",
        "inputs": inputs, "stage": stage, "groups": {k: group(v,index) for k,v in groups.items()},
        "cost": cost, "coverage": {"actual_forward": len(verify["actual_forward_coverage"]), "correct_status": count(r["correct_input_status"] for r in verify["actual_forward_coverage"])},
        "primary_task_table": [{**r, "forward_branches":[c["branch"] for c in index[r["task_id"]]["branches"] if c["witness"]]} for r in groups["primary40"]],
        "unknown_tasks": [r["task_id"] for r in rows if r["status"] == "UNKNOWN"],
        "new_exclusions": 0,
        "pair_bound": {"conditional_statement": "Within the frozen image-and-parameter-bound model, two depths reproducing the same public images imply max absolute error at least half their separation for any common deterministic estimate; this is not a real-scene lower bound.",
                       "max_verified_pair_separation_mm": str(max(Q(r["initial_ae_mm"]) for r in rows if r["status"] == "FORWARD_WITNESS")),
                       "max_pair_minimax_absolute_error_mm": str(max(Q(r["initial_ae_mm"]) for r in rows if r["status"] == "FORWARD_WITNESS") / 2)}
    }
    print(json.dumps(result, indent=2, allow_nan=False))
if __name__ == "__main__":
    main()
