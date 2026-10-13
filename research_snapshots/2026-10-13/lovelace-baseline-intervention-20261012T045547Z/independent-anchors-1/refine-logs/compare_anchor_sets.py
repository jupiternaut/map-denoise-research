"""Read-only set comparisons after the accepted A8 batch; no solver calls."""
from collections import Counter
from datetime import datetime, timezone, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import socket

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/anchor-v1"


def load(path):
    return json.loads(path.read_text())


def in_group(row, group):
    wrong = Fraction(row["initial_ae_mm"]) != 0
    return (group == "all" or group == "shifted" and wrong
            or group == "primary" and wrong and row["variation_index"] == 1
            and row["truth_mm"] == "9007/14" and row["case"] != "equal_full_negative")


def main():
    if socket.gethostname() != "liekkas":
        raise RuntimeError("wrong host")
    if load(OUT / "STAGE_STATUS.json")["status"] != "COMPLETE":
        raise RuntimeError("formal batch incomplete")
    if load(OUT / "verification/VERIFY.json")["status"] != "PASS":
        raise RuntimeError("independent mechanical verification did not pass")
    rows = load(OUT / "results.json")
    arms = ("none", "camera", "appearance", "both")
    paired = {(r["task_id"], r["arm"]): r for r in rows}
    sets = {}
    for group in ("all", "shifted", "primary"):
        task_ids = {r["task_id"] for r in rows if r["arm"] == "none" and in_group(r, group)}
        cert = {arm: {tid for tid in task_ids if paired[tid, arm]["status"] == "CERTIFIED_INFEASIBLE"}
                for arm in arms}
        union = cert["camera"] | cert["appearance"]
        joint_only = cert["both"] - union
        prior_forward = {tid for tid in task_ids if paired[tid, "none"]["a7_status"] == "FORWARD_WITNESS"}
        patterns = Counter("/".join("1" if tid in cert[arm] else "0" for arm in arms)
                           for tid in task_ids)
        sets[group] = {
            "n": len(task_ids), "arm_certificate_counts": {arm: len(cert[arm]) for arm in arms},
            "camera_appearance_intersection": len(cert["camera"] & cert["appearance"]),
            "camera_appearance_union": len(union), "joint_only_count": len(joint_only),
            "joint_only_tasks": sorted(joint_only),
            "single_arm_certificates_not_retained_by_both": sorted(union - cert["both"]),
            "joint_only_prior_forward_count": len(joint_only & prior_forward),
            "certificate_patterns_none_camera_appearance_both": dict(sorted(patterns.items()))}
    primary_remaining = []
    for r in rows:
        if r["arm"] == "both" and in_group(r, "primary") and r["status"] != "CERTIFIED_INFEASIBLE":
            primary_remaining.append({key: r[key] for key in
                                      ("task_id", "observation_id", "case", "truth_mm", "incumbent_mm",
                                       "initial_ae_mm", "status", "a7_status")})
    result = {"stage": "A8", "analysis_kind": "post-verification paired certificate-set comparison",
              "results_sha256": hashlib.sha256((OUT / "results.json").read_bytes()).hexdigest(),
              "sets": sets, "primary_both_remaining": primary_remaining,
              "interpretation": "Joint-only means an exact exclusion found only by the combined arm; it is not a completeness theorem for either single-arm certifier."}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    for path in (ROOT / ("PAIRED_SETS_" + stamp + ".json"), ROOT / "PAIRED_SETS.json"):
        with path.open("x") as stream:
            stream.write(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
