"""Seal only the completed A9 directory; verify inherited seals read-only."""
import hashlib
import json
import socket
from datetime import datetime
from fractions import Fraction as F
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "continuous-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def check_files(data):
    for name, digest in data["files"].items():
        assert sha(Path(name)) == digest, name
    return len(data["files"])


def main():
    assert socket.gethostname() == "liekkas"
    assert str(ROOT) == "/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/continuous-anchors-1"
    assert not (ROOT / "FINAL_SEAL.json").exists(), "already sealed"
    assert not (ROOT / "ARTIFACTS.json").exists(), "index already exists"
    assert load(OUT / "STAGE_STATUS.json")["status"] == "COMPLETE"
    assert load(OUT / "verification" / "VERIFY.json")["status"] == "PASS"
    assert load(ROOT / ".aris" / "REVIEW_GATE.json")["status"] == "PASS"
    nsource = check_files(load(OUT / "SOURCE_LOCK.json"))
    ninput = check_files(load(OUT / "INPUT_PRELOCK.json"))
    check_files(load(ROOT / ".aris" / "REVIEW_GATE.json"))
    old = ROOT.parent / "independent-anchors-1"
    old_seal = load(old / "FINAL_SEAL.json")
    assert sha(old / "ARTIFACTS.json") == old_seal["artifact_index_sha256"]
    old_index = load(old / "ARTIFACTS.json")
    for name, digest in old_index["files"].items():
        assert sha(old / name) == digest, name
    for name, version in [("REPORT.md", "REPORT_20261013_103412.md"),
                          ("CHECKPOINT.md", "CHECKPOINT_20261013_103412.md"),
                          ("ANALYSIS.json", "ANALYSIS_20261013_103705.json")]:
        assert (ROOT / name).read_bytes() == (ROOT / version).read_bytes(), name
    a = load(ROOT / "ANALYSIS.json")
    b = a["metric_groups"]["both"]["shifted"]
    assert (b["n"], b["improved"], b["equal"], b["worsened"]) == (288, 122, 166, 0)
    assert F(b["final_MAE_mm"]) == F(4561, 4608)
    assert a["paired_gain_counts"]["both"] == {"improved": 15, "equal": 3, "worsened": 0}
    assert a["joint_task_set"]["joint_only"] == 29
    listed = {line.split("|")[3].strip() for line in (ROOT / "MANIFEST.md").read_text().splitlines()
              if line.startswith("| 2026-")}
    files = sorted(p for p in ROOT.rglob("*") if p.is_file())
    expected = {p.relative_to(ROOT).as_posix() for p in files} | {"ARTIFACTS.json", "FINAL_SEAL.json"}
    assert expected <= listed, sorted(expected-listed)
    index = {"stage": "A9", "host": "liekkas", "root": str(ROOT), "count": len(files),
             "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in files}}
    (ROOT / "ARTIFACTS.json").write_text(json.dumps(index, ensure_ascii=False, indent=2)+"\n")
    seal = {"stage": "A9", "host": "liekkas", "root": str(ROOT),
            "formal_status": "COMPLETE", "verification": "PASS", "arithmetic_review": "PASS",
            "artifact_index_sha256": sha(ROOT / "ARTIFACTS.json"),
            "indexed_files": len(files), "total_files_including_self": len(files)+2,
            "source_lock_sha256": sha(OUT / "SOURCE_LOCK.json"),
            "report_sha256": sha(ROOT / "REPORT.md"), "analysis_sha256": sha(ROOT / "ANALYSIS.json"),
            "verified_locked_sources": nsource, "verified_prelocked_inputs": ninput,
            "verified_old_A8_artifacts": len(old_index["files"]),
            "superseded_post_analysis": "ANALYSIS_20261013_103412.json paired_gain_counts only; see ASSURANCE.md",
            "assurance": "same-family provisional; finite synthetic family",
            "sealed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    (ROOT / "FINAL_SEAL.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2)+"\n")
    for name, digest in index["files"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / "ARTIFACTS.json") == seal["artifact_index_sha256"]
    print(json.dumps(seal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
