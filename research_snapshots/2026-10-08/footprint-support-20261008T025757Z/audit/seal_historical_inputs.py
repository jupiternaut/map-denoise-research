"""Read-only historical input seal, written solely into this audit directory."""

import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
HIST = AUDIT.parent.parent / "surface-owned-support-20261008T022918Z"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HIST / "mechanism/FIXTURES.json").read_text())
    fixed = [HIST / "support.py", HIST / "mechanism/run_mechanism.py",
             HIST / "mechanism/PROTOCOL.json", HIST / "mechanism/FIXTURES.json",
             HIST / "mechanism/OBSERVATIONS.json",
             HIST / "mechanism/OBSERVATIONS_LOCK.json",
             HIST / "mechanism/SELECTIONS.json"]
    fixtures = [HIST / "mechanism/fixtures" / f["arrays"] for f in manifest]
    old_obs = json.loads((HIST / "mechanism/OBSERVATIONS.json").read_text())
    baselines = [HIST / "mechanism/curves" / o["curve_file"] for o in old_obs
                 if o["arm"] in ("plane_full9", "plane_connected9")]
    files = {str(p): digest(p) for p in fixed + fixtures + baselines}
    result = {"historical_root": str(HIST), "file_count": len(files),
              "files": files, "all_fixture_manifest_hashes_match": all(
                  files[str(HIST / "mechanism/fixtures" / f["arrays"])] == f["sha256"]
                  for f in manifest)}
    target = AUDIT / "HISTORICAL_INPUTS_SEAL.json"
    if target.exists():
        previous = json.loads(target.read_text())
        changed = [p for p, value in previous["files"].items()
                   if files.get(p) != value]
        print(json.dumps({"recheck": True, "changed": changed,
                          "all_fixture_manifest_hashes_match": result[
                              "all_fixture_manifest_hashes_match"]}))
        if changed:
            raise SystemExit(1)
    else:
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"created": str(target), "file_count": len(files),
                          "all_fixture_manifest_hashes_match": result[
                              "all_fixture_manifest_hashes_match"]}))


if __name__ == "__main__":
    main()
