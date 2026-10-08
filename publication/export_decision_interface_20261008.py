"""Byte-preserving publication of a completed decision-interface experiment.

One-time exporter for the verified source host. Original files are read-only.
The separate publication verifier is portable; this exporter deliberately is not.
"""
import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

from export_mixed_pixel_20261008 import PATTERNS, TEXT, digest

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path("/srv/slam-research/grf/map-denoise/runs/decision-interface-20261008T052305Z")
DEST = REPO / "research_snapshots/2026-10-08" / SOURCE.name
MANIFEST = REPO / "publication/DECISION_INTERFACE_20261008_MANIFEST.json"


def main():
    assert socket.gethostname() == "liekkas"
    assert REPO == Path("/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1")
    assert SOURCE.is_dir() and not DEST.exists() and not MANIFEST.exists()
    planned, excluded, hits = [], [], []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(SOURCE)
        reason = None
        if path.is_symlink():
            reason = "symlink not followed"
        elif ".aris" in rel.parts:
            reason = "private review traces; public structured audit retained"
        elif "__pycache__" in rel.parts or ".pytest_cache" in rel.parts or path.suffix == ".pyc":
            reason = "runtime cache"
        elif path.suffix not in TEXT | {".npz", ".png", ".svg"}:
            raise ValueError("Unreviewed file type: " + str(rel))
        if reason:
            excluded.append(dict(source=str(path), bytes=path.lstat().st_size, reason=reason))
            continue
        if path.suffix == ".npz":
            assert rel.parts[0] in {"calibration", "confirmation"}
            assert rel.parts[1] in {"data", "scores"}
        assert path.stat().st_size < 90_000_000
        if path.suffix in TEXT | {".svg"}:
            content = path.read_text()
            for kind, pattern in PATTERNS.items():
                for match in pattern.finditer(content):
                    hits.append(dict(source=str(path), kind=kind,
                                     line=content[:match.start()].count("\n") + 1))
        published = rel.with_name("SOURCE_AGENTS.md") if rel.name == "AGENTS.md" else rel
        planned.append((path, DEST / published, digest(path), path.stat().st_size))
    if hits:
        print(json.dumps(dict(secret_pattern_locations=hits), indent=2))
        raise SystemExit("Review detections; values deliberately not printed")
    files = []
    for source, target, sha, size in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copy2(source, target)
        assert digest(source) == digest(target) == sha
        files.append(dict(path=target.relative_to(REPO).as_posix(), source=str(source),
                          bytes=size, sha256=sha,
                          transformation="filename only: archived instructions"
                          if source.name == "AGENTS.md" else "none"))
    payload = dict(created_at=datetime.now(timezone.utc).isoformat(), source_host="liekkas",
                   target="https://github.com/jupiternaut/map-denoise-research", branch="main",
                   archive_roots=[DEST.relative_to(REPO).as_posix()], files=files,
                   excluded=excluded, credential_pattern_detections=0,
                   source_files_modified=False, new_research_executed=False,
                   scope="Completed B0/B1/B2/B3 simulation including synthetic arrays, failed gate, corrections and figures; no real-data backup.",
                   portability="Original host paths retained. Companion verifier reads only repository files; full runners require path and environment adaptation.")
    with MANIFEST.open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(dict(files=len(files), bytes=sum(x["bytes"] for x in files),
                         arrays=sum(x["path"].endswith(".npz") for x in files),
                         excluded=len(excluded)), indent=2))


if __name__ == "__main__":
    main()
