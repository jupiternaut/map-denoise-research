"""Locate an official COLMAP binary. Does not install software or claim a GPU run."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SEARCH_DIRS = [
    Path("/usr/bin"),
    Path("/usr/local/bin"),
    Path("/home/grf/.local/bin"),
    Path("/srv/slam-research/grf/map-denoise/envs"),
]


def probe() -> dict:
    which = shutil.which("colmap")
    found = []
    if which:
        found.append(which)
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        try:
            for path in root.rglob("colmap"):
                if path.is_file() and os_access(path):
                    found.append(str(path))
                    if len(found) >= 8:
                        break
        except (OSError, PermissionError):
            continue
        if len(found) >= 8:
            break
    unique = list(dict.fromkeys(found))
    versions = []
    for exe in unique[:3]:
        try:
            result = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=8, check=False)
            versions.append({"exe": exe, "returncode": result.returncode, "head": (result.stdout or result.stderr)[:240]})
        except (OSError, subprocess.TimeoutExpired) as exc:
            versions.append({"exe": exe, "error": str(exc)})
    return {
        "which": which,
        "found": unique,
        "versions": versions,
        "status": "FOUND" if unique else "NOT_FOUND",
        "official_baseline": "BLOCKED" if not unique else "AVAILABLE_NOT_RUN",
    }


def os_access(path: Path) -> bool:
    return path.stat().st_mode & 0o111 != 0


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
