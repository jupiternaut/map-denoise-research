"""Resume-safe ETH3D training laser-scan download. Official URLs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from paths import ETH3D_DOWNLOADS, ETH3D_EXTRACTED, LOGS, require_liekkas

OFFICIAL_PAGE = "https://www.eth3d.net/datasets"
DOC_PAGE = "https://www.eth3d.net/documentation"
BASE = "https://www.eth3d.net/"
LICENSE = "CC BY-NC-SA 4.0 (stated on https://www.eth3d.net/)"
FILES = (
    "data/courtyard_scan_raw.7z",
    "data/courtyard_scan_clean.7z",
    "data/delivery_area_scan_raw.7z",
    "data/delivery_area_scan_clean.7z",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def download_one(rel: str) -> dict:
    url = urljoin(BASE, rel)
    name = Path(rel).name
    dest = ETH3D_DOWNLOADS / name
    part = ETH3D_DOWNLOADS / (name + ".part")
    ETH3D_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    if dest.exists() and dest.stat().st_size > 0:
        digest = sha256_file(dest)
        return {
            "name": name,
            "url": url,
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": digest,
            "status": "already_present",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "-C",
        "-",
        "-o",
        str(part),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "name": name,
            "url": url,
            "path": str(part),
            "status": "failed",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-2000:],
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
    digest = sha256_file(part)
    part.replace(dest)
    return {
        "name": name,
        "url": url,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "status": "downloaded",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "license": LICENSE,
        "source_page": OFFICIAL_PAGE,
        "local_hash_is_not_official_signature": True,
    }


def list_archive(archive: Path) -> list[dict]:
    proc = subprocess.run(
        ["7z", "l", "-slt", str(archive)],
        capture_output=True,
        text=True,
        check=True,
    )
    members = []
    current: dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith("Path = ") and current:
            members.append(current)
            current = {"path": line[7:]}
        elif line.startswith("Path = "):
            current = {"path": line[7:]}
        elif line.startswith("Size = ") and current:
            current["size"] = int(line[7:])
        elif line.startswith("Packed Size = ") and current:
            packed = line[14:].strip()
            if packed:
                current["packed_size"] = int(packed)
        elif line.startswith("Attributes = ") and current:
            current["attributes"] = line[13:]
    if current:
        members.append(current)
    # 7z -slt repeats the archive path as the first Path entry.
    return [m for m in members if m.get("path") and m["path"] != str(archive) and m["path"] != archive.name]


def safe_extract(archive: Path, dest_dir: Path) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    members = list_archive(archive)
    unsafe = []
    total = 0
    for member in members:
        rel = Path(member["path"])
        if rel.is_absolute() or ".." in rel.parts:
            unsafe.append(member["path"])
        total += int(member.get("size") or 0)
    if unsafe:
        return {"archive": str(archive), "status": "unsafe_members", "unsafe": unsafe}
    if total + 2 * 1024 ** 3 > free_bytes(dest_dir):
        return {
            "archive": str(archive),
            "status": "insufficient_space",
            "uncompressed_bytes": total,
            "free_bytes": free_bytes(dest_dir),
        }
    marker = dest_dir / ".extract_ok"
    if marker.exists():
        return {
            "archive": str(archive),
            "status": "already_extracted",
            "dest": str(dest_dir),
            "members": members,
            "uncompressed_bytes": total,
        }
    proc = subprocess.run(
        ["7z", "x", "-y", f"-o{dest_dir}", str(archive)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "archive": str(archive),
            "status": "extract_failed",
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[-2000:],
        }
    marker.write_text(json.dumps({"archive": archive.name, "members": len(members)}, indent=2))
    return {
        "archive": str(archive),
        "status": "extracted",
        "dest": str(dest_dir),
        "members": members,
        "uncompressed_bytes": total,
    }


def main() -> None:
    require_liekkas()
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    records = []
    for rel in FILES:
        print(f"download {rel}", flush=True)
        rec = download_one(rel)
        print(rec.get("status"), rec.get("bytes"), rec.get("sha256", rec.get("stderr", ""))[:80], flush=True)
        if args.extract and rec.get("status") in {"downloaded", "already_present"}:
            scene = rec["name"].split("_scan_")[0]
            variant = "raw" if "_raw." in rec["name"] else "clean"
            rec["extract"] = safe_extract(Path(rec["path"]), ETH3D_EXTRACTED / scene / variant)
            print("extract", rec["extract"]["status"], flush=True)
        records.append(rec)
    out = {
        "host": "liekkas",
        "official_page": OFFICIAL_PAGE,
        "documentation": DOC_PAGE,
        "license": LICENSE,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    dest = LOGS / "eth3d_download.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
