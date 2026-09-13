from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def dump_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
