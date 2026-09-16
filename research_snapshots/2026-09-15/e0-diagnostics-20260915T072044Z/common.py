"""Read-only access to the exact old E0 source and structured artifact helpers."""
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'e0'
SOURCE = OLD / 'e0_mechanism_test.py'


def load_old():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('frozen_e0', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plain(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)
        f.write('\n')
