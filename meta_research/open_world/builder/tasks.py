"""Construction tasks. Builder-only. Not a participant answer key."""
from __future__ import annotations

from typing import Dict, List

from builder.plant import CONSTRUCTION_SPECS


def tasks() -> List[Dict]:
    mem = {
        "C0_linear": 0.0,
        "C1_cubic_damp": 0.0,
        "C2_memory": 0.0,
        "C3_drift": 0.0,
        "C4_memory_pair_B": 1.6,
        "C5_unrepresentable": 0.8,
    }
    out = []
    for spec in CONSTRUCTION_SPECS:
        out.append(
            {
                "system_id": spec.system_id,
                "mismatch": spec.mismatch,
                "init_mem": mem[spec.system_id],
                "spec": spec,
                "split": "construction",
            }
        )
    return out
