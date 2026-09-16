"""New systems for the capability trial. Construction six is smoke only."""
from __future__ import annotations

from typing import Dict, List

from builder.plant import HiddenSpec


CAPABILITY_SPECS: List[HiddenSpec] = [
    HiddenSpec(
        system_id="T0_linear",
        mismatch="none",
        m=1.0,
        k=3.2,
        c=0.35,
        seed=11,
        split="capability",
        notes="linear; no new structure required",
    ),
    HiddenSpec(
        system_id="T1_cubic_x",
        mismatch="dynamics",
        m=1.0,
        k=3.0,
        c=0.2,
        k_nl=1.1,
        seed=12,
        split="capability",
        notes="cubic restoring force; v3 not required",
    ),
    HiddenSpec(
        system_id="T2_memory",
        mismatch="memory",
        m=1.0,
        k=4.0,
        c=0.3,
        memory_gain=1.6,
        memory_decay=0.4,
        seed=13,
        split="capability",
        notes="hidden drive memory; same visible (x,v) after opposite pre-drives",
    ),
    HiddenSpec(
        system_id="T3_drift",
        mismatch="observation",
        m=1.0,
        k=4.0,
        c=0.45,
        drift_rate=0.07,
        seed=14,
        split="capability",
        notes="sensor bias grows with time",
    ),
    HiddenSpec(
        system_id="T4_mem_x3",
        mismatch="memory",
        m=1.0,
        k=3.6,
        c=0.25,
        k_nl=0.9,
        memory_gain=1.3,
        memory_decay=0.45,
        seed=15,
        split="capability",
        notes="memory + cubic x; not a named FAMILY_SPEC entry",
    ),
    HiddenSpec(
        system_id="T5_mem_slow",
        mismatch="memory",
        m=1.0,
        k=4.0,
        c=0.2,
        memory_gain=1.1,
        memory_decay=0.22,
        seed=16,
        split="capability",
        notes="slower memory; history IC matters more",
    ),
]


def tasks() -> List[Dict]:
    return [
        {"system_id": s.system_id, "mismatch": s.mismatch, "spec": s, "init_mem": 0.0}
        for s in CAPABILITY_SPECS
    ]
