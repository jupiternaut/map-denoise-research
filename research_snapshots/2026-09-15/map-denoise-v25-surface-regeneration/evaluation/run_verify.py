"""Recompute E_sym for sealed PLYs without importing evaluation.metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration")
sys.path.insert(0, str(ROOT))

from evaluation.verify import recompute
from src.v25.io_util import read_json, write_json
from src.v25.paths import EVAL_ONLY

RUN = ROOT / "runs" / "2026-09-14T165314Z"
ARMS = (
    "identity",
    "fusion_wta",
    "restricted_single",
    "restricted_point_move",
    "v25_atlas",
    "v25_wta_atlas",
    "v25_depth_cc",
    "v25_gated_move",
)


def main() -> int:
    rois = read_json(RUN / "rois.json")
    rows = []
    for roi in rois:
        spec = EVAL_ONLY[roi["scene_id"]]
        for arm in ARMS:
            ply = RUN / roi["roi_id"] / f"{arm}.ply"
            if not ply.is_file():
                continue
            got = recompute(ply, spec["laser"], spec["mask"], roi["aabb_min_mm"], roi["aabb_max_mm"], 0.8)
            prod = None
            metrics_path = RUN / roi["roi_id"] / "metrics.json"
            if metrics_path.is_file():
                for item in read_json(metrics_path):
                    if item.get("arm") == arm:
                        prod = item
                        break
            delta = None
            if prod and prod.get("E_sym_mm") is not None:
                delta = abs(float(got["E_sym_mm"]) - float(prod["E_sym_mm"]))
            rows.append(
                {
                    "roi_id": roi["roi_id"],
                    "arm": arm,
                    "verify_E_sym_mm": got.get("E_sym_mm"),
                    "prod_E_sym_mm": None if prod is None else prod.get("E_sym_mm"),
                    "abs_delta": delta,
                    "verify_n": got.get("n_output_eval"),
                }
            )
            print(f"{roi['roi_id']} {arm} verify={got.get('E_sym_mm')} delta={delta}", flush=True)
    write_json(RUN / "VERIFY.json", rows)
    deltas = [r["abs_delta"] for r in rows if r["abs_delta"] is not None]
    print("max_abs_delta", max(deltas) if deltas else None, "n", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
