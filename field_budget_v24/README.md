# V24: equal-budget repair fields and allocation diagnostics

This is a completed, bounded mechanism experiment, not a new default denoiser.
Both reused scenes are exposed. No refitting, new ROI, GT-guided selector,
download, GPU work, or GitHub push was performed.

- [Research report](REPORT.md)
- [Frozen protocol](PROTOCOL.md)
- `field_ops.py`: input-only budget-normalized candidate construction.
- `confidence.py`: signed gain ranking for alpha under fixed probe actions.
- `run.py`: seal cached/native and new candidates, evaluate, aggregate.
- `verify.py`: independent reconstruction and evaluation; imports neither runner
  nor construction/confidence modules, reads PLY using Open3D instead of plyfile.

Run from `/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B \
-m unittest discover -s field_budget_v24 -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B field_budget_v24/run.py
```

The runner creates a unique directory under `/srv/slam-research/grf/map-denoise/runs`.
Verify the printed path with the same interpreter and `field_budget_v24/verify.py RUN_PATH`.
The completed run is:
`/srv/slam-research/grf/map-denoise/runs/field-budget-v24-uvg3jss3`.

Reproduction needs the frozen V22 cache at
`/srv/slam-research/grf/map-denoise/runs/reconstruction-v22-qayc8gft`
and two local DTU laser references recorded in `REFERENCE_MANIFEST.json`.
This is not yet a portable public reproduction bundle. Runtimes reuse saved
fitting and exclude original V22 fitting costs. A run on different hardware
must report its own timing and verification rather than inherit this run's numbers.

Core outputs: `SEALED.json`, `RESULTS.csv/json`, `SUMMARIES.csv/json`,
`COMPARISONS.csv/json`, `CONFIDENCE.json`, `CONFIDENCE_SUMMARY.json`,
`RUN_SUMMARY.json`, `VERIFICATION.json`, and actual `.npy` candidate point arrays.
No old source or checkpoint was overwritten. New source lives independently.
