# Reproduce and inspect

Host `liekkas`; Python `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
Code: `/home/grf/Documents/Codex/2026-09-26/continuous-step-lab-20260926T100617Z`.
Artifacts: `/srv/slam-research/grf/map-denoise/runs/continuous-step-20260926T100617Z`.
Frozen dependencies and data are explicit in `common.py`; no downloads needed.

Run from the code directory, with `PYTHONDONTWRITEBYTECODE=1` and single-thread
OMP/OPENBLAS/MKL. `common.py` also fixes those numerical thread counts.

```bash
python -B -m unittest -v test_segment_oracle.py test_step_observation.py test_evaluation.py
python -B step_observation.py --scene 55
python -B step_observation.py --scene 65
python -B step_observation.py --scene 69
# All three inference seals must exist before evaluation.
python -B evaluate.py --scene 55
python -B evaluate.py --scene 65
python -B evaluate.py --scene 69
python -B evaluate.py --finalize
python -B plot.py
python -B verify.py --seal
python -B verify.py
```

Scene-level jobs can run in parallel (at most3 single-thread workers). Writers use
exclusive creation and must not run again on sealed outputs. Reproduction means
explicitly configuring a NEW output path, preserving the prior run. Pure tests
and `verify.py` are read-only. The optional `--case scan55_roi0__native` smoke
writes a separate `observation_smoke` tree, not the actual inference tree.

Oracle PLY is labelled DIAGNOSTIC, with original input retained outside the fixed
evaluation support. Do not deploy it. Actual grid/quadratic PLYs have all original
rows and never consult that support mask or evaluator references.
