# E0-D independent module diagnostics

This directory is a completed CPU experiment, not a replacement for old E0.

- REPORT.md: outcome, tables and limitations.
- PROTOCOL.md: frozen main design.
- DISCRETIZATION_DIAGNOSTIC.md: separately declared numerical follow-up.
- COMMANDS.md: execution record and environment.
- VERIFICATION.json: source integrity, unit tests and stored-array checks.
- results/{detector,projection,boundary,guards}: raw NPZ/JSON/CSV.

Safe verification in this exact directory:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B verify.py
```

This rechecks existing outputs and refreshes VERIFICATION.json; it does not
rerun experiments or modify old E0. Experiment entrypoints are listed in
COMMANDS.md. Preserve existing raw results rather than rerunning over them.
