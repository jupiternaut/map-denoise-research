# Commands and exact environment

Host: liekkas. Directory:
`/home/grf/Documents/Codex/2026-09-16/e0-e-selective-revision-20260915T160016Z`

Use the existing Python, without bytecode or new installs:

```bash
cd /home/grf/Documents/Codex/2026-09-16/e0-e-selective-revision-20260915T160016Z
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s tests -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B verify_outputs.py
```

The second command recomputes saved metrics and refreshes only this experiment's
`results/INDEPENDENT_RECHECK.json`; all input outputs and historical directories
remain untouched. It checks the recorded historical and source hashes.

The original execution command was:

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B run_experiment.py
```

It intentionally refuses to rerun over the existing `results/`. For independent
regeneration, make an explicitly identified new experiment workspace, preserve
frozen sources, and declare that workspace rather than treating it as this run.
Do not remove or replace this run's results to make the command succeed.

Regenerate the derived static figures (overwrites only generated figures):

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B figures.py
```

Runtime observed: Python 3.12; NumPy2.2.6, SciPy1.15.3, Matplotlib3.11.1.
Exact Python version and original execution checks are in
`results/RUN_VERIFICATION.json`. No external services or GPUs are needed.
