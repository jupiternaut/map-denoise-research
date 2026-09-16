# Commands

Exact host: liekkas.
Working directory: `/home/grf/Documents/Codex/2026-09-16/e0-f-incumbent-support-20260916T060448Z`.
No installed packages, GPU, external publishing, or historical writes.

```bash
cd /home/grf/Documents/Codex/2026-09-16/e0-f-incumbent-support-20260916T060448Z
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s tests -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B run_experiment.py
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B verify_outputs.py
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B inspect_real_evidence.py
```

All commands completed successfully (auxiliary inventory syntax fixed before successful run).
`run_experiment.py` refuses to overwrite existing results. Reproduction needs a declared
new output workspace; do not remove or edit this sealed result set.
The verifier refreshes only new `INDEPENDENT_RECHECK.json` and derived loss ledger;
inventory refreshes only this workspace's inventory. Neither changes historical inputs.

