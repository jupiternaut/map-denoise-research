# Commands and dependencies

Target: host `liekkas`, workspace
`/home/grf/Documents/Codex/2026-09-26/reserved-evidence-lab-20260926T011717Z`.

Python: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
Each command used `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
and Python `-B`. No GPU, install, download or external mutation.

## Executed order

```bash
python -B -m unittest -v test_paired_features.py
python -B extract_evidence.py --scene 24
python -B extract_evidence.py --scene 37
python -B extract_evidence.py --scene 55
python -B extract_evidence.py --scene 65
python -B extract_evidence.py --scene 69
python -B train_selectors.py
python -B infer_selectors.py --scene 55
python -B infer_selectors.py --scene 65
python -B infer_selectors.py --scene 69
python -B evaluate_reserved.py
python -B audit_evidence.py
python -B audit_selection.py
python -B audit_metrics.py
python -B plot_results.py
python -B finalize.py
```

Independent evidence audit overlapped training; scenes were CPU-parallel. Outputs use
exclusive creation, so production scripts intentionally refuse to overwrite this sealed run.
Do not delete checkpoints to rerun them. A future reproduction needs a separately authorized
fresh output directory and the frozen source files, with all absolute dependencies preserved
or explicitly remapped and reported.

This is a local reproducible experiment, not yet a standalone distribution: `common.py`
declares the three prior frozen workspaces and dataset root; the preceding workspace's
`data/train.npz`, models/decisions and `evaluation/*/point_errors.npz` are required.
Dataset files are not bundled or copied into this workspace.

## Read-only verification after completion

```bash
python -B -m unittest -v test_paired_features.py
python -B finalize.py --verify-only
```

The final verifier checks stage seals, source hashes and final hashes; creation mode also
records a fresh unit-test result. Audit programs write new result files exclusively, so their
existing PASS reports are verified rather than overwritten.

Figure provenance is recorded in `figures/MANIFEST.json`.
