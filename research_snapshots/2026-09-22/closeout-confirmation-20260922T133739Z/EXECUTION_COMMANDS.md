# Commands used in the execution phase

Host `liekkas`; workspace:
`/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z`.
Interpreter `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.

Every Python command used `-B`, `PYTHONDONTWRITEBYTECODE=1` and numerical thread
variables `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
The output directories are append-only/exclusive; do not rerun them over the
completed artifacts. For reproduction use a new copy of scripts/package with
empty result folders and the same input/model hashes. Metadata/old protocol
snapshots are preserved in the original workspace.

Executed order:

```text
python -B fetch_inputs.py
python -B validate_adapter.py
python -B run_closeout.py --scene 40

python -B run_closeout.py --scene 55
python -B run_closeout.py --scene 65
python -B run_closeout.py --scene 69

python -B test_evaluator_contract.py
python -B fetch_references.py
python -B evaluate_closeout.py
python -B audit_execution.py
python -B summarize_execution.py
```

The three confirmation constructors run concurrently, each one CPU thread; all
finish and seal before reference retrieval starts. Evaluator-contract tests use
synthetic arrays only and can run during construction. Output audit is read-only
and can run during scoring. Summarization must wait for completed scoring.

No COLMAP reconstruction command has been executed. See the baseline readiness
report for its still-unresolved installation and five-view adapter requirements.
No GPU computation or old workspace modification is part of these commands.

New-scene model inference requires only the input meshes, photos, camera files
and packaged models. `evaluation_only/` and `REFERENCE_MANIFEST.json` are opened
only by the reference retrieval/evaluation phase, never the constructor/selector.

The downloaded tar.gz contains other scenes, but only the four specified input
scenes are extracted. Raw image/calibration license terms must be checked before
any redistribution; no third-party data is uploaded by this experiment.
