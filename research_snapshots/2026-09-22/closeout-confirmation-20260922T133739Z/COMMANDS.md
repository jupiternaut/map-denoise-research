# Commands and execution boundary

These commands use the exact host `liekkas` and workspace below. No installation
is required for the preparation tests. They do not execute new-scene evaluation.

```bash
cd /home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
V28_SOURCE=/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z \
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s tests -v

python3 -B baseline_probe.py --skip-development-inputs
```

Completed metadata probe: `python3 -B data_probe.py` (writes DATA_READINESS.json).
Do not rerun it over the frozen snapshot; use a new audit directory if refreshing
metadata. Completed one-time seal: `python3 -B verify_preparation.py`; it refuses
to overwrite PREPARATION_AUDIT.json.

The unified training command is in REPRODUCIBILITY.md and deliberately refuses
to overwrite TRAINING_LOCK/models. Do not rerun training to tune confirmation.

Portable runtime invocation and array/camera contract: `package/README.md`.
It contains no evaluator or new-scene loader. Official COLMAP stage argv are
templates in BASELINE_READINESS.json, not a completed executable baseline.

**Pending implementation**, not fictitious commands:

- bounded download/extraction driver with separate input/reference manifests;
- scan40 input calibration/ROI/source adapter and fresh-scene method runner;
- official five-view camera/track adapter plus COLMAP installation;
- native-support-fixed evaluator and output-first reference-opening lock;
- clean environment and a complete data-to-table reproduction command.

No background process or GPU wait automation was launched by this preparation.
