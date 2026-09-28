# Run and reproduce

Target host: `liekkas`. Workspace:
`/home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z`.

All experiments use the existing Open3D environment, one CPU thread per process,
no GPU, and no dependency installation. The scripts refuse to overwrite their
output directories. For an actual rerun, copy the source scripts to a **new empty
directory**, leaving all old sources and this sealed run read-only. Their `ROOT`
is the script directory; dataset/source paths are recorded in `common.py`.

```bash
cd /home/grf/Documents/Codex/2026-09-26/relative-gain-lab-20260926T004220Z
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GAIN_PYTHON=/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
"$GAIN_PYTHON" -B prepare_data.py
"$GAIN_PYTHON" -B train_experiment.py
"$GAIN_PYTHON" -B infer_replay.py --scene 55
"$GAIN_PYTHON" -B infer_replay.py --scene 65
"$GAIN_PYTHON" -B infer_replay.py --scene 69
"$GAIN_PYTHON" -B evaluate_replay.py
"$GAIN_PYTHON" -B plot_results.py
```

The three inference processes were run concurrently. Inference seals must all
exist before evaluation opens the reference geometry. `prepare_data.py` opens
reference labels only for the two development scenes, never replay scenes.

## Tests

```bash
"$GAIN_PYTHON" -B -m unittest -v test_baseline_models test_innovation_models test_training_contract test_evaluation_contract audit_tests
```

Pure synthetic implementation tests are distinct from the independent audit of
real data, thresholds, and metrics. See `AUDIT_*.json` and audit source files.
All old data are machine-local dependencies; this is a provenance-rich local
reproduction package, not yet a standalone downloadable dataset distribution.

## Data products

- `data/train.npz`: 79,594 archived training rows, 78,598 on calibration support.
- `training/`: out-of-scene development predictions, threshold grid, fitted models,
  pre-fit protocol/source lock, final SHA-256 seal.
- `inference/scan*/`: 60 cases, 32 fixed selector arms; decision masks, scores,
  frozen-policy and two predeclared-lead point clouds, with no evaluation labels.
- `evaluation/`: fixed-support errors, diagnostic random/oracle masks, all metrics,
  archived-baseline reproduction checks, independent PLY round-trip checks.
- `figures/`: source-linked publication plots; `REPORT.md`: interpretation.

The random controls and fixed-A oracle are evaluation-only diagnostics, not
deployable methods. Reference information never selects replay thresholds.
