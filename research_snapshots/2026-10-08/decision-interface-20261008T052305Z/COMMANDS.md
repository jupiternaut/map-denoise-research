# Reproduction commands

Host: liekkas. Work directory: /srv/slam-research/grf/map-denoise/runs/decision-interface-20261008T052305Z.
Interpreter: /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. No installation or GPU use.

Run in order below. Commands are a recipe, not a claim that every phase ran. Actual status is recorded in phase_events and CHECKPOINT. All result paths refuse overwrite; reproduce into a fresh copy of source with its own output directory, keeping historical source paths readable.

```text
python -B preflight.py
python -B experiment.py prepare-calibration
python -B experiment.py score-calibration
python -B experiment.py calibrate
python -B experiment.py replay
python -B experiment.py evaluate-replay
```

Only when replay/evaluation/GATE.json passed=true:

```text
python -B experiment.py prepare-confirmation
python -B experiment.py score-confirmation
python -B experiment.py confirm
python -B experiment.py evaluate-confirmation
```

Then execute audit/verify_results.py using its --help arguments and inspect the report. B0 unit tests, calibration curves, replay and confirmation are different kinds of evidence; do not merge their samples or call this a real-scene evaluation.

## Recorded execution defect

In this frozen source, both evaluate commands write and seal every scientific output, then exit 1 because the last stdout JSON contains numpy.bool_. Do not rewrite the original failure record or treat process exit 1 as a scientific negative result. Inspect audit/EVALUATION_LOGGING_ERROR.md and verify all artifacts before adding an artifacts-complete receipt:

```text
python -B audit/record_evaluation_completion.py replay
python -B audit/record_evaluation_completion.py confirmation
```

The current run already contains these receipts. They state artifacts_complete_stdout_error, not successful process exit. Do not rerun into existing sealed output directories.
