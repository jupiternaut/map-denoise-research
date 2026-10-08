# Run boundary

Host liekkas; this exact directory is the new decision-interface experiment. Older runs and the repository plan are read-only. Write only assigned new files here; never overwrite generated results. Use CPU, at most four workers, no GPU, external API or new dependencies. All authored edits use apply_patch.

Use the approved plan at /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/planning/decision-interface-20261008/refine-logs/EXPERIMENT_PLAN.md. B0 implementation checks precede B1 calibration; freeze calibration before B2 replay; B3 only if the registered B2 gate passes. Confirmation does not tune rules. Prediction receives images, supplied cameras, candidate depths, incumbent and locked calibration, never evaluation truth. Calibration truth is a separately budgeted input. Record exclusions and KEEP in denominators.

Source imports from mixed-pixel-20261008T041249Z are read-only; set PYTHONDONTWRITEBYTECODE=1. No Git push in this run. Final assessment must separate interface repair, imaging contribution and transfer.
