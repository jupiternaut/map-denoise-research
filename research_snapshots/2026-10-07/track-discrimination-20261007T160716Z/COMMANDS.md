# Commands — liekkas / frozen run

Exact run: `/srv/slam-research/grf/map-denoise/runs/track-discrimination-20261007T160716Z`.
Python: `/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python`.
CPU only; no dependency installation and no GPU job.

## Read-only verification (safe on this frozen run)

From the run root:

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B verify_run.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B test_integration.py
```

From the `theory/` directory:

```bash
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B -m unittest -v test_kernel.py
/srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B verify_artifacts.py
```

`verify_run.py` checks 34 hash entries, 3,584 metric rows, seven methods and 63 decisions.
Decision verification independently evaluates exact rational squared distances at interval endpoints rather than importing the inference formula. It does not certify physical identity or interval coverage.

## Commands executed to create results (do not rerun in sealed directory)

1. `prepare.py` prepared candidate-blind image/camera requests.
2. `observation/extract.py` ran self-tests, locked configuration, produced observations and sealed them.
3. `theory/mechanism.py` rendered fixed synthetic scenes. After the first run, only the gain arithmetic was hardened; the scene parameters were not changed. See `theory/RUNLOG.md`.
4. `experiment.py freeze` locked source and evidence hashes.
5. `experiment.py infer` selected frozen candidates, without opening evaluation files, and sealed predictions.
6. `experiment.py evaluate` verified exact coordinate identity and reused their historical laser distances.

The main pipeline opens results in exclusive-create mode, so it deliberately refuses reruns into existing output files. The synthetic renderer does not have that protection. Never invoke it again on this sealed directory; a reproduction needs a distinct, explicitly identified run directory.

## Scope

The metric is ROI-equal nearest original laser-vertex MSE, not the full official DTU benchmark. Camera-ray squared-error bounds and nearest-vertex evaluation are different losses. Both old scenes and the 21 residual requests have been exposed during development. No deployment, GitHub push or new-scene confirmation was performed in this turn.
