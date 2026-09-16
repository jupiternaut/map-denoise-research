# Real support bridge v1

An executed, photo-fixed real-object suitability assay. **Not a new denoiser.**

The question is whether the existing scan24 reconstruction and independent laser
reference can support the local parallel-surface mechanism studied in V15/V16.
V23's failed photometric selector is not modified, and V16's synthetic dependency
experiment is not repeated.

- [Result and decision](REPORT.md)
- [Protocol fixed before reading ROI reference geometry](PROTOCOL.md)
- `assay.py`: camera/mesh/reference ROI extraction and plane descriptions.
- `test_assay.py`: six basic geometry and input-contract unit checks.
- `verify.py`: independent checks of the saved run.

Completed run:
`/srv/slam-research/grf/map-denoise/runs/real-support-bridge-v1-9de7jep3`

```bash
cd /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B real_support_bridge_v1/assay.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover \
  -s real_support_bridge_v1 -p 'test_assay.py' -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B real_support_bridge_v1/verify.py \
  --run /srv/slam-research/grf/map-denoise/runs/real-support-bridge-v1-9de7jep3
```

The assay creates a unique run; it does not overwrite old runs. Reference plane
labels are diagnostics, not public physical surface ground truth. The saved
`provisional_parallel_pair` is a scope-screen flag, not proof of (im)possibility.
No scan ID is substituted for layer identity; no point spacing is substituted for
sensor noise. No data download, training, GPU experiment or GitHub push occurred.
The verifier also uses exclusive output creation: the recorded run already has
`VERIFICATION.json`; use a fresh assay run to repeat the full write-producing check.
