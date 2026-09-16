# E0 diagnostic implementation

Target host: liekkas. Work only in this new directory. The sibling `e0/` is
read-only, including its source, protocol, outputs and caches. Import it with
PYTHONDONTWRITEBYTECODE=1. Do not install packages, use GPU, or publish.

Use `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`.
Use apply_patch for source and documentation edits. Programs may write their
own experimental outputs under this directory. Keep output arrays and per-case
metrics, including negative results. Do not choose parameters using evaluation
truth. Calibration labels are allowed only in the declared calibration channel.

Read PROTOCOL.md before implementation. Parallel ownership: detector agent owns
detector.py and tests/test_detector.py; guard agent owns guards.py and
tests/test_guards.py; main agent owns other files. Do not edit each other's files.

No claims of full CPR-2 success, FDR validity under a tolerance null, real-scene
validation, or cross-domain generalization arise from these module diagnostics.
