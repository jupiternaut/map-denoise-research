# Visibility revision experiment

Exact host liekkas. Code/docs here; bulky outputs in
`/srv/slam-research/grf/map-denoise/runs/visibility-revision-20260926T085529Z`.
All prior workspaces/data read-only. CPU only. Use existing open3d-019 Python with
-B and single-thread BLAS. Do not install, download, publish or alter deployment.

Implement input-derived per-view occlusion evidence and learn KEEP/A/B selection.
Dev24/37 only for model/threshold selection;55/65/69 exposed replay only after freeze.
No GT, native unperturbed surface, evaluator errors, or condition names in evidence.
Rasterize the CURRENT case input only; never the native parent for injected cases.
This is an occlusion proxy, not independent observed depth or certified visibility.
Fixed existing A/B positions. Record failed arms alongside improvements.

Root owns common.py, training, inference, protocol, final report.
visibility_evidence owns visibility_features.py/extract_visibility.py/test_visibility.py,
  outputs/evidence and feature documentation. No evaluator reference access.
visibility_evaluation owns evaluate.py/test_evaluation.py and outputs/evaluation.
visibility_audit owns tests/audit of root learner and read-only evidence attribution.
Use apply_patch for code/docs. Avoid generic frameworks; test specific contracts.
