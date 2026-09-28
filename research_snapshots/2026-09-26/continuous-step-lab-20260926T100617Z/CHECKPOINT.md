# Completed construction checkpoint

Host liekkas. Code `/home/grf/Documents/Codex/2026-09-26/continuous-step-lab-20260926T100617Z`.
Artifacts `/srv/slam-research/grf/map-denoise/runs/continuous-step-20260926T100617Z`.

Implemented and ran all60 replay cases,10 evaluation arms,600 metric rows. Actual
grid/quadratic point clouds sealed before reference evaluation. Exact analytic
finite-reference continuous oracle separately labelled and exported. No new data,
GPU, downloads, deployment, Git or thesis changes. Previous checkpoints read-only.

Main: old discrete oracle -3/+3 gains69.9427/63.2670%; continuous74.6905/64.9853%.
Fixed-prior-route continuous oracle51.2480/40.8043%. Actual grid47.2105/38.5482%
vs prior46.9944/39.0222%; native actual -6.3972%,0/12 improvingROI. No promotion.

Candidate representation has more potential, but branch/KEEP selection dominates
remaining gap for large offsets. Photo grid supplies small conditional gains;
quadratic interpolation is not a consistent gain. Constant shrinkage is a strong
small-perturbation comparator. Next proposed experiment is joint branch/step
selection on development data, not more interpolation on the frozen route.

22 tests; independent brute-force and photo-score audits; full artifact/source
verification via `python -B verify.py`. Run verify only on this sealed checkpoint.
Do not repeat writer commands on these destinations. No background job remains
after final handoff. Report and figures distinguish potential from actual results.
