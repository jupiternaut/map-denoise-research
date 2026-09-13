# T2 specialized filter construction — 2026-09-11

Host liekkas. New work only; the six-track checkpoint and old map projects remain read-only.

## Target

User update: prioritize the specialized filter itself, not completeness, exhaustive boundaries or
formal verification. Expand the observed gain of separating frame bias from layer proportions.
Deliver actual operators, practical-effect sweeps, concise mechanism derivation, outputs and report.
Keep basic fair comparison and no-GT rules; do not make a general proof a prerequisite for construction.

Workspace: /home/grf/Documents/Codex/2026-09-11/map-denoise-t2-boundary-v1
Run root: /srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704
Python: /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python
All processes: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1.

## Parallel ownership

1. theory_bias: alternative/ — an independent practical per-frame split/align/consensus filter;
   short mechanism derivation only; concentrate on geometry gain, not complete formal proofs.
2. geometry_reference: candidate/ — strengthen scalar joint estimation using per-frame mixture
   proportions and better initialization, with baseline ablations and legal Input only.
3. efficiency: baseline/ — locate and execute relevant official joint registration where practical;
   independently implement a strong observation-only same-scalar reference if needed, precisely named.
4. root: common data/evaluation and sweep runner, integration, held-back seed replication,
   output audit, integration, report and checkpoint.

## Operator interface

Use the immutable Input class from
/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py.
Input fields: xyz_mm (N,3), frame (N), roi (N), sigma_mm, bias_bound_mm.
The caller supplies the local normal coordinate as column 2; no truth passed to the operator.
Each new module must export METHODS and estimate(inp, method), returning (xyz_mm, bias_mm, info).
Info must include status and enough estimated geometry/association for diagnostics; finite arrays.
Freeze module and configuration before root confirmation seeds. All outputs saved before evaluation.

## Planned axes and integrity

Vary layer separation/noise, layer proportion imbalance, points per frame, crossed support,
bias amplitude, provided noise misspecification, normal error and frame-dependent spatial bias.
Single-plane and observationally ambiguous paired cases remain in every integrated conclusion.
No GT/oracle used by deployment or method selection. Official ICP is an external baseline but not
a substitute for all joint methods. Every extra piece of information is disclosed.
Normal error is a controlled coordinate perturbation, not a claim of tested normal estimation.
Synthetic new seeds are replication, not independent real-scene generalization.

The first boundary sweep is development. A follow-up new-seed subset is run after source freezing;
no tuning after seeing it. Default comparison includes the original scalar joint model,
frame-centering + same mixture, raw XYZ mixture, and any stronger same-information reference.
Report runtime and point count alongside normal/3D error, layer spacing and single-plane splitting.
Keep failed/weak conditions; do not call a high acceptance rate a geometric gain.
