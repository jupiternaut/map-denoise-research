# Six-track feasibility experiments — 2026-09-11

Host: liekkas. User requested parallel constructive experiments and an advisor report.
This is a new experiment workspace, not a restoration or overwrite of the old project.

## Preserved checkpoint

`/srv/slam-research/grf/map-denoise/checkpoints/20260911-1404-pre-paper/research-state.tar.gz`
passed its existing SHA256 check at 16:00 Asia/Shanghai. Old project, paper and raw
results are read-only dependencies. Do not alter them, including generating bytecode there.

## Six questions

1. Identifiability: does crossed frame support distinguish true layers from frame bias?
2. Estimation: can a legal-input constrained frame-bias estimator improve geometry?
3. Filtering: does separation-aware filtering help when bias correction is held fixed?
4. Evaluation: do geometry/structure/coverage metrics distinguish paired failure modes,
   and is existing real observation support sufficient for the proposed task?
5. Efficiency: can we reduce redundant mixture computation while preserving final output?
6. CAD route: can an automatic slow XYZ-only reference give a gain against relevant existing
   baselines on more than the old two-plane toy? CAD-like procedural geometry must not be
   presented as an acquired CAD dataset.

## Ownership and resources

- Agent theory_bias owns `tracks/t1_t2/` and corresponding new run subdirectories.
- Agent geometry_reference owns `tracks/t3_t6/` and corresponding new run subdirectories.
- Agent efficiency owns `tracks/t5/` and corresponding new run subdirectories.
- Root owns `tracks/t4/`, this plan, integration and final report.
- Run root: `/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600`.
- Python: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
- Limit each process to one BLAS/OpenMP thread; avoid more than one process per agent.
- No large dataset downloads, environment changes, deletion, or new paid services.
- Use apply_patch for code/docs; scripts may write their own measured outputs.

## Evidence rules fixed before runs

All exploratory cases are development evidence. Preserve every run, variant and failure.
Use same legal input for algorithmic comparisons; XYZ-only versus provenance-aware is an
information ablation, not an equal-information method victory. GT oracles are diagnostic only.
Scalar common-bias models with supplied normals/noise bounds must be labelled restricted
models, not general 3D or real-sensor solutions. Do not label own simplified code JRMPC/BALM.
Each track must emit source, executable test/run instructions, raw JSON/CSV, representative
outputs, REPORT.md, and limitations. Save algorithm outputs before evaluating them.

Do not rank theory, benchmark construction and speedup as if all were denoising algorithms.
Compare tracks by evidence of geometric benefit, same-input external baselines, independent
object transfer, compute cost and remaining unknowns. A follow-up confirmation set must be
selected only after candidate freezing; no fresh generalization claim from this initial round.

The first deliverable is a six-row advisor-ready feasibility report, not a claim that six
complete research projects or six original algorithms have been finished.
