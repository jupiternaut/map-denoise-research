# Relative-gain laboratory

Exact host: liekkas. Writable root: this directory only. All previous research
workspaces, datasets, archived outputs, and environments are read-only. Use
`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`; disable bytecode
and set OMP/OPENBLAS/MKL thread counts to 1. CPU only for this bounded run.
No installation, data download, GitHub push, GPU takeover, or process termination.

The task is implementation and actual experiments, not another proposal.
Frozen A candidates are shared by every primary arm. Train/tune only on scan24
and scan37. Historical scan55/65/69 are exposed replay, NOT unseen confirmation.
Their labels must not enter model fitting, threshold selection, or feature design.
Use only cached observable 64-dimensional A features in learned policies.
GT may enter training and the separate evaluator, never policy.score(X).
All arms and failures remain visible. Retain identity and the old frozen policy.

Ownership: root owns data preparation, protocol, orchestration, evaluation and
report. Baseline agent owns baseline_models.py and BASELINE_NOTES.md. Constructor
owns innovation_models.py and CONSTRUCTION.md. Auditor owns audit_*.py and
AUDIT*.md/json, and may read all new code. Do not edit another owner's files.
Use apply_patch for code/document edits. Runtime-generated results are exclusive
new artifacts. Ask root before changing shared APIs. Do not tune on replay results.

Shared policy API: `train_policies(X, gain, e0, e1, sample_weight, seed)` returns
dict[str, policy]. Each policy exposes `score(X)` -> finite 1D float array, larger
means more willing to move, and is joblib serializable. It accepts no scene,
condition, labels, reference, or filesystem path at inference. Each policy exposes
`default_threshold` (float or None). Root chooses calibrated thresholds by two
scene-held-out development folds under a common, locked procedure.
