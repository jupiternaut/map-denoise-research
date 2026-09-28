# Joint revision experiment

Exact host liekkas. Write only this new workspace; all old runs/data remain read-only.
Task: implement and test observation-only KEEP/A/B joint selection, with candidate-
specific photometric support, not another series of scalar veto rules. No deployment,
Git publication, downloads or installs. CPU only; existing open3d-019 Python, -B,
OMP/OPENBLAS/MKL threads=1. Development24/37; exposed replay55/65/69. No replay
reference in features, training or calibration. Label oracle outputs explicitly.

Keep code direct: one data adapter, pure decision functions, one trainer, one evaluator.
Check shapes/row identities, view split, loss units, exported coordinates and unchanged
sources; do not construct a general validation framework. Preserve failed outcomes.

Root owns common.py, router.py, train.py, infer.py, protocol/report and finalization.
support_evidence owns support_features.py, extract_support.py, evidence/, its tests.
joint_evaluation owns evaluate.py, evaluation/, test_evaluation.py.
joint_audit owns audit.py, AUDIT.json/md, test_router.py and read-only implementation review.
Use apply_patch for code/docs. Outputs exclusive. No more methods after replay results.
