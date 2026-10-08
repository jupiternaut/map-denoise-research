# Mixed-pixel geometry experiment

Host: liekkas. Language: zh. Writable root: /srv/slam-research/grf/map-denoise/runs/mixed-pixel-20261008T041249Z.
Plan (read-only): /srv/slam-research/grf/map-denoise/plans/mixed-pixel-study-20261008T035954Z/refine-logs/EXPERIMENT_PLAN.md.
Historical runs and research repositories remain read-only. User authorized implementation and experiments according to that plan, not GitHub/GitBook publication or deployment.

- Use apply_patch for authored files. Generated artifacts through reproducible programs. No installs, GPU, API keys or external writes. Existing Python: /srv/slam-research/grf/map-denoise/envs/colmap-cuda12-421/bin/python -B. Keep BLAS threads=1 and worker count<=2 given current memory use.
- Stage order E0 -> E1 -> conditional E2. Do not open confirmation data until the main method and decision interface pass development gates. Failed E1 must not be relabeled as permission to tune on E2.
- Six arms KEEP/full9/estimated-fixed/estimated-dynamic/oracle-fixed/oracle-dynamic. Source images/cameras/candidates identical across arms. Ordinary estimation may read only reference+training view; scoring view cannot fit appearance. Two folds share data and are not independent trials.
- Freeze appearance, boundary and background estimate before scoring depth. Candidate changes only geometric projection/coverage. Fixed-alpha fixes per-subray ownership at incumbent depth but retains candidate texture warp. Both arms score the same raw integer pixel ROI; no per-candidate pixel dropping.
- Oracle auxiliaries explicitly privileged, loaded only by separate oracle stage. No true depth, label, phase, scene name or rendered ownership enters ordinary estimator. Do not import historical renderer in ordinary inference.
- Preserve old P endpoint; also report raw predictive curves and candidate ranking so support-gate failure is not confused with absent information. Absolute and relative residual constants are experimental choices, not probability guarantees.
- New E1: 24 base worlds +12 background pairs, three incumbents 540/600/660, six arms =>648 decisions. Truth depth600 and candidates450/540/600/660/900 are controlled fixture design, not general transfer. Legacy30 at900 separately.
- Maintain correct-input and both shift directions; retain rejected cases in denominators. F=B negative control must produce no invented information. Pair/group by base scene; no pixel or fold pseudoreplication.
- Source/protocol hashes, auxiliaries and predictions are sealed before geometry evaluation. Record implementation repairs and do not overwrite a sealed run.
- Work assignments: predictor agent owns predictor.py and predictor tests/notes; fixture agent owns fixtures.py, e0.py, fixture tests/notes; baseline agent owns baseline.py, baseline tests/notes. Main owns runner/evaluation/protocol/report. Auditor only writes audit/ and never edits method or evaluation.
