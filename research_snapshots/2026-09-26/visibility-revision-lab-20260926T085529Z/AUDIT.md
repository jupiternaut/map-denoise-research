# Independent audit

Status: PASS. Source review, 17 targeted tests and completed-fit audit pass.

The useful contrast is the same fixed KEEP/A/B geometry, same fitting rows and
labels, with reserved photometry alone, input-derived visibility features, then
visibility–photometry interactions. Modest and richer estimators must be compared
within the same feature family. The development-selected arm must be locked before
reading replay outcomes; replay-best is a diagnostic, not a replacement winner.

The visibility representation distinguishes a candidate behind input-supported
foreground, a candidate in front of that support, and missing support. A local
inverse-depth plane fitted to actual projected support positions avoids treating
ordinary plane slant as an occlusion. The current input supplies its own depth map:
the original point may support itself, and geometry outside the cropped ROI is
absent. Self-owner, support, and residual features expose those limitations; they
do not turn this proxy into independent measured visibility.

`test_learner.py` has 12 passing tests: 96/176/192 feature dimensions and exact
prefixes; unchanged shallow baseline hyperparameters; shared sample weights with
candidate-specific labels; correct prediction columns; exclusion of unsupported
rows from calibration loss; separate recovery and native-protection objectives;
feasibility and deterministic winner selection; zero-displacement semantics; exact
fixed-candidate outputs; and front ownership of two actual input layers. The five
evidence tests additionally pass. One initial test incorrectly demanded the literal
`keep` action for zero-edit output; a threshold can make the same physical decision.
That test assertion was corrected, not the learner.

Independent camera checks covered all 245 cameras in the five scenes. The maximum
half-resolution projection error was 4.55e-13 pixels, physical camera-depth error
6.83e-13 mm, and center-coordinate error 4.55e-12 mm. All five input scale matrices
were positive isotropic similarities. The camera loader calls a COLMAP reader that
parses sparse points too, but does not use those positions; it does not load a native
mesh or evaluator reference.

Training source review confirms train24/predict37 and train37/predict24 separation.
Inference reads only frozen models, fixed candidates and observation features. The
completed-fit audit reconstructs all 24 selected policy objectives and their choices
within the saved calibration grid, all 12 fold identity hashes, all 12 fitted models'
weighted initial predictions, and all four development winners. The 79,594-by-two
`base_shallow` OOF scores reproduce the previous `independent_absolute` scores exactly
(maximum difference zero). The 78,598 fitting rows are unchanged. Initial predictions
match the directly weighted gain means to 6.1e-15. All four declared development
objectives select `base_shallow`, so the added visibility proxies and richer models
have not earned selection in this experiment. No replay metrics have been read by
this auditor at this checkpoint.

Reproduce the fitted-artifact check from this directory with the existing Python:
`python -B -c 'import json; from test_learner import audit_completed_fit; print(json.dumps(audit_completed_fit(), indent=2))'`.
