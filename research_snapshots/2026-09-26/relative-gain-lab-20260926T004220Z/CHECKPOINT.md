# Completed checkpoint

State: COMPLETE — bounded construction, exposed replay, independent audit, plots.
Host: liekkas. No experiment process remains running. No GPU task was started.

Primary entry: `REPORT.md`. Figure: `figures/replay_report.png`.
Reproduction: `COMMANDS.md`; independent checks: `AUDIT_REPORT.md`.

Four gain constructions + seven controls trained on scan24/37; locked before
60-case replay. All32 policy arms, 40 random controls and one fixed-A oracle retained.
No new external confirmation; no deployment-default or thesis changes; no push.

Normalized balanced native/−1/+1/−3/+3 MSE reductions:
−1.246855%, +1.757%, +3.799%, +39.039394%, +32.057528%.
Positive means improvement; native remains worse than identity. All11 native-priority
threshold searches were infeasible and explicitly returned KEEP. Not a safe-denoising win.

Implementation tests: 22 root-run unittest cases passed in30.734s
(`test_baseline_models`, `test_innovation_models`, `test_training_contract`, `audit_tests`),
then6 root-run `test_evaluation_contract` cases passed in0.005s: total28.
Independent data/threshold/metric audits all PASS; PLY differences0.

Read the audit before using the native-risk comparison: normalized native edits
only0.377% and does not beat its matched random controls there; ±3mm does.
Do not relabel this old-data replay as new blind validation.

Possible next question, not yet executed: does independent reserved-view evidence
for both incumbent and candidate identify useful native corrections? Any extension
must use a new workspace and lock its protocol before accessing a new test set.
