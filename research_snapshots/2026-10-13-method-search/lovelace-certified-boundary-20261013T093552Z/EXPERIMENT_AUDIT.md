# A13 Experiment Audit

Date: 2026-10-13. Exact host: liekkas. Logical root: /srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z. Its documented physical resolution is /mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z.

Auditor: fresh /root/a13_integrity_review, requested route GPT-6-Astra / ultra. Review independence: same-family. Semantic acceptance: provisional. Deterministic arithmetic replay: accepted for the saved artifacts only.

## Overall verdict: PASS within the declared simulation scope

Integrity status: pass. No blocking defect, fabricated truth, self-normalized score, missing formal prediction, or mismatched reported primary number was found. This is not external acceptance or a guarantee for arbitrary real images. REPORT_DRAFT.md and CHECKPOINT_DRAFT.md were read only after primary arithmetic.

## A. Ground-truth provenance: PASS

The frozen A12 data_builder.py samples latent depth from the seeded RNG before rendering (lines 119–165), renders observations from depth, camera offsets, material colors, bias and noise, and only then creates seven incumbents. It never obtains truth from an estimate or baseline output. A13 run_experiment.py:68–82 invokes the frozen generator with new seeds 20261013131/132/133 after checking the source freeze. The evaluator loads private truth at run_experiment.py:226–247 after the complete-prediction gate.

The audit read all 18 stored truth records and replayed exact photographic integration, clipping, camera acquisition and appearance acquisition directly from stored latents in independent_audit.py:72. All exact images/anchors and their binary64 public transports match. For every world, the generating nuisance point satisfies every reconstructed necessary constraint and its box at the true depth. Every world has exactly one exact correct incumbent and six shifted incumbents.

The inference boundary was read directly: worker.py:14–42 passes only observation to prepare, and one incumbent at a time to decode. search_method.py:21–73 and the frozen free method do not read task populations or private truth. Worker has tasks only for dispatch; this is a code/interface boundary, not an OS sandbox or historical access log.

## B. Score normalization: PASS

run_experiment.py:174–183 and 241–247 compute raw absolute/squared depth errors. MSE/MAE divide by task counts; the optional improvement percentage divides by the identity input MSE, not a statistic of the method's own predictions. Raw values and denominators are preserved. The independent audit rebuilt all 756 metric rows directly from stored predictions and true depths, reproduced METRICS.csv exactly, and recomputed every SUMMARY subgroup. No prediction-max/min normalization exists in the metric path.

For each budget, the 108 shifted inputs have identity MSE 20.772441760185202 mm². Certified/numeric MSE is 6.536208253123856 and MAE 1.951385119364746, with 70 improved / 38 unchanged / 0 harmed. All 18 correct inputs remain unchanged. These denominators include the three negative-control worlds.

## C. Result existence, completeness, timing and freeze: PASS

All 108 manifest jobs, result files, receipts, running records and completed terminal records exist and agree on identity. The complete cross product is 3 arms × 18 worlds × 2 budgets. Each job has seven predictions; all 756 arm/budget/task tuples are unique and match original task IDs/incumbents. No missing or dropped failure denominator was found; all worker return codes and all prediction failure fields are successful.

All 40 SOURCE_FREEZE.json entries match current bytes. The recorded source-freeze digest binds both generation and PREDICTIONS_READY. Observed UTC timestamps:
- source freeze 10:04:54.588654;
- generation 10:04:55.387770;
- formal manifest 10:04:55.456852;
- all predictions ready 10:08:59.982641;
- independent verification 10:09:11.922544;
- metrics/summary 10:09:12.088306 / 10:09:12.089122.

All result timestamps precede PREDICTIONS_READY. This establishes consistency with the intended ordering through code, hashes and filesystem evidence; it is not an adversarial tamper-proof execution history. The audit checks the 40 named freeze entries, not an assertion that all Python/toolchain/transitive imports are fully sealed.

COSTS matches receipts plus worker costs. Total cold-process wall time is 244.35631888371427 s; child CPU is 244.25637700000016 s; controller batch wall time is 244.52401880198158 s. No reported prepare overrun >0.25 s or prediction failure occurred. Per-world cold-process means:
| Arm | 6-second budget | 24-second budget |
|---|---:|---:|
| numeric | 0.2740491861 s | 0.2740258675 s |
| certified | 0.3969686782 s | 0.3967653021 s |
| legacy | 4.4665898879 s | 7.7669521273 s |

Legacy/certified time ratios are 11.2517438604 and 19.5756838771. Certified MSE is 0.9752426381% lower than legacy@6, but 2.1720618890% higher than legacy@24. Online cold-process times exclude controller input-JSON creation and later independent verification, as the draft states.

## D. Executed metrics, certificates, partitions and decoder: PASS

The metric aggregation functions are called by score and their outputs are present in the saved JSON/CSV. analyze_costs.py:17–58 is backed by saved states and receipts; its proof cost totals/ratios agree with independent recomputation.

The standalone verifier was read line by line. verify_certificates.py:263–324 validates observation binding, branch/endpoints, variable order, rational syntax, positive nonduplicate multipliers, row layout, claimed margin and a strictly positive exact contradiction. Lines 327–394 check per-branch partition coverage, retained hull and fallbacks. The audit called it on all 36 formal certified states and six diagnostic states. Every state passes. It separately recomputed every saved decoded prediction as incumbent projection (or identity fallback) and compared all outputs.

There are 594 formal certificate occurrences: 297 distinct certificates repeated across two budgets. Every distinct formal certificate was additionally checked by materializing the complete reconstructed constraint array and independently summing the full rational box margin, instead of trusting the verifier's sparse-row materialization. All margins agree and are strictly positive. This extra algebra check reuses the standalone verifier's geometric reconstruction; the direct point-world rendering/acquisition replay is separately authored in the audit script.

Across the 594 occurrences, support sizes are 1:24, 2:140, 3:70, 4:80, 5:220, 6:50, 7:10; mean 3.8787878788. Each budget has 297 attempts and 297 accepted exclusions. All 36 numeric/certified full retained-cell lists and exclusion proposal sequences agree exactly; their 252 predictions agree exactly. The six dev records contain the same 89 numeric proposals and all 89 verify.

All certified searches report boundary_resolution. There are no observed naturally rejected proposals, no confirmation-set conservative widening cases, and no new result from extending the fast-method budget from 6 to 24 seconds. test_harness.py:26–35 exercises refusal by forcing a false verification result; that test is not evidence for empirical rejection frequency.

## E. Scope and draft claims: PASS with explicit ceilings

Independent sampling scope is 18 simulated worlds in three seed blocks of six, spanning three rear branches and two material strata, with three designated negative controls. Seven incumbents are repeated evaluations within a world, not seven scenes. The two budgets repeat the same worlds; they are not independent replications. The six exposed dev worlds are diagnostic only.

REPORT_DRAFT.md numerical claims match the evidence. Its post-score residual claim was independently recomputed from original saved predictions: certified vs legacy@24 is worse on all 15 informative worlds and equal on all three negative controls; negative controls account for 0.6091450332539206 of certified squared residual error. This descriptive decomposition is not a preregistered inferential endpoint or a parameter-selection justification.

C1 is supported in these worlds: the exact-certified implementation retains a large cold-process time advantage with a measured precision tradeoff against legacy@24. C2 is supported only for the observed proposals: all accepted, no output change. The study cannot estimate how often future float exclusions will fail certification. No statistical-significance, noninferiority, cross-mechanism or real-point-cloud conclusion follows.

The legacy interface uses Q(str(x)) at frozen library/methods.py:16–29. Its rational solver works on decimalized readings, unlike A13's closed binary64 rounding preimages. REPORT_DRAFT.md:72 discloses this asymmetry, so the comparison does not imply identical transport-certification guarantees.

## F. Evaluation type: simulation_only

The truth is a sampled latent physical world, not dataset-provided real-world ground truth and not an estimate-derived pseudo-reference. Exact algebra validates contradictions in the reconstructed necessary relaxation of the frozen camera/material/noise model. Its link to all model worlds depends on the correctness of the necessary-condition derivation and stipulated input/noise bounds. It is not a theorem about arbitrary real images, point-cloud completeness, topology or unknown nuisance distributions.

## Action items and claim impact

No experiment or implementation correction is required by the evidence inspected. Retain the stated limits in the final report, particularly the legacy@24 accuracy advantage, repeated-budget nature, lack of naturally refused certificates, and simulation-only scope. Replace draft review-in-progress language only after this audit is linked; do not promote same-family semantic review to external or cross-family acceptance.

- C1: supported, limited to the frozen 18-world comparison.
- C2: needs qualifier, all observed proposals verified; rejection/widening behavior was not naturally exercised.
- Exact saved algebra/partition/decoder results: deterministically verified.
- Arbitrary real-world safety, broad robustness, optimality or novelty: unsupported.

## Audit artifacts and execution

independent_audit.py and INDEPENDENT_AUDIT_RESULTS_FINAL.json contain the reproducible checks and input hash inventory. The first saved-artifact audit used 2.466 CPU s; the final pass including post-score residual checks used 2.757 CPU s. No inference rerun, new world, model tuning, original result edit or ancestor edit occurred. The existing inference sources stayed frozen.

Reviewer trace: .aris/traces/experiment-audit/2026-10-13-a13/. The initial unnecessary SSH attempt failed host-key checking; local hostname/path verification then established the exact target. No SSH trust/configuration was changed. A progress message initially misstated the freeze entry count as 38; this was corrected to 40 before the verdict. No result discrepancy was found.

## Final report consistency check

Read REPORT.md and CHECKPOINT.md in full after publication and compared them with the reviewed drafts. The report retains all verified numbers, the legacy@24 accuracy disadvantage, simulation-only scope, transport-interface asymmetry, no-natural-refusal qualification and same-family/provisional semantic status. The finalized versions differ from the latest reviewed drafts only in completion status and audit evidence links. No new experiment was run for this check. PASS remains unchanged.

- REPORT.md and REPORT_2026-10-13_18-23.md: SHA-256 5e44f635415ab4bbc8176577f5c314cf7da8b9927d645fb96fb71070a1800162; identical bytes.
- CHECKPOINT.md and CHECKPOINT_2026-10-13_18-23.md: SHA-256 6da2085fd084edeae13bd2c6a9250dd2b45997b6a93dad9057bc78805eeceb99; identical bytes.

Final follow-up request/response and metadata are saved as 002-final-report.* in the same private trace directory. All requested audit work is complete.
