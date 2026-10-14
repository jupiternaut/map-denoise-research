Overall: **WARN, same-family/provisional**. The controlled replay results pass independent recomputation; no blocking mathematical or result-integrity defect found. Warnings concern preregistration provenance, exhaustive cost wording, and interpretation of the small timing difference.

Target verified: host `liekkas`; exact requested root `/srv/slam-research/grf/map-denoise/runs/lovelace-proof-reuse-20261014T074405Z`, resolving to its documented storage mount. No files edited.

For evidence below, `R` denotes that exact root; `A14` denotes `/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z`.

| Check | Finding |
|---|---|
| A. Truth provenance/leakage | PASS within replay scope. Jobs strip tasks to ID/incumbent; `prepare` receives only observation/budget/options. Truth loads only in scoring after verification. Evidence: `R/experiment.py:54`, `R/worker.py:40`, `R/experiment.py:103`. Historical groups are computed after scoring (`R/experiment.py:130`). Re-ran the reviewed A14 arithmetic auditor: all 18 worlds, 7,776 photo components, 25,218 forward checks and 630 historical predictions pass, maximum numerical discrepancy 0. |
| B. Normalization | PASS. Errors are raw depth squared errors, with the same truth/incumbent and denominator of 108 perturbed inputs. No new fitted normalization or rescaling. Evidence: `R/experiment.py:107`, `R/experiment.py:110`, `R/experiment.py:118`. |
| C. Result existence/recomputation | PASS. Recomputed 252 tolerance predictions and 1,890 switch predictions directly from raw outputs. Independently reverified all 36 tolerance states/784 exclusions and all 270 switch states/6,870 exclusions. All hashes, exact rational truth coverage and unchanged correct inputs pass. Evidence: `R/outputs/tolerance/SUMMARY.json:5`, `R/outputs/switches/READY.json:3784`, `R/outputs/switches/VERIFICATION.json:2`. |
| D. Actual executed code | PASS, with a substantive negative result for inheritance. Hot ordering changes real calls; 1,128 events start at a nonzero tile. Inheritance is wired into the real path but has zero eligible parents, rechecks and hits. Evidence: `R/reuse_search.py:65`, `R/reuse_search.py:97`, `R/outputs/switches/PROFILE.json:96`, `:129`, `:131`. |
| E. Actual scope | PASS if stated narrowly: fixed four-box necessary relaxation, same exposed 18 synthetic worlds, scalar depth/hull projection. This does not establish physical image feasibility, natural-image performance, full point-cloud repair, or superiority to external methods. Evidence: `R/PROTOCOL.md:7`, `R/boundary_diagnostic.py:377`, `A14/REPORT.md:3`. |
| F. Evaluation type | WARN against calling this new confirmation. It is explicitly exposed-data development replay. Three repetitions measure timing, not 54 independent worlds. Evidence: `R/PROTOCOL.md:7`, `:9`, `R/outputs/switches/SUMMARY.json:1319`. |

Specific correctness checks:

- Pure tolerance change is exactly the original A14 source with the single `.025`→`.001` replacement at `R/pure_search.py:91`. All 63 tolerance, 65 switches and 10 supplement frozen entries currently match.
- Parent reuse is mathematically sound: same observation/camera/model scope, same branch, interval containment, child constraint reconstruction, positive exact contradiction margin, then fallback on failure. Evidence: `R/reuse_kernel.py:31`, `:61`, `:75`, `:88`, `:95`, `:122`. The underlying multiplier checker rejects nonpositive/duplicate/invalid entries at A13 `cert_kernel.py:145`.
- The all-off bridge passes beyond selected outputs: all 216 comparisons match bounds, retained cells, stopping reason and excluded cells. I reproduced these comparisons independently. Evidence: `R/switch_runner.py:73`, `R/outputs/switches/STATE_EQUIVALENCE.json:2812`.
- Full boundary coverage passes: reconstructed all 72 distinct frontier cells and their complete source mapping; independently validated every one of 288 box results. Counts are 125 exact-infeasible and 163 exact necessary-LP-feasible, no unknown. Every cell retains at least one feasible necessary relaxation. Evidence: `R/boundary_diagnostic.py:57`, `:183`, `:369`, `R/outputs/boundary-tolerance.json:32210`. These witnesses do **not** prove physical feasibility or that finer camera subdivision will help.

Supported quantitative results:

- Perturbed MSE: `6.794983958823187 → 6.763968234545869 mm²`; all correct inputs unchanged and all true depths covered.
- Hot versus off: tile proposal LPs `5124 → 4947`; observed cold mean `0.8950207792 → 0.8858515868 s`, a 1.02447% reduction.
- Inheritance: zero hits and unchanged 2,100 tile certificate LPs; no inheritance speedup claim is supported.
- Timing qualification: after averaging repetitions within each world, hot is faster in 12 worlds and slower in 6; mean difference −0.00917 s, world-level standard error 0.00718 s. Report the observed difference modestly, without claiming a robust general speedup.

Remaining reporting warnings:

1. **Diagnostic implementation was not prospectively frozen by the supplied freeze records.** `boundary_diagnostic.py` first appears at `R/FREEZE_supplement.json:6`; that file’s mtime is 16:07:38, after boundary output at 16:05:26. The diagnostic records a matching script hash (`R/outputs/boundary-tolerance.json:32209`), and all exact claims recheck. Describe this accurately; do not claim every implementation was frozen before every stage.
2. **276.362649 CPU seconds is a partial ledger**, correctly named `total_child_and_development_cpu`, not exhaustive research/workflow cost. It excludes recorded verification CPU of 0.757066882 + 6.728526472 seconds, diagnostic parent CPU, orchestration/scoring and audits. Evidence: `R/switch_runner.py:15`, `R/outputs/switches/PROFILE.json:573`, `R/outputs/boundary-tolerance.json:32222`.
3. `R/refine-logs/EXPERIMENT_TRACKER.md:5` remains stale and should reflect completed stages.

No benchmarks were rerun. Successful major audit recomputations consumed approximately 10.39 measured CPU seconds across processes; ancillary reads/checks were not exhaustively metered.
