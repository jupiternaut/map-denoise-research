# Continuous-step geometry experiment

Exact host: liekkas. This new code directory is writable. Artifacts only in
`/srv/slam-research/grf/map-denoise/runs/continuous-step-20260926T100617Z`.
All historical experiments and datasets remain read-only. CPU, existing Python,
single-thread BLAS; no installations, publishing, deployment or thesis changes.

Construct continuous positions along the fixed input-to-A and input-to-B segments.
Distinguish exact finite-reference diagnostic oracles from observation-only methods.
Reference laser and evaluation support are evaluator-only inputs. Actual methods
must not use either, injected-condition names, or an unperturbed native parent.
All current scenes are exposed replay, not unseen confirmation. No selection of
new methods or parameters after viewing replay evaluation outcomes.

Root owns common.py, PROTOCOL.md, integration/evaluation/report. continuous_oracle
owns segment_oracle.py and its tests. continuous_observation owns
step_observation.py and its tests. visibility_evaluation audits read-only and may
write AUDIT_REVIEW.md. Use apply_patch for code/docs; preserve immutable inputs.
