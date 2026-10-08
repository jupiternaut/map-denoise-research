# Research contract

Original goal: preserve correct geometry and repair displaced geometry. Approved claims C1/C2 and all thresholds are defined by the repository EXPERIMENT_PLAN.md (2026-10-08 decision-interface).

Primary arm ED+S1+M; competitors never replace it after results. Budget: B0 deterministic checks, 48 independent calibration objects, 36 sealed old worlds (30 distinct tensors), conditionally 48 new confirmation scenes. CPU only. New independent midpoint renderer7x7 versus frozen9-ray predictor.

Outputs: calibration artifact, candidate curves, predictions, per-case evaluation, rejection/fallback accounting, gate, report/checkpoint, reproducible audit. No real-data claim, no prior report overwrite, no deploy default change.

Addendum: B0 also checks the conditional separation lemma: for true candidate in C and total prediction error bounded by eta in one common RMS norm, minimum separation >2eta implies unique residual argmin recovery. A counterexample outside that sufficient condition is permitted. Estimated MAD is not eta. This mathematical check does not introduce a new empirical algorithm or extend the confirmation gate.

The bridge skill's template file is absent; this explicit compact contract substitutes for that document template only. Experiments are manually gated CPU phases, not the GPU queue scheduler.
