# Error log

## E1 — isolation detector missed the cheat (phase 1)

- **When:** first `phase1_audit.py` run.
- **Effect:** honest policies passed, but `cheat` also had `n_fail=0`. Contract `cheat_must_fail` was false.
- **Cause:** unqueried labels were shifted by a constant `+13`. `argmax` of those labels is invariant under a constant shift, so a peeking policy that used `argmax(hidden)` did not change its next action.
- **Fix:** rewrite unqueried labels to `17*x+3`, and score the cheat as `hidden*31 + x`. Detector now fails 6/12 cheat trials. Honest 36/36 still pass.
- **Rerun scope:** `phase1_audit.py` only. Original mismatch files not modified. Main conflict experiment uses the rewritten-label detector, not the constant shift.

## E2 — write_file JSON object coercion (session plumbing)

- Direct `write_file` of a JSON object was rejected (`content must be a string`). Session metadata was written via `execute_code` → `write_file` with `json.dumps`.
- No experimental outcome affected.

## E3 — shared-instance sampler empty (phase 2)

- **When:** first `smoke.py` against `conflict_experiment.py`.
- **Effect:** `build_shared_instance` exhausted 20_000 draws. No development or eval run.
- **Cause:** dirty label was a raw integer offset at `x=16`, which sits between init points `{0,32}`. With slopes in `{-1,0,1}`, no H1 member can hit both endpoints and miss the middle, so the “H1 explains, H0 does not” filter was empty.
- **Fix:** put the corrupted observation at `x=32`, on the far side of `t∈{24,32}`, so `{0,16}` stay on the linear piece and H1 can match a different right-hand piece. ASSUMED construction.
- **Rerun scope:** `conflict_experiment.py` sampler only; phase-1 artifacts unchanged.
