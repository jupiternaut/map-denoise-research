# Phase 1 correction (original MISMATCH_REPORT.md kept)

Original files not modified. SHA256 of `mismatch_experiment.py` still `302a8b6f8c19d480d95697481c5299a5c7f5f8421865796e7ac44d540f0f22cd`.

## Random vs disagreement before H0 is refuted

Same per-task seed (`7000+0`) on the 60 type-1 eval tasks.

- 33/60: H0 already empty after the two initial queries `{0,32}`. No subsequent query. Sequences identical.
- 27/60: live H0 members still agree on all unqueried `x` (`nunique=1`), so disagreement **falls back to random**. With the shared seed it then draws the same `x` as the random policy (often 48).
- **0/60** issued a true disagreement query before H0 was refuted.
- `mean_true_disagreement = 0`, `mean_fallback_zero = 0.467`.

The previous “~0.4 query earlier H0 refute” is not a disagreement-sampling effect on this split. It is init plus random fallback. The type-1 L1 gain at budget 4 still stands as sampling *after* H0 is gone, which was outside this check.

## Isolation

Two `HiddenWorld` objects: same queried map and RNG. Unqueried answers in B are rewritten to `1000+x` (not stored as `_unused`).

- Honest selectors (fixed / random / disagreement): 36/36 next action unchanged.
- Cheating selector (max residual of unqueried labels vs current delivery): 28/36 actions change. The 8 misses are cases where the cheater already picked the same `x` on the original residuals. Uniform `+13` (old test) caught 0/36 and is not a valid cheat detector.

The original isolation test could not have caught a leak. This one can, and honest policies still pass.
