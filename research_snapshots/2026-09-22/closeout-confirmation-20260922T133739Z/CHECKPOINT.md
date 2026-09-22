# Checkpoint — design and release preparation complete

2026-09-22, host liekkas. New workspace only was edited.

## Decisions not to reopen on confirmation outcomes

- Main: post_A_keep; secondary: post_AB_keep.
- Old scenes24/37 development; scan40 adaptation; scan55/65/69 confirmation.
- Native point support fixed across all five conditions; native and perturbation
  results separate. The injected displacement is per-point reference-ray offset.
- Do not choose winners after evaluation, tune the three confirmation scenes,
  use evaluator geometry for alignment, or substitute self-written fusion for
  official COLMAP. Meta-research is not a prerequisite to close the geometry work.

## Actual state

Unified models and portable API ready; 7 tests passed twice on this host.
24 old-fold cases / 1,181,301 rows replay identically. This is compatibility,
not independent performance of the newly combined-scene models.
Metadata confirms four scene meshes and selective reference members, and a
3,561,809,006-byte tar.gz photo/calibration archive. Planned total ~3.918 GB.
No new-scene contents downloaded/opened. No independent quality results.
COLMAP not found in bounded checks, GPU ~6.7/8.2 GB occupied; no installation,
no reconstruction, no process termination. Baseline adapter still not implemented.

## Next concrete action

Implement bounded retrieval and a scan40 adapter, preserving the frozen method.
This is within the proposed closeout work, not a request to redesign geometry.
Prepare official COLMAP separately without modifying the old Python environment.
Only after input/coordinate and view-list validation should confirmation start.
All three confirmation outputs must be sealed before references are opened.

## Scope of verification

PREPARATION_AUDIT.json checks package files, 48 archived training input files,
and old V28 top-level manifest hashes. It does not rehash all historical datasets,
recompute all old experiments, or certify a full external reproduction.
Unrelated files in the historical Git working tree remain untouched; no Git push.
