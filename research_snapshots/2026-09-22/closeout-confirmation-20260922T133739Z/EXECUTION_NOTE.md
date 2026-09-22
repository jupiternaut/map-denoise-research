# Execution addendum — 2026-09-22

User authorized starting the experiment after the sealed preparation.
Original preparation files, package models, source snapshot and protocol remain
unchanged. This addendum records execution, not a replacement preregistration.

- New inputs downloaded to the predeclared `/srv/.../closeout-confirmation-v1/`
  directory. Four input meshes verified against published LFS SHA-256. The input
  gzip archive is retained with its calculated hash. 228 selected files extracted,
  734,513,352 bytes; no nonselected scene images extracted.
- Eight old ROI source lists replay exactly with the new input adapter. Physical
  scale/projection checks and sparse reprojection criteria mirror the old adapter.
- The scene40 construction is run before the three confirmation constructions.
  Confirmation processes share the same sealed package/model and protocol, one
  numeric CPU thread each. Concurrency is across scenes, not changing algorithms.
- Reference retrieval is gated on all four `SEALED.json` outputs. It is a separate
  command; evaluation verifies both the output seals and reference checksums.
- All construction outputs preserve original rows, all seven arms and all five
  input conditions. There is no per-test model/threshold change.
- Evaluator unit counterexamples test output leaving support and empty output,
  followed by independent SciPy/Open3D nearest-distance checks on actual native ROIs.

Official COLMAP remains a separate unfinished baseline. No substitute is relabeled
as official; no GPU allocation, driver change or unrelated process termination.
The three-scene CPU comparison can establish its own outcomes without pretending
the official-baseline work package is complete.
