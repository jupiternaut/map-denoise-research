# Real support bridge v1 — domain fit before algorithm expansion

2026-09-13, host liekkas. Development assay, not V24 or new-scene confirmation.

Question: does an existing real reconstruction contain locally parallel, spatially
separated surface support that makes the V15/V16 mechanism relevant? If so, does
the existing spatial estimator add value over same-information simple fitting?

V16 already tested the synthetic position/label-dependence switch. Do not repeat it.
The six ETH3D patches lack independent physical layer identities; scan37 includes
curved/crossing metallic objects. Neither is relabelled as parallel thin layers.

## Fixed object choice (before reading reference geometry or fitting results)

Use scan24 photo 0000.png and its actual COLMAP pixel calibration. Photo shows
a building facade with recessed windows. Pixel rectangles [left,top,right,bottom]:

- window_a: [640, 371, 736, 472]
- window_b: [750, 432, 842, 526]
- wall_control: [720, 419, 756, 450]

These are semantic selections from the photo, not selected by method performance.
They are three regions on ONE already-exposed scene, not independent scenes.
A window contains frames/reveals and is not assumed to be two parallel planes.

Use the published GeoSVR mesh and independently acquired DTU laser reference.
Apply the exact linked scale matrix once and the V23-checked COLMAP calibration.
Camera centers must agree with linked metadata; no ICP or GT-fit alignment.

## First deliverable: real geometry suitability assay

Project original mesh vertices into each rectangle. Remove back surfaces using
first-hit raycasting on the unchanged mesh: range discrepancy <= 0.75 mm.
Reference samples use the same image rectangle and a deliberately broad 20 mm
first-hit range band to exclude other sides of the building. Report range-band
counts at 5/10/20 mm; this is an input-conditioned ROI, not complete GT visibility.
No GT reference fitting enters a deployable geometry estimator.

Describe the reference with deterministic sequential RANSAC planes (seed fixed,
0.30 mm point-to-plane threshold, 1000 trials, at most 6 planes), retaining all
residual points and reporting sensitivity at 0.15 and 0.60 mm. These are descriptive
geometric hypotheses, NOT sensor noise estimates, physical identities or new GT.
Inspect photo overlay, reference cross-sections and plane assignments.

A provisional two-plane candidate requires, at the fixed 0.30 mm tolerance:
two dominant planes each >= 15% support, together >= 70%, normal angle <= 5 deg,
and local separation >= 0.5 mm. Record all criteria, not only a pass flag.
This engineering scope screen is not a theorem or a guarantee of physical layers.
If it fails, do not run a forced parallel-layer algorithm and call it a test of
V15's domain claim. Stop the branch here and report the mismatch.

## Conditional comparison, only if the target is suitable

Use the identical original mesh vertices/support/input-only coordinate frame for
the strong constant shared-slope mixture and spatial shared-slope mixture already
implemented in V16. Include identity, single plane and official RIMLS.
Use the same fixed sigma sensitivity grid (0.25/0.5/1.0 mm); none is called known
sensor sigma and no reference-based per-region selection is a deployable method.
No reference layer labels, planes, noise or correct slopes enter the estimators.
Seal outputs before geometry scoring. Report each configuration, distance to the
reference, reference coverage and fitted geometry; no single best-number upgrade.

Because inputs are finished mesh vertices, even a positive result would support
a local surface-estimation task, NOT causal claims about raw first returns or
joint scan-pose correction. If reference geometry fits but existing simple methods
already suffice, do not automatically expand the method.

## Preservation and decision

New files and a unique new run directory only; preserve old checkpoints and V23.
No downloads, drivers, environments, GPU implementation, paper, advisor brief or push.
Useful outcomes are (a) a genuine task match worth a later frozen comparison,
(b) an explicit real-input mismatch, or (c) no margin beyond a simple baseline.
None is rebranded as a new successful general-purpose filter.
