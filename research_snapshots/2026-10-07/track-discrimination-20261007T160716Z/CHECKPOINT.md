# Checkpoint — 2026-10-07T16:41:11Z

Host liekkas; this run only writable. Completed CPU-only experiment and independent audit; no deployment or GitHub push. Repository HEAD observed ca078aed3df4f348ccb708281488fcc9bfd70a11; pre-existing untracked AgentRx source files left untouched.

## Result

Goal: preserve valid geometry, repair errors. Frozen candidate-blind star gives ROI-equal MSE16.856879 vs photo23.955251 (29.6318% reduction), with3 improved/2 worsened/479 unchanged of484 primary points; cycle23.027711 (3.8720%) with1/1/482. New photo≤1→>1mm harm2/1. Primary cycle criterion NOT MET. Both branches only reroute21 historical residual requests; no new-scene confirmation.

## What is usable

- Conditional exact affine gain kernel and 13 tests.
- Raw candidate-blind image search with sealed curves and requests.
- Two major star distance reductions:30.929151→0.519399mm;44.106924→1.768738mm.
- Reproducible peak-representation omission: query5 and75 have nonpeak witnesses passing existing criteria; query95 has subthreshold source-source evidence; query48 passes despite worsening evaluation.

## Resume here

Read REPORT.md, observation/DIAGNOSTIC.md, EXPERIMENT_AUDIT.md and AUDIT_RESOLUTION.md. Keep old code/results/seals readonly. Next implementation question is complete threshold-support plus a single shared 3D feasibility variable, with unchanged candidates and thresholds. Do not promote star or adjust deployment on these exposed cases. New-scene confirmation remains outstanding.

## Verification

See COMMANDS.md for safe read-only commands. verify_run.py verifies34 hash entries,3584rows and63 exact-rational decisions. Independent audit additionally checks all metric fields, original image curves, and614 original laser distances (max difference0). Audit WARN is same-family/provisional and retains the missing initial protocol snapshot limitation.
