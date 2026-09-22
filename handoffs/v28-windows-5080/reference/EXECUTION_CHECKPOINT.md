# Current checkpoint: bounded confirmation completed

2026-09-22, host liekkas. Original preparation snapshots remain unchanged.
Read EXPERIMENT_REPORT.md for current results; README/CHECKPOINT/READINESS files
from preparation describe their earlier point-in-time state.

## Completed

- Input retrieval: 3,561,809,006-byte photo/calibration archive plus four meshes;
  228 specified input members extracted. Four selective official references and
  eight masks/planes retrieved only after method outputs were sealed.
- Adapter: old eight view lists exactly replayed; all four new camera checks pass.
- Construction: scan40 adaptation plus scans55/65/69 confirmation, 80 cases,
  seven fixed arms, 560 PLY outputs. Same method/model/source hashes across scenes.
- Evaluation: 560 rows, of which420 confirmation; fixed native point masks.
- Audit:80 cases pass exact routing/KEEP/count checks, max injection magnitude
  discrepancy3.63e-14mm;16 independent nearest-neighbor comparisons agree exactly.
- No GPU work, driver changes, old source/model changes, or GitHub publication.

## Frozen main result

Confirmation MSE improvement vs identity: native −8.12%; −1mm −2.00%; +1mm +9.74%;
−3mm +36.08%; +3mm +33.79%. All twelve ROI improve for each3mm sign, none for native.
Preregistered3mm recovery passes; native net-improvement test fails. Default remains
identity. Do not replace the main method with a secondary winner or tune these
now-exposed confirmation scenes and call them unexposed again.

## Honest limits / pending

Official COLMAP still NOT_RUN; its absence is not a negative algorithm result.
Clean-environment end-to-end reproduction remains incomplete.
The AABB reference set includes many surfaces outside the input photo ROI,
limiting interpretation of completeness. A posthoc support diagnostic records
this; neither original scores nor acceptance tests were changed.
This is same-source held-out scene evidence, not cross-source universality or
physical thin-layer identity verification. Reference uncertainty is unquantified.

Next contribution-focused step: finish the mature same-information/system
comparison and package this conditional-recovery evidence. A new geometry search
is not started automatically. All execution processes have completed at sealing.
