# Checkpoint

Host liekkas, 2026-09-26. Local field construction and all 60 exposed replays complete.
Inference: 7 actual arms, 420 PLY. Main evaluation: 12 arms, 720 metric rows.
All three inference seals preceded evaluation-reference access; all three evaluation
seals and consolidated evaluation seal exist. No old artifact, deployment default,
Git remote or thesis was changed.

Primary `multi_field` is not promoted. Actual MSE gains for −3/+3mm are
23.8866%/21.9479%, below prior recovery46.9944%/39.0222%; native worsens83.3516%.
Expanded diagnostic oracle reaches91.94%/93.10%, mostly because of the broad
nine-position pool, not field predictions (increment only~0.28/~0.18pp over pool).

Next work should change how a surface predicts observations and preserve broad
candidate availability, not call the larger oracle deployable accuracy. No new
threshold search is authorized by this checkpoint itself.

Additional field-only capacity decomposition, if present, is explicitly post-hoc
diagnosis in a separate directory and does not modify the sealed main results.
It is complete: field-only KEEP/K2 oracle reaches62.69%/62.66% on −3/+3mm,
so both candidate-domain restriction and selection contribute to the deficit.
See REPORT.md, COMMANDS.md, VERIFICATION.json and the representation audits.
