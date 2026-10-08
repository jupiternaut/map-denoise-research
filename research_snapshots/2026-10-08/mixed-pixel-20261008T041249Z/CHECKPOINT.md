# Checkpoint — mixed pixels E0/E1

- Host: liekkas.
- Exact root: /srv/slam-research/grf/map-denoise/runs/mixed-pixel-20261008T041249Z.
- Authorized task: implement/run approved mixed-pixel plan; no publication or deployment.
- E0 analytic and independent integration: DONE.
- Legacy30 full9 reproduction: DONE; MAE66, max score diff5.55e-16.
- E1: DONE; 36worlds,24design groups,30distinct image tensors,648decisions; code/input/prediction/evaluation seals exist.
- Mechanism: dynamic estimated mixture ranks600 first in30 informative worlds;6equal-color worlds flat. This is simulation development and fixed candidate truth, not transfer.
- Main endpoint: ED MAE56.67/6.67/56.67 for initial540/600/660; N10/0/10. ED=EF=OF=OD endpoints. E2 gate false.
- Loss locations:24worlds lack required training scale; four correct-input configurations (two distinct images) are moved600->660 by support-interval mean although raw candidate600 is best.
- Diagnostics:576pixel artifacts reproduce sealed loss exactly; per-region residuals, pairwise ordering, regret, background pairs saved.
- E2/E3 pressure/E4: NOT RUN. Do not claim blocked hardware: scientific gate failed. No background process remains after final verification.
- Planned next: separate decision-interface experiment, not silently replace P in this seal; avoid tuning on future E2.
- Final audit: audit/EXPERIMENT_AUDIT.md and JSON; same-family provisional. Audit verifier v1 error is retained; v2 corrected its interval formula.
- Historical runs read-only. No GitHub/GitBook push, no new dependencies.

Continue by reading REPORT.md, METHOD.json, approved plan, and audit. Never overwrite existing LOCK/SEAL/RESULTS. New methods require a new version/run, not rerunning execute_pipeline.py in this directory.
