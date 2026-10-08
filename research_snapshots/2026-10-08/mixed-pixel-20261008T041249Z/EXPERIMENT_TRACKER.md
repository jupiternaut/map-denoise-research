# Experiment tracker

| ID | Status | Evidence | Decision |
|---|---|---|---|
| E0 analytic contracts | DONE | e0/RESULTS.json | passes numerical checks; analytic self-consistency is not performance |
| E0 independent integration | DONE | e0/INTEGRATION.json; test_integration.py | 4/4 contracts pass |
| Legacy full9 | DONE | legacy_replay/RESULTS.json |30cases exactly same decisions; floating difference <1e-12 |
| E1 six-arm development | DONE, endpoint gate FAILED |648decisions; evaluation/ | mechanism retained; current P pipeline not adopted |
| Pixel/residual replay | DONE, descriptive only | diagnostics/pixels/ |576NPZ; losses exactly match seals |
| Integrity audit | DONE, WARN | audit/ | 11307 deterministic checks passed; same-family provisional, scope warning |
| E2 sealed confirmation | NOT RUN | no confirmation dataset generated | E1 gate false |
| E3 stress/cross diagnostics | NOT RUN | — | no threshold tuning or extra model search |
| E4 real replay | NOT RUN | — | no new real-data claim |
| GitHub/GitBook publication | NOT REQUESTED THIS RUN | — | no push |

Source/input/method locks were retained. New follow-up experiments must use a new run/version and may not overwrite this result.
