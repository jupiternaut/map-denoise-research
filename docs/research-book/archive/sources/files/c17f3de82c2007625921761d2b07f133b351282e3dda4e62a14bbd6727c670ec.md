# Construction run v0: executable models, not a catalog query

Old 2340-world scheduler contest is frozen and was not continued.
This directory is the first *running* open-model challenge, not a
paper and not a closed-book LLM exam.

Question: starting from a simple, possibly wrong 2-D linear/nonlinear
fitter, can a policy *propose a new structure* (new state or new
measurement map), commit a prediction, then intervene — and beat a
wide 2-D library plus a non-symbolic residual predictor on a locked
holdout?

This is a **trusted-algorithm experiment**: the plant lives in-process;
policies see only API returns. There is no OS sandbox. Do not call it
a closed-book LLM science test.

LLM-ACES already jointly constructs equations and adaptively samples
trajectories.[1][5] LLM-AutoSciLab and LLM-SR already generate
hypotheses or equation programs.[2][3][6] Sparse identification of
nonlinear dynamics is the classical equation-discovery baseline.[4]
Official LLM-ACES requires PySR + an LLM API; this machine has
neither. Gap recorded; this run is **not** their official
implementation and is not named as one.

## What ran (MEASURED)

Host: `liekkas`. Wall 24.7 s. Six construction systems, five policies,
budget 8 billed actions (init traces count). Holdout menu locked in
`host/score.py` (including two memory-branch interventions that share
measurable `(x,v)` and differ only in hidden memory). Scale = RMS of
holdout measurements, frozen by the host.

| policy | constructors | experiment schedule |
|---|---|---|
| A | 2-D library `linear2`,`nl2` | covering menu |
| B | same 2-D library | family-disagreement, with committed predictions |
| C | open: 2-D + `memory3` + `obs_drift` | same covering menu as A |
| D | same open constructors as C | commit-then-intervene; optional rest calibration / drive split |
| ridge | lag-1 ridge residual (non-symbolic) | covering menu |

A vs C is a **representation** change. D−C is the sequential increment
under shared constructors. D−A is a system comparison, not attributed
to scheduling.

Isolation tests (`tests/test_host.py`): honest next action invariant;
cheat that peeks unrun labels changes after rewrite; commit after
labels refused; fake zero prediction has large realized MSE; memory
branches differ on the memory plant and not on the linear plant.

## Construction NMSE (not a blind eval)

| system | mismatch | A | B | C | D | ridge | strongest |
|---|---|---:|---:|---:|---:|---:|---|
| C0 | none | **0.00013** | 0.00014 | 0.00013 | 0.00013 | 0.00057 | A |
| C1 | dynamics (cubic damp) | **0.00011** | 0.00011 | 0.00011 | 0.00011 | 0.00062 | A |
| C2 | memory | 0.723 | 0.534 | **0.233** | 0.233 | 0.626 | C |
| C3 | observation drift | 0.375 | 0.382 | 0.288 | 0.288 | **0.123** | ridge |
| C4 | memory, other init mem | 1.236 | 1.043 | 1.308 | 1.308 | **0.877** | ridge |
| C5 | combined / stress | 0.362 | 0.429 | **0.217** | 0.271 | 0.353 | C |

C−A (construction, same covering plan):

- C0, C1: 0 (open constructors correctly stay on `nl2`)
- C2: **−0.490** (C selects `memory3`; A cannot)
- C3: −0.086 (C still selected `memory3`, not `obs_drift`)
- C4: **+0.071** (C selected `obs_drift` on a memory plant — wrong structure)
- C5: −0.145 (C `memory3` better than 2-D; exact recovery not required)

D−C (feedback under the *same* constructors): ≈ 0 on C0–C4; **+0.054**
(worse) on C5. Sequential revision did not add holdout precision once
the constructor set was shared.

B−A (active sampling inside the 2-D library): mixed; helps C2 and C4
somewhat, hurts C5. Not the main term.

## Predictions vs new labels (MEASURED)

B and D commit before the host will run that spec. Realized commitment
MSE on D:

- C0, C1, C2: small (6e-6, 4e-5, 5e-4)
- C3: 56.4 — committed `memory3` forecasts miss the drifted holdouts
- C4: 232 — committed `obs_drift` on a memory plant
- C5: 0.063

So “propose → commit → intervene” ran. On C3/C4 the committed
structure was the wrong one; the later labels did not repair it inside
budget 8.

## Memory diagnostic (the named hard case)

Holdout items `memory_branch_pos` / `memory_branch_neg` share
measurable `(x,v)=(0.5,0)` and the same future input; they differ in
hidden memory. On C2:

| policy | ordinary sine/step/chirp NMSE | +mem / −mem NMSE |
|---|---|---|
| A (`nl2`) | 0.61–0.98 | 0.53 / 0.73 |
| C/D (`memory3`) | **0.0006–0.0017** | **0.59 / 0.57** |
| ridge | 0.41–0.96 | 0.43 / 0.85 |

Adding a memory state repairs ordinary driven trajectories and still
fails the two branches that start from the same visible state. The
candidate always rolls out from `m0=0`. That is a structure win
without a full initial-state win — report both.

C4 is the same plant with `init_mem=1.6`. Open search picked
`obs_drift`. Ridge is strongest. Wrong structure is retained.

## Challenge target (input criterion, not a result)

Predeclared: D vs the strongest locked baseline, ≥20% relative NMSE
drop on ≥2 mismatch locations; no-mismatch median not worse than 5%.
Near-zero C0/C1 use absolute differences (already ~1e-4).

On this construction set:

- vs A: D wins C2 (large) and C5 (modest); ties C0/C1; **loses C4**
- vs ridge (often the strongest): D **loses C3 and C4**
- D−C is not a scheduling win

**The 20% challenge is not met.** Construction of a memory state
helped one mismatch location against the 2-D library and failed
another copy of the same plant. Observation drift was not identified;
a non-symbolic residual model beat every symbolic family there.

Do not hide that. Do not retune the scorer on these six systems.

## Neighbour systems (not reused)

| system | status here |
|---|---|
| LLM-ACES official[5] | repo exists; needs PySR + LLM; **not run** |
| LLM-AutoSciLab[6] | repo exists; needs LLM + PySR; **not run** |
| PySINDy / PySR | not installed; sklearn Ridge used as the non-symbolic arm |

## Keep / next lock

Keep: the apparatus (plant, API, commit-before-labels, locked
holdout, isolation tests) and the construction fact that **adding a
state can beat a 2-D library on one memory task**.

Stop claiming: sequential D over C; a 20% win; transfer; LLM-ACES
parity.

Next lock, if any, is still the TASK.md sequence: 12 development
systems then 24 frozen blind systems. Construction numbers do not
become that eval. Fixing C4 (wrong family) by peeking holdout tags
is forbidden. A legal next implementation change is: estimate hidden
`m0` from a pre-drive, or make `obs_drift` compete on calibration
rather than on driven residuals — declared before the 12-system lock,
not fitted on these six.

## Sources

[1] https://arxiv.org/abs/2606.25039 — LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search
[2] https://arxiv.org/abs/2605.24043 — LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs
[3] https://arxiv.org/abs/2404.18400 — LLM-SR: Scientific Equation Discovery via Programming with Large Language Models
[4] https://doi.org/10.1073/pnas.1517384113 — Discovering governing equations from data by sparse identification of nonlinear dynamical systems (Brunton, Proctor, Kutz 2016)
[5] https://github.com/scientific-discovery/LLM-ACES — Official LLM-ACES GitHub repository
[6] https://github.com/scientific-discovery/LLM-AutoSciLab — Official LLM-AutoSciLab GitHub repository
