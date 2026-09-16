# Formulation

Research object: `ResearchPolicy` (how an episode chooses the next experiment or revision).
Evaluation unit: one `ResearchEpisode` on a newly generated finite discovery task.
Question: how to discover and revise representations, objectives, and algorithms by experiments.

This file states the problem. It does not claim that the toy episode is general science.

## Unknowns (explicit)

At the start of an episode the learner does not know:

- `Representation`: which function family can express the hidden target.
- `OptimizationObjective`: which observable loss matches the external purpose.
- `EffectiveAlgorithm`: which completion of a family meets the external purpose.

New families, new observations, and new experiments are allowed. A fixed hypothesis library is not required by the task definition. The executable comparison uses a small discrete library so that policies can be run exactly; that is an implementation restriction, not a claim that science has a fixed library.

## External purpose versus proxy

External purpose (fixed, never rewritten to manufacture success):

> minimize L1 error of the delivered function on the whole finite domain `{0,...,15}`.

Proxy that a policy may optimize instead:

> L1 error on `{0,...,7}`.

These two quantities are different. A constant `2` is perfect on the proxy and has external L1 `28` (MEASURED in `RESULTS.json`). Optimizing the proxy is therefore not success.

## Information and actions

Initial information is the same for every policy: the domain, the budget, the action names, and the fact that queries return the true `y = f(x)`. The hidden table `f` is not given.

Permitted actions in an episode (subset of `TASK.tla`):

| Action | What it changes | Cost in the experiment `[ASSUMED]` |
|---|---|---|
| `Observe` | adds `(x, f(x))` | 1 per queried point; validator scan billed as 1 |
| `ConstructAlgorithm` | chooses hole values in the current family | 1 |
| `ReviseRepresentation` | replaces the family | 1 |
| `ReviseOptimizationObjective` | changes which observed points define the inductive loss | 0 extra beyond the observations |
| `SelectNextAction` | records a stall | 0 |

Terminal result of an episode: delivered family and holes, external L1, proxy L1, billed cost, trajectory.

Resource constraint: at most 8 billed steps. This is a shared step budget, not equal wall-clock time.

## Equivalence claim (stated scope)

Counterexample-guided inductive synthesis (CEGIS) is a **restricted instantiation**, not an equivalence between open-ended research and program synthesis.

Kept:

- a candidate is produced from a finite observation set;
- a validator may refute it relative to a stated specification;
- a counterexample extends the observation set;
- if the current sketch cannot fit the observations, the representation may be revised.

Dropped / not claimed:

- SAT encodings, bounded Sketch semantics, or Sketch syntax;
- that a validator for the external purpose always exists;
- that a few counterexamples suffice outside this episode (Solar-Lezama’s Bounded Observation Hypothesis is empirical for sketches, not a theorem about science);
- that CEGIS is the same as CEGAR. CEGAR refines an abstract *model*; CEGIS treats a finite input set as an abstraction of the *input domain* (Solar-Lezama, *Program Synthesis by Sketching*, 2008, §4.2). That is an analogy with a stated mapping, not identity.

## What this episode is not

- Not a context ablation on prior-project context. `TASK.tla` requires a new host session per run for that claim; this session does not supply it.
- Not a demonstration that “preparing a clean workspace” finished any ablation.
- Not a claim that one domain’s algorithm gain is a research-policy gain without comparison. The comparison is among policies on this episode only.
