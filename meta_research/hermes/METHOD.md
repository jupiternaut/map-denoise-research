# Method

Recommended research policy for this task: **counterexample-guided inductive synthesis with an optional representation revision**, evaluated against policies that freeze the family, chase a proxy, or query by a fixed schedule.

## Source

Armando Solar-Lezama, *Program Synthesis by Sketching*, PhD thesis, UC Berkeley, 2008.  
Public copy used: `https://people.csail.mit.edu/asolar/papers/thesis.pdf` (retrieved this session, 214 pages).

The synthesis problem is written as finding controls `φ` such that `∃φ ∀σ Q(φ,σ)` (thesis eq. 4.1.2). CEGIS avoids expanding the universal quantifier over the whole input space. Each iteration:

1. **Inductive synthesis.** Find `φ` consistent with the current observation set `E`.
2. **Validation.** Check `φ` against the specification on the remaining input domain. If it holds, stop. If not, return a counterexample input `σ`.
3. **Extend `E`.** Add the counterexample and repeat. If no control fits `E`, the sketch has no solution in that family.

The thesis notes an intellectual debt to CEGAR: viewing `E` as an abstraction of the input domain (Ch. 4, §4.2). Clarke et al. TACAS 2000 was not retrieved this session (HTTP 404/429); the CEGAR mapping is taken only from Solar-Lezama’s statement, not from the original TACAS paper.

Bounded Observation Hypothesis (thesis, empirical, not a theorem): a small set of inputs covering corner cases can determine a valid control. This episode tests whether that pattern helps when representation and objective are also unknown. It does not re-prove the hypothesis.

## Algorithm used here

Families `[ASSUMED]`:

- `constant`: `f(x)=k`
- `linear`: `f(x)=ax+b`
- `piecewise`: `f(x)=k` if `x<t` else `ax+b`

Hidden target `[ASSUMED]`: `f(x)=2` for `x<8`, else `x-6`. Independent enumeration: piecewise has 2 exact completions; linear has 0; best linear external L1 is 28.

Inductive synthesizer: exhaustive search over the discrete hole set. Prefer exact fit on `E`; otherwise minimize L1 on `E`, then proxy L1 as a tie-break `[ASSUMED fallback]`.

Validator: largest residual on unseen domain points `[ASSUMED adversarial oracle]`. Cost: 1 per scan. This is stronger than a cheap proxy test.

Representation revision: if `E` has no exact fit in the current family and budget remains, move `constant → linear → piecewise`.

External utility `[ASSUMED linear combination]`: `U = -L1_external - 0.25 * billed_cost`. Ranking is reported on this `U` and on raw L1 separately.

## Policies compared (same initial information, same 8-step budget, same external L1)

| Policy | What it may revise |
|---|---|
| `fixed_proxy_linear` | nothing; fits linear on `{0..7}` |
| `random_then_linear` | nothing; fixed query schedule, then linear |
| `proxy_then_holdout_linear` | objective only (proxy → all observed points) |
| `cegis_linear_fixed` | algorithm (holes) via CEGIS; family frozen |
| `cegis_piecewise_fixed` | algorithm via CEGIS on an adequate sketch |
| `cegis_refine_families` | family then holes, starting from `constant` |

## Measured outcomes (`RESULTS.json`)

Independent checks: all passed. Runtime ≈ 0.017 s, CPython 3.11.15.

| Policy | family delivered | external L1 | proxy L1 | cost | U | perfect |
|---|---|---:|---:|---:|---:|---|
| `cegis_piecewise_fixed` | piecewise `(8,2,1,-6)` | **0** | 0 | 6 | **-1.50** | yes |
| `cegis_refine_families` | linear `(0,2)` | 28 | 0 | 8 | -30.00 | no |
| `fixed_proxy_linear` | linear `(0,2)` | 28 | 0 | 8 | -30.00 | no |
| `proxy_then_holdout_linear` | linear `(0,2)` | 28 | 0 | 8 | -30.00 | no |
| `random_then_linear` | linear `(0,2)` | 28 | 0 | 8 | -30.00 | no |
| `cegis_linear_fixed` | linear `(1,-6)` | 36 | 36 | 8 | -38.00 | no |

Trajectory of the successful policy: seed `x=0`, synthesize piecewise `(6,2,0,2)`, counterexample `x=15`, synthesize `(8,2,1,-6)`, validator accepts. Two observations plus two syntheses plus one validation: billed cost 6.

## When the recommended policy helps, and when it fails

Helps when:

- the current sketch can express the target;
- a validator exists for the *external* purpose, not only for a proxy;
- counterexamples are added to `E` rather than used as a one-step score.

Fails when:

- the family is incomplete (`cegis_linear_fixed` ends at L1 36; CEGIS does not invent a new syntax);
- representation revision is allowed but the remaining budget cannot finish the next family (`cegis_refine_families` spends a step moving constant→linear, then stalls on `sketch_unsat_no_refine` before piecewise);
- only the objective is revised (`proxy_then_holdout_linear` sees holdout `x=12,y=6` and still returns the same linear `(0,2)`);
- the proxy is treated as the purpose (`fixed_proxy_linear`: proxy L1 0, external L1 28).

So CEGIS is not a general research improvement. On this episode it is better than proxy chasing *if and only if* an adequate representation is already in play or can be reached inside the budget.

## Evidence boundary

Toy success of `cegis_piecewise_fixed` is not a general research-policy gain. It is a comparison among six policies on one assumed episode.

Preparing this workspace is not a completed context ablation. `TASK.tla` would require new sessions that differ only in prior-project context; the host did not run that.

CEGAR original paper was not obtained; no claim is made about Clarke et al. beyond Solar-Lezama’s citation.

`TASK.tla` was not model-checked.
