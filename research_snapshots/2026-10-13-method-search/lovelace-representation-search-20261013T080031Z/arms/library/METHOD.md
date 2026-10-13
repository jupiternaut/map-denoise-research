# A12 fixed-library global selector

## Proposal and information boundary

Construction clock reference: 2026-10-13 08:11:14 UTC, the first recorded clock after reading the authorized instructions. Conservative finish target: before 08:30 UTC; construction cap is 20 minutes. Host is liekkas. The logical root resolves to the same physical USB experiments directory specified in AGENTS.md.

This author is reused, not fresh. Existing context includes same-family A7 verifier work, A8 descriptive results, and A10/A11 solver authorship. No new historical score inspections are used in A12. The author reads only this root's AGENTS.md, task_bundle/TASK.md, task_bundle/forward.py, task_bundle/dev.json, library/LIBRARY.md, library/methods.py, and this arm's own files. The free arm, hidden worlds, generator, holdout and other arm outcomes are excluded. The ancestor imports required by the fixed library execute as its existing dependency chain; no ancestor source is newly inspected or modified here.

Initial proposal: choose `coarse` globally. Reason: under a small cold-start budget, fewer staged subdivisions may leave more budget for complete initial domain elimination. This is a computational hypothesis, not an observed advantage or new representation.

Before any development estimator run, freeze the selection rule: evaluate all three library names once on each of the six public development observations at six seconds per prepare. Reuse each prepared state for that observation's listed incumbents. Choose the single name with smallest pooled coordinate MSE over all 42 development tasks; exact ties or differences at most 1e-12 prefer coarse, then depth, then accumulated. Failed or hard-overbudget prepares are scored with identity fallback and retained in the results. Separately report correct-incumbent damage, errors, latency and actual CPU. A lower MSE with correct-input damage is not described as a comprehensive win. At most one change of the global constant is allowed. No 24-second development tuning or per-observation label rule is used. The frozen choice is used at both formal budget slices.

## Mathematical object and interpretation

All three choices represent uncertain depth by a finite closed union of intervals within [540,660]. The variables and photo/anchor constraints are exactly those of the fixed library; this adapter changes none of their kernels, tolerances, material/bias bounds, clipping model or certificates. The library constructs each state from the current raw photo and camera/appearance anchors. Every ancestor stage is charged within that same cold-start call; no saved support or historical fit is supplied.

Geometry decoding uses the library's HULL rule: project one incumbent into the hull of the retained support, or keep it when the support is empty. The final adapter clamps only to the public [540,660] domain, which does not alter any valid in-domain library output. Each decode receives one incumbent, never the full incumbent group. State includes the library support, trace and calibration plus the globally chosen name; it contains no development truths.

Preserved: all registered library measurement inputs and all uncertainty retained by the selected solver. Approximated: the library's existing necessary outer relaxation and finite-budget refinement. Discarded by HULL: holes inside the support. Discarded by the selector: unchosen computational schedules, not new measurements. Classification: selection among fixed approximate representations/solvers; no new physical model, new information, equivalence theorem or originality claim.

Expected benefit, if any, is budget allocation across existing representations. Applicability is limited to the given physical model and this small exposed synthetic development family. A development-selected winner is not independently confirmed evidence; only the later inaccessible holdout can evaluate transfer.

## Cost and final selection

The only development comparison ran 2026-10-13 08:17:03–08:17:56 UTC. All 18 prepares succeeded, without identity fallback or a budget-plus-0.25-second overrun. There were no saved classifier atoms longer than 0.25 seconds. The largest complete prepare took 6.028006 seconds; the deadline is cooperative, not an operating-system preemption guarantee.

| Global choice | Tasks | MSE (mm²) | MAE (mm) | Improved / unchanged / worsened | Correct damaged / 6 | Prepare CPU (s) |
|---|---:|---:|---:|---:|---:|---:|
| Identity | 42 | 20.925031597 | 3.730723810 | 0 / 42 / 0 | 0 | — |
| coarse | 42 | 5.918716570 | 1.951950283 | 24 / 18 / 0 | 0 | 11.969606 |
| depth | 42 | 5.198924367 | 1.791636156 | 25 / 17 / 0 | 0 | 17.305826 |
| accumulated | 42 | 5.212607163 | 1.795298265 | 25 / 17 / 0 | 0 | 24.396774 |

Final choice: **depth**, made by the one permitted revision from the initial coarse proposal. The exact same choice is used for every observation and both formal budgets. Depth and accumulated gave identical task outputs in five development observations; depth was better in the remaining observation. This is a small development margin, not evidence of universal or statistically established superiority. Timed finite-budget boundaries can depend on runtime scheduling. The 24-second and hidden-data performance are unobserved here.

The complete comparison, including import and scoring work, consumed 54.018555 CPU seconds and 54.055982 wall seconds. Initial four interface tests consumed 0.402788 CPU seconds including imports; final four post-selection tests and an independent recomputation from all 126 saved decoded values consumed 0.335753 CPU seconds and passed. Total measured comparison/test/check CPU was 54.757096 seconds, comfortably below the 120-CPU-second cap. Brief inspection and document editing were not estimator runs. A process CPU guard was set to a 116-second soft / 118-second hard limit; it did not trigger. Candidate order was cyclically rotated by development observation index, and each library call performed its specified cold-start cache clear.

DEV_OUTCOMES.json records all 18 per-observation trials, all decoded predictions in the development task order, timing, errors, hulls and aggregate metrics. The dev input SHA-256 is `92f64359e21b0f7dca9748175bc8b0de9971f3ad8a8ab38132f37b4826fc3e76`. The initial proposal and tie rule were written before any estimator development run. No additional method rerun, 24-second tuning, hidden or formal evaluation is performed by this author. Method selection was complete by 08:18:42 UTC, about 7 minutes 28 seconds after the first recorded construction clock and below the 20-minute cap.
