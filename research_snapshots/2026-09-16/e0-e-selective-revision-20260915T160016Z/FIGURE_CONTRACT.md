# Evidence figure contract

Surface: reproducible local experimental PNG/PDF exports attached to the Markdown
research report, not an ad-hoc browser chart or hosted dashboard. Figures read
frozen results only. No method or threshold is chosen by plotted scores.

1. `mae_comparison`: horizontal signed bars, four panels (development/locked
   same-family × 1%/10% large-error fraction), exch alpha_tgt=.2. Six methods plus
   identity zero reference; mean of three seeds with individual seed dots.
   Question: which modules change final MAE? Zero line and common x scale.
   Palette: blue root plus neutral; methods identified by y labels, seeds by dots.
2. `veto_accounting`: two panels, same-family seeds201–203, exch alpha_tgt=.2,
   f_big=.01/.10. Loss contributions by all five actual source groups, TEST versus
   TEST+VETO; every contribution includes its actual n/N weight.
   Question: which subgroup benefits pay for which harms? Blue/open orange plus
   neutral zero. Not population-general attribution.
3. `paired_evidence`: grouped bars showing real-back/ghost final MAE by weak-valley
   amplitude, noise=.03, both seeds501/502, fixed TEST+VETO. Same output gets two
   different evaluation worlds. All four amplitudes shown; not a physical image
   renderer. Companion two curves use the first saved row at amplitude0 and.3,
   seed501/noise.03, chosen by fixed index rather than success.

Titles descriptive, units mm, cohorts/seeds/denominators in captions. Consistent
fonts, explicit palette, no red/green dependence. Inspect PNG before delivery.
No institutional logo or mark: these are third-party experimental artifacts.
