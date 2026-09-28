# Correction-source-disjoint paired evidence

Locked design before computing new photometric evidence or replay outcomes.
Host liekkas; development scan24/37; exposed replay scan55/65/69.

## Hypothesis and primary comparison

New source pixels might separate beneficial corrections from plausible but harmful
ones. Reuse the exact fixed A/identity point pairs from the preceding run.
Primary: X64+reserved paired evidence versus X64+refit paired evidence, using the
same 32-feature schema, model, target, labels, weights and calibration procedure.
This distinguishes extra view information from merely rescoring full point pairs
instead of using the previous interpolated anchor evidence. Also retain X64 and
X64+both. The predeclared lead is reserved_aug balanced, not a replay-picked winner.

## Views and geometry

Read the archived construction's five IDs: reference plus four fitting sources.
Rank all remaining cameras by number of native ROI points in image (positive depth,
8-pixel margin); require at least20, break ties by descending filename exactly as
the original adapter. Select the first four. Do not use photo costs or GT to rank.
All conditions of an ROI share this native-input-derived list. Assert reserved IDs
exclude all five originals; preserve image hashes, names and calibration hashes.
Missing four cameras is a reported unavailable case, not a replacement by used IDs.

Recompute PCA k24 normals on each full input point cloud, then evaluate both the
exact incumbent and A point with that SAME normal and 7x7 reference image patch at
half image resolution, using the frozen direct_evidence implementation. A's ray
offset must reconstruct its saved coordinates to tolerance1e-8 mm. No interpolation
of new scores. Score four old sources and four reserved sources identically.
This is source-disjoint from local correction, NOT from the original reconstruction,
camera calibration or shared reference. No conditional-independence guarantee.
No visibility oracle: invalid patches are explicit missing data, not good evidence.

## Features and methods

Each cohort contributes32 features: four incumbent costs, four proposal costs,
four paired ZNCC gains, four paired-valid flags, four incumbent-valid flags,
four proposal-valid flags; paired mean/min/max/std gain, positive fraction,
valid fraction, paired mean incumbent cost, paired mean proposal cost.
Costs are1-ZNCC in[0,2]. Missing costs1, missing gains0 and validity flags0.
Aggregates use paired support.

Four one-head shallow HGB models learn g/(e0+e1+0.01), with the preceding run's
exact parameters and case weights: cached64, fit_aug96, reserved_aug96, both_aug128.
Same original79594 development rows and78598 calibration-support rows. Two scene
folds; same101 quantile threshold grid and explicitALL/KEEP endpoints. Report
balanced, natural0, native-priority (native MSE≤identity and ±3 recovery≥5%).
Reuse preceding calibration function unchanged. Infeasible priority is KEEP,
not a success. New thresholds/models sealed before replay inference.

Fixed untrained arms: mean paired gain>0 with≥2 valid sources, separately fit and
reserved. Also intersect each gate with the previous normalized balanced mask.
Retain identity, A_all, frozen_gain and previous normalized balanced.
No reference information or scene/condition IDs in inference inputs.

## Queue and endpoints

Feature-extract24 development cases on archived sampled row IDs, and all60 replay
cases on full point rows. Train only development labels. Do not recalibrate on
replay. For each replay case export selected geometry and save all masks/scores.
Evaluate using the sealed previous run's fixed native support and reference-distance
arrays (valid because the A/identity coordinates are unchanged). PrimaryMSE,
MAE/P95, edit fraction, improved/harmed point fractions and ROI win counts.
Native and four injected conditions remain separate. No point-cloud topology,
completeness or physical layer-identity claim from fixed-row distance alone.

Add ten count/displacement-bin matched random controls for reserved_aug balanced
(support-matched only in evaluator; not a deployable rule). Identity/A/frozen and
previous normalized metrics must reproduce previous values. Per-point binary selection
problem does not imply global architectural optimality. The A/KEEP oracle is only
a ceiling for the current fixed candidate set and separable evaluation loss.

Stop after this queue and independent audit. Do not add an unplanned selector
because a result disappoints. A negative result can motivate visibility or new
candidate geometry next, but is not proof that new information can never help.
