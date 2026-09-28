# Independent implementation audit

Status: PASS. Exact target: liekkas, `/home/grf/Documents/Codex/2026-09-26/joint-revision-lab-20260926T080710Z`.

All 79,594 development rows from scenes 24/37 and all 78,598 calibration rows retain their original case and row identities. A and B losses reproduce the archived squared-millimetre gain labels. Candidate cached, reserved and new support features match their own artifacts; B does not reuse A features.

All 14 independent router tests pass. The fitted HGB initial predictions equal the independently recomputed weighted raw-gain means on 78,598 eligible rows. The shared models have 196/260 inputs; all 15 stored policy calibration summaries reproduce, and all natural policies use strict zero threshold. Actual shared-model scores and selected coordinates are exactly swap-equivariant on 1,536 sampled development rows. The audited method sources and sealed model hashes are recorded in AUDIT.json. No new replay outcomes or replay reference geometry were opened by this audit.

V28 already selected argmax(KEEP=0, post_A, post_B) from independent raw-gain regressors. This run changes the shared pair context, candidate-specific support and weighting. It does not establish a first KEEP/A/B method or literature novelty. Scenes 55/65/69 are exposed replay evidence, and scene-fold scores used for threshold selection are tuning evidence.
