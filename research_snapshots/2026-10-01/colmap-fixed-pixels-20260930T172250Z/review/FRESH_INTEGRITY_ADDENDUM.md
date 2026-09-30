# Fresh integrity follow-up snapshot

Verdict remains **WARN**; `review_independence=same-family`; `acceptance_status=provisional`.

This addendum records the final reporting changes made while the first review was being completed. It does not replace the original review snapshot or alter code, predictions, metrics, or the original seal.

**PASS — scoped CSV correction.** The original `ROI_METRICS.csv` ambiguity is preserved and disclosed; the derivative `ROI_METRICS_SCOPED.csv` fixes its reporting semantics. Direct verification of all 96 rows confirmed unchanged original error/count fields, explicit preservation of method-wide coverage/yield, and correct scope-specific coverage/yield. Correction hashes match both CSV files. This is already described in finding 1 of FRESH_INTEGRITY.md. All 6,144 POINT_METRICS rows agree with the independently recomputed point distances. The report and both PNGs accurately present the main fixed-497-point comparison and its limitations.

**PASS — final report qualifier.** `REPORT.md:90` now explicitly discloses that the current RUN_LOCK did not directly bind imported evaluator code and GT provenance JSON, records any supplemental binding as post-run, retains WARN/provisional/same-family, and recommends locking the full dependency set before a future run. These statements match the audit evidence. All earlier numerical claims remain unchanged. Only REPORT.md and COMMANDS.md differ from the first review's artifact snapshot; every other listed artifact hash still matches.

**PASS — additional provenance evidence, without retroactive locking.** Direct inspection of `/srv/slam-research/grf/map-denoise/runs/upstream-photo-holdout-20260930T113213Z/SEALED.json` confirms that its `files` map binds `evaluate.py` to `46003825691f594c50f389cdd13f4f495dbfe9bb30fa5c8c6097c5ac55c44a44` and `audit_sources.py` to `877459dcaad2ba1b5affcc250020817f2785471b860de2159c02552e56092cad`; both match current bytes. The warning is therefore specifically about the current evaluator's incomplete enforced dependency chain, not the absence of any older source seal. The current evaluator checks the camera-replay prediction seal (`evaluate_mvs.py:26`), not this upstream source seal.

**PASS — pinned environment instructions.** `COMMANDS.md:7` now installs `requirements.txt`; its seven package/version pins exactly match the independently inspected installed environment. `COMMANDS.md:23` explicitly locates new replay directories under this project's runs directory, preserving run_colmap.py's sibling OLD/BASE references. These commands were inspected, not executed again.

| Final artifact | SHA-256 |
|---|---|
| REPORT.md | b0477c70920d3e6e7946fa236824cba2b841b77216b25e5e4dedac4182f583ef |
| COMMANDS.md | 41a812a4bea17d17a7db770f92799641358fbc19c19825245a8ce4879b497d2e |
| requirements.txt | 2113439ed80cd490f8b12009984b1f3c6cab21cda42794617361bd8f965e2f34 |
| review/FRESH_INTEGRITY.md | badcc617cb637d56d1b0f30089ba9bcc168c0157a0a31bb225d9020844ef506c |
| review/FRESH_INTEGRITY.json | 900a3a8902e45e40f8a3a87fae91f38561024e2761f86f9d78973f7da4bed6ef |

The earlier review's REPORT.md and COMMANDS.md hashes identify the inspected earlier versions. This addendum supplies the hashes of their later reviewed versions; it does not imply the earlier snapshot still matches those two live files. No new numeric or execution failure was identified.
