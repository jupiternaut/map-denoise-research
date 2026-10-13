Verdict: PASS — provisional, same-family source review.

BLOCKING issues: None found.

NON-BLOCKING issues: The calibrated/shared-bias wording was corrected during review. I reread the final PROTOCOL.md: it now correctly distinguishes calibrated nine-material witnesses with marginalized T_cal from joint witnesses carrying three shared bias coordinates. Both remain necessary relaxations without camera-offset witnesses.

Verified source behavior:

- Exact z,z geometry preserves camera uncertainty; clipping, inequality directions, signed links, material bounds, and calibrated versus joint bias handling agree with the frozen implementation and independently reconstructed rows.
- All containing closed leaves are included, including shared endpoints; gaps remain gaps. Original A5 classifications are copied unchanged. Empty original support remains SOURCE_EXCLUDED.
- Global witnesses must satisfy every containing same-branch local box and original interval rows. Contradictory local/global evidence fails verification.
- Exact certificates require strict rational contradiction; witnesses satisfy every row and bound. UNKNOWN remains undecided.
- All 648 records, 980 local checks, and 1944 global checks are saved and reread before private evaluation. The verifier reconstructs evidence without importing the new solver or runner.
- Fixed reporting denominators, postgate actual-parameter coverage, exact target identity, and the shared main/verification deadline are enforced. Mechanical PASS does not require a favorable scientific result.

I ran the existing suite on liekkas, from the exact A6 root, using the specified Python: 45 tests passed in 2.242 seconds.

No files edited, private/formal data parsed, or formal batch executed. This review establishes no empirical outcome or physical feasibility.

## 宿主处理

预正式只更正PROTOCOL.md及其快照的H_remaining措辞，reviewer已重读。未改代码，未运行正式数据。完整prompt/final回应位于私有.aris/traces/source-review/2026-10-13_run01；本review为同族暂定，非外部物理模型审核。

