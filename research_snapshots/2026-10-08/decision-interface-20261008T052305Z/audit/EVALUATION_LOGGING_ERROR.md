# Evaluation stdout serialization defect

B2 evaluation saved ROWS/SUMMARY/GATE/RESULTS and their SEAL, then failed at the final stdout print because two comparisons remained numpy.bool_ and raw json.dumps did not call clean(). The saved artifacts use dump(clean(...)) and are complete. The original failed phase receipt is retained.

Independent metric reconstruction in REPLAY_RECHECK.json passed every per-row/summary/gate check and 324 old-control decisions exactly. Four unrelated early checker errors concern older SEAL schema assumptions, being corrected in the checker rather than any experiment source. No prediction, threshold, calibration parameter, input or evaluator arithmetic is modified.

record_evaluation_completion.py checks the exact observed exception, required outputs and hashes, then writes a new receipt identifying the stdout defect. It does not hide the nonzero process exit. Frozen experiment.py is retained; B3 may reproduce the same post-save printing error, in which case the same receipt checker applies. Reproduction commands should expect that cosmetic exit after inspecting sealed outputs, not blindly rerun into existing paths.
