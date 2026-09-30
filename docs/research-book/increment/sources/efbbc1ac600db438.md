# 来源快照

原始路径：`/srv/slam-research/grf/research-skill-comparison-20260930T065500/batch2-20260930T074146Z/RUNTIME_CORRECTION.md`

# Per-run evaluation clock correction

Recorded before inspecting any replacement or remaining-slot scientific score.
No replacement program or evaluator retry is allowed.

Installed Shinka's `async_runner.py` sets `AsyncRunningJob.start_time` to
`proposal_started_at` (line2948). Its `launch/scheduler.py:349-357` compares that
field against the configured evaluation timeout. Slot1 was submitted for actual
evaluation at16:03:21 but killed16:04:01; slot2 was submitted16:06:18 and killed
16:07:01. About140 seconds of model generation was incorrectly subtracted from
each180-second evaluation allowance. Neither has a complete metrics.json. They
remain consumed infrastructure-invalid slots, not measured algorithm defeats.

The coordinator was briefly paused during diagnosis and resumed. For the same
original remaining slots3/4, `timing_guard.py` pauses only our verified coordinator
PID1485586 while its expected evaluation child completes. The unchanged canonical
evaluator, not the paused coordinator, still enforces the actual180-second whole
evaluation allowance and CPU1. The guard resumes the coordinator on exit/failure.
It changes no package source, task config, proposal, metric, data or old artifact.
No new search slots or failed-program hand edits are introduced. Its receipt
records the exact affected slots, process identities and coordinator pause times.

Upstream completion logs may still print the misleading combined proposal+eval
timeout after resumption; completed artifacts and evaluator exit status distinguish
this from an actually unfinished evaluation. Framework/provider retries, if any,
must be logged separately from the four algorithm slots; the workaround itself
does not retry evaluations or replace slots.
