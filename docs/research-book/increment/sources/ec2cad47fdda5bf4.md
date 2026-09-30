# 来源快照

原始路径：`/srv/slam-research/grf/research-skill-comparison-20260930T065500/batch2-20260930T074146Z/timing_recovery/REPORT.md`

# Preserved-output recovery and scheduler clock audit

Exact host: liekkas. All new artifacts are under this timing_recovery directory.
RECOVERY_RULES.md was written before opening gen_1 score arrays. The recovery
script never imports a proposal or calls fit, predict, worker, run_evaluation or
the evaluator CLI. No replay reference, generation directory, framework source,
database, pipeline, threshold or selection rule was modified.

## Recovery outcome

gen_1 is a **valid_measurement_recovered**, separate from its historical failed
scheduler job. Root alone decides whether the recovered measurement enters
selection. It does not erase a consumed proposal slot or turn the old job into
an official successful scheduler execution.

| Condition | MSE reduction versus KEEP |
|---|---:|
| native | -1.5940981461% |
| minus3 | 35.6074518446% |
| plus3 | 31.6580035694% |
| all | 28.1028344353% |

All eight preregistered gates passed. Control D was reconstructed first from its
saved scores; combined_score, every public metric, routes and materialized points
matched the original exactly. gen_1's two SCORES archives passed full ZIP CRC and
array loading, expected shape, finite scores and scalar timing checks. Every
WORKER_INPUT field matched the frozen development data, sampling, canonical macro
weights and case boundaries exactly in shape, dtype and value. All frozen parent
hashes matched the preflight manifest. Bound inputs stayed unchanged throughout.

gen_1 source SHA256:
3864138d31fae1339c567fbae3bff67302e397c710da8ee17da54260ce157ea4.
This is byte-identical to the code stored in original DB program
8748dd84-0c51-4bd1-90ee-b8df219ac539. The archived rewrite and generated response
agree with it except for the terminal newline added during application. Source
mtime and ctime precede evaluation launch. Saved SCORES files do not embed a
source hash; provenance therefore combines archive/DB identity, timestamps and
the unchanged canonical evaluator, rather than claiming tamper-proof attestation.

Canonical results are in GEN1_CANONICAL_METRICS.json and GEN1_CANONICAL_OOF.npz;
RECOVERY_AUDIT.json records gates, provenance, all input hashes and limitations.
CONTROL_D_PARITY.json records the independent no-fit aggregation parity check.

## Timing evidence

The original database records sampling140.534662seconds and evaluator-parent
lifetime39.499484seconds. Logs show launch16:03:21 and scheduler kill16:04:01.
However, the second worker's complete SCORES file was written at16:04:32.728320,
after the parent was killed. The preserved worker continued without a new fit
request. Its original invocation had already been launched at16:03:58.

| Quantity | Seconds |
|---|---:|
| Saved two-fold fit+predict wall sum | 65.920169 |
| Launch-to-last-output conservative wall bound | 72.728320 |
| Recovery script wall / CPU before small report writes | 2.956811 / 2.956364 |
| gen_1 payload checks plus canonical aggregation wall / CPU | 1.113835 / 1.113706 |
| Saved fit+predict plus full recovery wall and1second allowance | 69.876980 |
| Original elapsed bound plus recovery wall and1second allowance | 76.685132 |
| Conservative original+recovery CPU bound | 149.413005 |

All bounds are below180seconds. Original worker CPU was not recorded directly.
The CPU bound deliberately permits two CPU-seconds per elapsed second: the
canonical single-thread coordinator plus at most one serial single-thread model
worker, then adds measured recovery CPU and a1second report allowance. This is
an upper bound, not a fabricated measured original CPU total. It includes the
orphan-worker interval after the original parent's premature termination.

gen_2 lacks the second fold payload and scores. gen_3 lacks both fold outputs.
Only their file completeness was recorded; no aggregation, fit or inference was
performed for them. Neither can be recovered as a full measurement from the
currently preserved files.

## Independent clock diagnosis and future repair recommendation

Audited installed sources are under
/srv/slam-research/grf/science-skills-20260930/envs/shinka/lib/python3.12/site-packages/shinka/.

- launch/scheduler.py349-352 uses time.time()-job.start_time for the local timeout.
- core/async_runner.py2948 assigns start_time=proposal_started_at, even though the
  job separately stores evaluation_started_at and evaluation_submitted_at.
- core/async_runner.py5113-5121 already records the evaluation timestamps after
  the local submission returns. Its own _is_job_hung at5143-5156 uses those
  evaluation timestamps first, unlike scheduler.check_job_status.
- launch/local.py delegates kill to the parent Popen. The canonical evaluator
  starts each model worker in a new session. Killing only the outer process can
  therefore leave its worker alive, as the second complete output demonstrates.

The minimal future clock correction is to make scheduler.check_job_status prefer
evaluation_started_at, then evaluation_submitted_at, and use legacy start_time
only for job types with no evaluation timestamp. Keep proposal_started_at for
sampling/pipeline accounting; do not overwrite it to hide generation latency.
For a durable implementation, capture a monotonic evaluation-start clock at
actual local process launch and use the same clock in both timeout guards.
Retain the canonical180second evaluation cap; increasing it would obscure the
bug. Poll completion before deciding to time out a job. Timeout cleanup must
terminate the evaluator's complete descendant process tree and reap it; merely
killing the outer process group is insufficient when a child creates a new
session. No such framework repair was made by this audit.

Audited file SHA256 values:

- scheduler.py: c32d0682ac0c572d01dc10b49887cb54eb7d6b6ba760ba74c132465a1366266b
- async_runner.py: 153a6e1b9a7bdb673767bd3109de085c2f4e60cae07a0107858c5318f7557fed
- local.py: 16eccbc364b642383ad8c77034a06645f1f2b7d0e9bd71862ceb5db7f99c6a48
