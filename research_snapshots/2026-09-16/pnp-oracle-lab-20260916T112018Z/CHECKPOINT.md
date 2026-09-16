# Checkpoint

Host: liekkas. New workspace: pnp-oracle-lab-20260916T112018Z.

Completed: finite program synthesis, exact sensing-policy comparison, composed
learned-model planning, posthoc tied-optimum audit, raw-row recomputation, unit
tests and independent behavioral/optimization audit. See REPORT.md.

Primary counts: A27,648 rows; B3,630 solver rows and58,080 world traces; C82,944 rows;
tie audit131 reversed pairs (59 distinct model/target/budget triples). These are
not111k independent worlds or scientific tasks.

Before final handoff run `python3 verify.py`. It records29 unit tests,10 independent
check groups, a fresh-output replay and equality excluding measured timings in
verification.json; SHA256SUMS.json hashes source and artifacts. The replay never
overwrites primary data. Timings in a replay need not match first-run timings.

Read-only old checkpoint checksum:86cea9000349650e4a51c7ffa25ea3e09e9a7354bd00d5bfb932d8343ca1eb1b.
Original protocol checksum:c06d34af5ba8637f19565d46cd6d91b36376b608697870f3a5b46516a97842f0.

Scope limits: no proof of P=NP, no asymptotic solver claim, no LLM training/API call,
no physical controller, no hard realtime certificate, no point-cloud improvement.
The two new workspaces are experiment and replay outputs on the same host, not a
replacement for the user's existing studies. No GitHub push, dependency installs
or GPU processes were started. No persistent background job is intended.

Next research candidate (not executed): sample-consistent competing programs plus
budgeted new label acquisition versus strong fixed evidence plans. Do not deploy
the true-label tie oracle as an algorithm or resume the old saturated generator.

