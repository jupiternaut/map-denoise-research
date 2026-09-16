# E0-F incumbent-support selective revision

Start with [REPORT.md](REPORT.md) and [PROTOCOL.md](PROTOCOL.md).
Run instructions: [COMMANDS.md](COMMANDS.md).

This implements an observation-only geometric correction gate, not a neural router.
It improves same-family high-error geometry but increases wrong-layer damage and fails
the low-error non-regression requirement. It is not the new default.

Sources and raw geometry are preserved for reproduction; see SOURCE_LOCK.json,
results/RESULTS.json, results/cases and results/stress.
