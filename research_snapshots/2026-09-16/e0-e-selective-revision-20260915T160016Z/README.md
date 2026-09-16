# E0-E selective geometric revision

Start with [REPORT.md](REPORT.md). This is a completed bounded experiment:
48 main configurations, 9 methods, 16 paired evidence sets, and actual PLY outputs.

Core: [selective_operator.py](selective_operator.py).
Pre-run design: [PROTOCOL.md](PROTOCOL.md).
Reproduction: [COMMANDS.md](COMMANDS.md).
Numbers: [results/SUMMARY.json](results/SUMMARY.json),
[results/METRICS.csv](results/METRICS.csv).

The prototype reduces damage substantially but loses roughly half of the available
large-error repair gain. It has conditional same-family improvement at 10% large
errors, slight regressions at 1%, and does not identify physical layers from
ambiguous cost curves. Default remains identity. No real-world promotion.

No historical files, packages, GPU processes, or remote repositories were changed.
