# Completed checkpoint

Host `liekkas`; 2026-09-26. Status: completed exposed-scene replay; no background job.

The locked queue completed: five source-evidence batches, four normalized-gain models,
three replay scenes / 60 cases, 20 actual arms, 41 evaluated arms including diagnostics,
2,460 metric rows, 240 exported PLY files. Independent evidence, decision, metric and
figure audits passed. Final verification and immutable file hashes are recorded by
`finalize.py` in `FINAL_VERIFICATION.json` and `FINAL_SEAL.json`.

Primary reserved-source evidence beats matched fitted-source evidence on the five
aggregate conditions, most clearly on ±3 mm recovery. Relative to the prior normalized
method it improves large-offset recovery but increases native damage: native −4.516%,
−3 mm +46.773%, +3 mm +37.120% MSE reduction versus identity.

Native-priority reduces edits rather than establishing native restoration. Its reserved
variant edits zero native development rows and 61 native replay rows; replay native MSE
still worsens0.0403%. Do not promote it as a no-harm theorem or select it retrospectively
as the primary arm. No method from this run changes the deployment default.

All original A/identity coordinates, prior experiments and evaluation arrays remain
read-only. No GPU, package installation, data download, Git commit/push, thesis edits or
new-scene confirmation occurred. The data are exposed development evidence.

Architecture finding: move from candidate scoring alone to separated construction and
verification evidence. This is one tested architectural factor, not global architectural
optimization. A/KEEP cannot add missing surface samples. The full task still needs
protection plus repair, and coupled coverage/structure evaluation.

If continuing, do not add another threshold on this replay. A bounded next comparison
would cross the frozen evidence interfaces with an alternative boundary/occlusion-aware
candidate, then evaluate a frozen full pipeline on genuinely new scenes. This is a
proposal only; it was not launched by this checkpoint.

Primary entry: `REPORT.md`. Theory detail: `ARCHITECTURE_ANALYSIS.md`.
Figure: `figures/reserved_evidence_balanced.png` (PDF/SVG also available).
Dependencies and verification commands: `COMMANDS.md`.
