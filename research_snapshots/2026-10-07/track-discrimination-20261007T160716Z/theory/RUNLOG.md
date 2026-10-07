# Run log

- First run: protocol already sealed at SHA-256 `bb09e651d986015ad06d31bebba31f79aeadfec6d494e344ff676c9214d0b7e3`; 11 unit tests passed; 12 cases completed in 1.647 s. Positive: star keeps / common track improves +2.4 for each seed; repeated texture both keep; wrong-layer occlusion both accept harmful -4.2; shared calibration error both accept harmful -5.28.
- Numerical hardening after first run: changed only affine gain arithmetic to exact rational operations on represented float inputs and outward rounding at return. Added rounding and severe-cancellation tests. No scene, motif, threshold, tolerance, candidate, seed, matching or set construction changed. Rerun is for arithmetic hardening, not parameter selection. Final code hashes and runtime are in `results/results.json`.
- No real candidate, GT, historical outcome or evaluation file was read. Only the run-root AGENTS.md and PROTOCOL.json were consulted.
