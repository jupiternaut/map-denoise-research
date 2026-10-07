# Selector attribution experiment

## Pipeline Status
language: zh
Host: liekkas
Only writable experiment root: /srv/slam-research/grf/map-denoise/runs/selector-attribution-20261007T180539Z

Goal: preserve valid geometry and repair errors. This run isolates observation construction, selection ranking and acceptance in the previous fixed-candidate development replay.

- Historical runs, candidate coordinates, camera/image evidence, dataset references and Git repositories are read-only.
- CPU only, no installs, no GPU jobs, no deployment or uploads.
- Do not repair the matching extractor, alter thresholds, add candidates, or fit to evaluation truth in this ablation.
- Design is retrospective, on already exposed scenes; do not call it preregistered blind confirmation.
- Lock the completed protocol and code before executing new arm predictions; preserve exact protocol text, not only hashes.
- Policy input cannot contain nearest-reference distances, physical labels, or evaluation fields. Seal all predictions before evaluation.
- Same fallback and current photo output for empty/unsupported/missing observations where comparisons share an input interface; report any exception explicitly.
- If old/new evidence does not support a faithful common interface, do not invent a clean causal factorial attribution percentage. Report actual conditional contrasts and path dependence.
- Tie rules must be deterministic, shared and specified; include current output as a possible KEEP where relevant.
- All arms and failures reported. No selection of winning configuration by ground truth.
- Use apply_patch for authored code/documents. New generated outputs use exclusive create where practical.
- New test failures may be fixed before lock; post-lock fixes require separate version and explicit disclosure.

Delegated reviewers must read primary files directly. Fresh integrity reviewer writes only audit/. Private review traces stay under .aris/ and are not published.
