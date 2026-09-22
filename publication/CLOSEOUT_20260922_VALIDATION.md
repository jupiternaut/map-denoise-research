# Publication validation

Target: liekkas, existing GitHub repository `jupiternaut/map-denoise-research`, branch main.

- 166 copied artifacts / 11,691,562 bytes checked against source SHA-256 and staged Git blobs; exact expected staging set checked before this validation note.
- 753 excluded artifacts listed in the manifest; old source files and unrelated untracked AgentRx files remain untouched.
- Credential-pattern scan of all staged content: no matches. This is a pattern check, not an absolute security guarantee.
- Runtime suite executed from the publication copy: 5 passed; 2 archive-replay tests skipped because V28_SOURCE was not configured. Existing saved test results remain historical.
- Evaluator contract assertions passed on read-only rerun. Initial invocation reached exclusive report creation and refused to overwrite the archived EVALUATOR_TESTS.json; rerun redirected only report emission to stdout. No archived report was replaced.
- Default Git whitespace check flags historical CSV CRLF line endings. With CR-at-EOL recognized, check passes; original bytes were retained.
- No new scientific experiment, independent environment reproduction, full dataset replay or external-baseline experiment was performed for this publication.
