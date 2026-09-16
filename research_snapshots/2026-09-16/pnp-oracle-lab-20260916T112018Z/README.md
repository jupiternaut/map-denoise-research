# Finite exact-oracle lab

Start with REPORT.md (Chinese), THEORY.md and research_plan.md. This lab uses
bounded exhaustive program synthesis and decision-tree dynamic programming as a
surrogate for removing computational search error. It does not test P=NP.

Python standard library only for experiments and audits. No model API, GPU, install,
external service or previous research project is required. See COMMANDS.md.

Modules:
- learning.py: immutable expression programs and exact weighted ERM.
- planning.py: exact adaptive, optimal fixed subset, and prefix sensing policies.
- composition.py: fitted-program planning with evaluator-only true-goal oracle.
- tie_audit.py: posthoc, evaluator-only envelope of equally model-optimal policies.
- audit_checks.py: independent grammar, optimizer, replay and raw-row checks.
- analyze.py: raw-row summaries and chart data; chart_query.sql is actual executed SQL.

Results are under results/. SHA256SUMS.json and verification.json record provenance.
Experimental rows are not independent environments; unique target counts are in
REPORT.md and the catalogs. Original source code and result selection were not
changed to make adaptive planning win.

