# V24 compact comparison chart

Question: At the uncapped 0.025 mm per-patch displacement budget, how do the three
repair fields change point-to-reference MAE across the two scenes?

Takeaway: Similar movement budgets do not produce a scene-independent ordering;
distance gains must be read alongside the retained coverage-change column.

Surface: native Data Analytics grouped bar preview accompanying the experiment,
not a separate hosted dashboard or replacement for the local research report.
Six rows: scene x field; 18 or 24 patches per scene, all exposed. The first budget
is used because it was uncapped in all 42 patches, not selected for a winner.
Group on field, categorical x=scene, numeric y=MAE gain (mm), identity zero line.
Retain counts, absolute accuracy, recall, recall difference and actual budget in
the exploration table. No time axis, no point-count inference, no new confirmation.
Use native categorical palette (blue/gold/olive roots where supported), visible
field legend and explicit labels rather than relying only on colour. Exact table
in REPORT.md is the fallback if widget delivery fails. No external publication.

Provenance: chart_data.py imports the frozen per-patch RESULTS.json into an
in-memory SQLite table and executes the saved query. Output chart_payload.json
contains the query, source path, all six rows and display encodings.
