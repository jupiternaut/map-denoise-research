# Preserved failed first V9 attempt

The runner stopped during generation on the first case, before loading any evaluation truth:
`TypeError: dict() got multiple values for keyword argument 'budget'`.

The search metadata already contains `budget`; the runner supplied it a second time.
Partial outputs and source snapshots are retained. They are not part of the completed V9 evaluation.
The duplicate keyword was removed in the live V9 runner; the next run uses a new directory.
