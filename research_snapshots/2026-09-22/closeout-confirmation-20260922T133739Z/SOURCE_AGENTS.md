# Closeout preparation and confirmation protocol

Host liekkas. This new workspace is writable; prior V28/data/workspaces read-only.
Current task: design and start bounded closeout work, not construct another method.
No downloads of new scene pixels/GT before a data/scene manifest and protocol lock.
No global installation, driver modification, old environment writes, GPU takeover,
process termination or GitHub publication. Package metadata/docs/network probes
are allowed. Ask root before any large transfer or installation.

Root owns protocol, data plan, readiness report and integration. Baseline agent
owns baseline_probe.py and BASELINE_READINESS.*. Packaging agent owns package/,
prepare_release.py, tests/, TRAINING_LOCK.json and REPRODUCIBILITY.md. No overlap.

Main method is frozen V28 post_A_keep; A/B/KEEP is secondary, not a replacement
selected after future outcomes. Uniform model trained on both old development
scenes is allowed, using archived training labels and fixed hyperparameters.
No new-scene GT may affect models, thresholds, ROI, candidate rank or source views.
Primary native and perturbation outcomes are separate. All rejection rows count.
This turn may complete preparation without falsely claiming independent results.
