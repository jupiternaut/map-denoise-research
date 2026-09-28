# Run and reproduce

Host: `liekkas`. Working directory:
`/home/grf/Documents/Codex/2026-09-26/multisurface-field-lab-20260926T102842Z`.
Python: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
Data outputs: `/srv/slam-research/grf/map-denoise/runs/multisurface-field-20260926T102842Z`.

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -v
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B infer.py --scene 55
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B infer.py --scene 65
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B infer.py --scene 69
# Only after all three inference directories have SEALED.json:
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B evaluate.py --scene 55
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B evaluate.py --scene 65
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B evaluate.py --scene 69
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B evaluate.py --finalize
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B figures.py
```

Three single-thread scene workers may run concurrently. Existing output folders
are intentionally not overwritten. A repeat requires a new explicit OUT path
in a new code snapshot, not deletion of these records. This is a local replay
package and retains read-only imports from earlier frozen experiments; it is not
yet a standalone redistributable dataset/runtime bundle.

## Actual construction

`current input + archived A/B + calibrated photographs`
→ `9 directly scored physical ray positions`
→ `shared local K1/K2 quadratic inverse-depth fields + latent responsibilities`
→ `fresh 4-position photometry`
→ `KEEP/surface choice`
→ `candidate-conditioned visibility`
→ `local image-weighted label graph`.

Every stage emits positions/evidence before any reference is opened. The
evaluator separately computes diagnostic oracles; these never feed the methods.
The old recovery, direct point WTA, K1, K2, visibility and graph are separate arms,
not a pipeline from which the best result is picked using the reference.

## Scope

This first field prototype changes local geometry representation and assignment.
It retains the input row count and reference rays. It does not yet create new
samples, connect tiles globally, construct a watertight SDF, or identify topology.
The camera views formerly reserved from A/B construction are now fitting inputs,
so they are not held-out validation views for this new method.
