# Local commands and reproducibility scope

Exact target: host `liekkas`, directory
`/home/grf/Documents/Codex/2026-09-26/boundary-cross-lab-20260926T014111Z`.
Interpreter: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
All runs use `-B`, `PYTHONDONTWRITEBYTECODE=1`, and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

The locked queue is:

```bash
python -B -m unittest -v test_candidate_contract.py
python -B prepare_data.py
python -B extract_b.py --scene 24
python -B extract_b.py --scene 37
python -B extract_b.py --scene 55
python -B extract_b.py --scene 65
python -B extract_b.py --scene 69
python -B train_cross.py
python -B infer_cross.py --scene 55
python -B infer_cross.py --scene 65
python -B infer_cross.py --scene 69
python -B evaluate_cross.py
```

Independent audit scripts, figure generation and finalization follow the data queue;
their actual checks/status are recorded in final verification. Scene scoring is
parallel and overlaps development-label preparation; training starts only after
both development evidence seals and data seal exist. Evaluation starts only after
all inference seals exist. No replay truth enters candidate extraction or selectors.

Outputs are exclusively created: rerunning generation inside the sealed workspace
must fail rather than overwrite. A future reproduction needs a separately authorized
fresh output directory, the frozen scripts and their source dependencies. This is not
yet a self-contained portable package: common.py points to three prior experiment
workspaces plus the local dataset root. Historical A/B PLYs and cached features are
reused; original candidate-construction cost is not included in new scoring timings.

No installs, new datasets, GPU occupation, Git operations or deployment changes form
part of this run. No background job should remain at final handoff.

The completed independent audit/figure commands were:

```bash
python -B audit_data.py
python -B audit_evidence.py
python -B audit_selection.py
python -B audit_metrics.py
python -B plot_results.py
python -B finalize.py
python -B finalize.py --verify-only
```

Only the final `--verify-only` command is intended for repeat execution in this sealed
directory. Audit and plot scripts also use exclusive output creation. The finalizer
reruns all six contract tests and verifies all eleven stage seals, their sources,
both preceding final seals, figure manifests, audit hashes and required output counts.
