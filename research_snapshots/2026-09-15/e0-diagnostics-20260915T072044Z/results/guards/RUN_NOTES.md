# Experiment C: completed guard diagnostic

Executed on verified host `liekkas`, in the exact diagnostics directory, using
the existing Python environment with bytecode disabled and CPU thread cap 2.
The original E0 source was loaded read-only through `common.load_old()`.

## Commands actually executed

Working directory:
`/home/grf/Documents/Codex/2026-09-15/e0-diagnostics-20260915T072044Z`

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B -m unittest discover -s tests -p test_guards.py -v
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B guards.py
```

Ten unittest methods passed, including full parameter-matrix subtests. An
additional read-only artifact audit checked all 54 NPZ files, all 216 CSV rows,
and exact saved-array identity in all 9 observationally paired worlds, with
different saved truth arrays.

The script refuses to overwrite an existing nonempty output directory. A
deliberate rerun must select a fresh directory below `results/guards` using
`--output results/guards/<fresh-run-name>`.

## Results

There are 54 cases: 36 independently generated regular observation fixtures,
9 alternative-truth reuses of the layer fixture, and 9 same-XY collision
positive controls differing by random translation. This is not 54 independent
scientific scenes. Each regular case has 400 points; each collision control
has 2 points and 1 moved endpoint.

The observed-gap guard's results, aggregating all spacings and seeds:

| Fixture | Alarms / proposed moves | Source-surface MAE before | Proposal MAE | Guarded MAE |
| --- | ---: | ---: | ---: | ---: |
| Wrong merge of real layers | 3,600 / 3,600 | 0.050346 mm | 3 mm | 0.050346 mm |
| Correct ghost merge, identical observations | 3,600 / 3,600 | 2.999590 mm | 0 mm | 2.999590 mm |
| Within-layer noise repair | 0 / 3,600 | 0.049890 mm | 0 mm | 0 mm |
| Rigid translation | 0 / 3,600 | 0.300000 mm | 0 mm | 0 mm |
| Smooth tilted plane repair | 0 / 3,600 | 0.050128 mm | 0 mm | 0 mm |
| Same-XY collision control | 9 / 9 | 0 mm | 3 mm | 0 mm |

Both guards fire on all 9 same-XY collision controls. The old guard fires on
no regular grid case at h=0.8 or h=1.6. For fixed distinct XY grid points,
3D separation is bounded below by XY separation and hence by h, regardless of
z-only motion. Therefore old NN<0.4 is physically unreachable at h=0.8.

At h=0.4, the old guard flags 364/400 moved points in each real-layer merge
case and the identical ghost-world case. This is a strict-threshold floating
point artifact: NN minimum 0.39999999999999947 mm, only
5.551115123125783e-16 mm below 0.4. These alarms are retained but cannot count
as structural detection. The observed-gap rule flags 400/400 in both worlds
at each h, as expected from its contraction criterion. See `CORRECTIONS.md`
for the one description-only correction to the original summary wording.

## Timing and limits

The bounded run took 0.395846 seconds measured inside `guards.py` from before
loading old E0 through per-case artifact writing, before writing the final
summary. This excludes interpreter startup and imports. Across all cases,
old guard calls totaled 0.016921 seconds and new guard calls 0.013186 seconds;
median calls were 0.340 ms and 0.264 ms respectively. These are single-run
small-fixture timings, not a scaling or speed superiority claim. RAM was not
measured by this module, and no GPU was used.

Source-surface errors preserve point correspondence and fixed XY support;
they do not reassign a point to whichever truth sheet is nearest. Truth is
used only to evaluate output geometry. Both selectors receive only observed
geometry and prescribed motion. The observed-gap guard uses all original
candidate pairs once, flags moved endpoints, and never recomputes after veto.
Its single-pass test deliberately exhibits a contraction that can appear
after a neighbor's veto, so the rule carries no final-output safety guarantee.

Recommendation: retain the new construction as mechanism evidence for
incumbent-relative separation preservation, together with its complete false
veto of valid ghost removal. It is not sufficient as a universal structural
guard. Disambiguation needs extra assumptions or observations absent from
these paired inputs. These diagnostics do not establish an end-to-end CPR-2
success, autonomous reconstruction, real-scene transfer, or deployment readiness.
