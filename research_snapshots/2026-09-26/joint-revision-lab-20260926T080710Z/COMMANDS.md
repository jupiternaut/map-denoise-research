# Commands

Host: liekkas. Workspace: `/home/grf/Documents/Codex/2026-09-26/joint-revision-lab-20260926T080710Z`.
Interpreter: `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
For every command use `-B`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

```bash
python -B -m unittest -v test_router.py test_support.py test_evaluation.py
python -B extract_support.py --scene 24
python -B extract_support.py --scene 37
python -B extract_support.py --scene 55
python -B extract_support.py --scene 65
python -B extract_support.py --scene 69
python -B train.py
python -B infer.py --scene 55
python -B infer.py --scene 65
python -B infer.py --scene 69
python -B evaluate.py
python -B audit.py
python -B plot.py
python -B verify.py --seal
python -B verify.py
```

Development extraction precedes fitting; replay extraction can run concurrently.
All models are sealed before replay inference; all inference is sealed before evaluation.
Generation commands create exclusive outputs and should not be repeated inside the sealed
workspace. Reproduction uses a fresh directory and the recorded previous workspace/data
dependencies in common.py; this is not yet a portable standalone package.

No GPU, installs, downloads, Git operations or deployment changes.

`verify.py --seal` is a one-time finalization command. In this completed workspace,
use only `verify.py` for integrity checks; it does not regenerate results.
Scientific figure: `figures/joint_revision.png` (also PDF/SVG), with plotted values
and their independent numerical checks in the same directory.
