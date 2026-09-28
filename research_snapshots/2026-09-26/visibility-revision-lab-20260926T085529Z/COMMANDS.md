# Run and inspect

Host `liekkas`. Code directory is this folder. Artifacts:
`/srv/slam-research/grf/map-denoise/runs/visibility-revision-20260926T085529Z`.

Python `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python`.
Set `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```bash
python -B -m unittest -v test_visibility.py test_learner.py test_evaluation.py
python -B extract_visibility.py --scene 24
python -B extract_visibility.py --scene 37
python -B extract_visibility.py --scene 55
python -B extract_visibility.py --scene 65
python -B extract_visibility.py --scene 69
python -B train.py --arm base_shallow
python -B train.py --arm base_rich
python -B train.py --arm geometry_shallow
python -B train.py --arm geometry_rich
python -B train.py --arm interaction_shallow
python -B train.py --arm interaction_rich
python -B train.py --finalize
python -B infer.py --scene 55
python -B infer.py --scene 65
python -B infer.py --scene 69
python -B evaluate.py
python -B plot.py
python -B verify.py --seal
python -B verify.py
```

Extraction for replay can overlap development fitting. Inference requires all training
locked and its scene evidence sealed. Evaluation requires all three inference seals.
At most three single-thread CPU workers are used concurrently. Generation outputs are
exclusive; do not rerun these writes in a sealed experiment. A new reproduction needs
new output paths and the frozen prior input/data dependencies in common.py.

`verify.py --seal` is one-time finalization, after report review and all agents have
finished writing. Use only `verify.py` for a completed bundle's read-only integrity
check. Figures are PNG/PDF/SVG under the artifact directory's `figures/`.
