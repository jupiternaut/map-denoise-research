#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/srv/slam-research/grf/map-denoise/tools/gsplat-v2:/srv/slam-research/grf/map-denoise/envs/gpu-v1-overlay
export PATH=/srv/slam-research/grf/map-denoise/tools/gsplat-v2/bin:$PATH
export CUDA_HOME=/srv/slam-research/grf/map-denoise/tools/cuda-nvcc-12.1
export CXX=/srv/slam-research/grf/map-denoise/tools/gcc12-local/usr/bin/g++-12
export CC="$CXX"
export TORCH_EXTENSIONS_DIR=/srv/slam-research/grf/map-denoise/tools/gsplat-build
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.6
export OPENBLAS_NUM_THREADS=1
/srv/slam-research/grf/map-denoise/envs/pathnet-v5/bin/python /home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1/published_outputs_v2/render_room.py
