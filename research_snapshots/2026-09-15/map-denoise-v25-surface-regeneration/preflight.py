#!/usr/bin/env python3
"""Read-only V25 handoff preflight; does not install, reconstruct, or claim GPU ownership."""

import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = Path('/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration')
OLD = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
DATA = Path('/srv/slam-research/grf/map-denoise/datasets')
DOCS = ('AGENTS.md', 'AGENT.md', 'START_HERE.md', 'TASK.md', 'research_plan.md',
        'DATA_SOURCES.md', 'ACCEPTANCE.md', 'GPU_POLICY.md', 'CHECKPOINT.md',
        'README.md', 'LONG_RUN.md', 'TASK_QUEUE.md', 'IDEA_LAB.md',
        '.cursor/rules/v25.mdc')


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def main():
    actual = Path(__file__).resolve().parent
    host = socket.gethostname()
    if host != 'liekkas' or actual != EXPECTED or Path.cwd().resolve() != EXPECTED:
        print(json.dumps({'status': 'TARGET_MISMATCH', 'host': host,
                          'script_dir': str(actual), 'cwd': str(Path.cwd()),
                          'expected': str(EXPECTED)}, indent=2))
        return 2
    required = [EXPECTED / f for f in DOCS]
    required += [OLD / f for f in ('field_budget_v24/REPORT.md', 'loss_alignment_v23/REPORT.md',
                                   'loss_alignment_v23/calibration.py', 'reconstruction_v22/operator.py')]
    inputs = {}
    for scene in (24, 37):
        images = DATA / f'loss-alignment-v23/scan{scene}/image'
        names = sorted(p.name for p in images.glob('*.png'))
        required += [DATA / f'loss-alignment-v23/scan{scene}/sparse/0/{f}'
                     for f in ('cameras.bin', 'images.bin', 'points3D.bin')]
        inputs[str(scene)] = {'images': str(images), 'count': len(names),
                              'names_0000_to_0048': names == [f'{i:04d}.png' for i in range(49)]}
    required += [DATA / f for f in (
        'real-closure-v21/cameras_geosvr_linked.npz', 'published-outputs-v1/scan24_mesh.ply',
        'reconstruction-v22-scan37/cameras.npz', 'reconstruction-v22-scan37/scan37_mesh.ply',
        'published-outputs-v2-reference/stl024_total.ply',
        'published-outputs-v2-reference/ObsMask24_10.mat',
        'reconstruction-v22-scan37/stl037_total.ply',
        'reconstruction-v22-scan37/ObsMask37_10.mat')]
    missing = [str(p) for p in required if not p.is_file()]
    ok = not missing and all(x['names_0000_to_0048'] for x in inputs.values())
    gpu = shutil.which('nvidia-smi')
    report = {
        'status': 'INPUT_PATHS_READY' if ok else 'INPUTS_INCOMPLETE',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'host': host, 'workspace': str(actual),
        'missing': missing, 'images': inputs,
        'disk_free_GiB': round(shutil.disk_usage(actual).free / 2**30, 2),
        'python_running': sys.executable,
        'read_only_legacy_python_exists': Path('/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python').is_file(),
        'colmap_in_PATH': shutil.which('colmap'), 'uv_in_PATH': shutil.which('uv'),
        'gpu_policy': 'Single snapshot only; read GPU_POLICY.md. No launch, lock, or idle guarantee.',
        'method_status': 'V25_NOT_IMPLEMENTED_BY_HANDOFF',
        'scope': 'Path/name/tool probes only; no image decode, reference evaluation or reconstruction.',
    }
    if gpu:
        report['gpu_snapshot'] = probe([gpu, '--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'])
        report['compute_processes'] = probe([gpu, '--query-compute-apps=pid,used_memory',
                                             '--format=csv,noheader,nounits'])
    else:
        report['gpu_snapshot'] = {'error': 'nvidia-smi not found'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
