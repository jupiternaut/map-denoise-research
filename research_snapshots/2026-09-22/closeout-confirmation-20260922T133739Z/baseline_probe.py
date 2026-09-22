#!/usr/bin/env python3
"""Read-only, bounded COLMAP readiness audit for the exact liekkas workspace.

Print JSON; never install, download, stage images, run reconstruction, or import
GPU libraries. No evaluator reference is opened. A missing prerequisite is not
an observed baseline-quality failure. Run with python3 -B baseline_probe.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z')
V28 = Path('/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z')
DATA = Path('/srv/slam-research/grf/map-denoise/datasets/loss-alignment-v23')
ENV_ROOTS = [Path('/srv/slam-research/grf/map-denoise/envs'),
             Path('/srv/slam-research/grf/ai4s-toolkit/envs')]
PIN = {'repository': 'https://github.com/colmap/colmap', 'tag': '4.2.0',
       'commit': 'be5e29168d4aff238409d60424812df66aac919f',
       'backend_for_this_host': 'CUDA', 'installed': False,
       'provenance': 'official tag, verified by git ls-remote on 2026-09-22; not an installed build'}


def command(argv: list[str], timeout: int = 8) -> dict:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        return {'argv': argv, 'returncode': proc.returncode,
                'stdout': proc.stdout[:14000], 'stderr': proc.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'probe_error': str(exc)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bounded_names(root: Path, max_depth: int, max_entries: int = 30000) -> dict:
    """Search names only, without following directory symlinks or reading contents."""
    record = {'root': str(root), 'max_depth': max_depth, 'max_entries': max_entries,
              'entries_seen': 0, 'hits': [], 'unreadable': [], 'truncated': False}
    if not root.exists():
        record['status'] = 'ABSENT_ROOT'
        return record
    pending = [(root, 0)]
    while pending:
        folder, depth = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for item in entries:
                    record['entries_seen'] += 1
                    if record['entries_seen'] > max_entries:
                        record['truncated'] = True
                        record['status'] = 'BOUNDED_INCOMPLETE'
                        return record
                    if 'colmap' in item.name.lower() or 'openmvs' in item.name.lower():
                        record['hits'].append(item.path)
                    if depth < max_depth - 1 and item.is_dir(follow_symlinks=False):
                        if item.name not in {'.git', 'node_modules', '__pycache__'}:
                            pending.append((Path(item.path), depth + 1))
        except OSError as exc:
            record['unreadable'].append({'path': str(folder), 'error': str(exc)})
    record['status'] = 'COMPLETE_WITH_UNREADABLE' if record['unreadable'] else 'COMPLETE'
    return record


def local_software() -> dict:
    paths = [Path(p) / 'colmap' for p in os.environ.get('PATH', '').split(os.pathsep) if p]
    paths += [Path('/usr/bin/colmap'), Path('/usr/local/bin/colmap'), Path('/home/grf/.local/bin/colmap')]
    envs = []
    for parent in ENV_ROOTS:
        if not parent.is_dir():
            continue
        for env in sorted(parent.iterdir()):
            if not env.is_dir():
                continue
            paths.append(env / 'bin/colmap')
            packages = sorted(str(p) for p in env.glob('lib/python*/site-packages/*colmap*'))
            packages += sorted(str(p) for p in env.glob('*colmap*'))
            envs.append({'root': str(env), 'python': str(env / 'bin/python'),
                         'python_exists': (env / 'bin/python').exists(),
                         'colmap_package_name_hits': packages})
    scans = [bounded_names(Path('/opt'), 5),
             bounded_names(Path('/srv/slam-research/grf/map-denoise/tools'), 5),
             bounded_names(Path('/srv/slam-research/grf/map-denoise/models'), 2),
             bounded_names(Path('/home/grf/.cache/pip'), 5),
             bounded_names(Path('/home/grf/.cache/uv'), 4),
             bounded_names(Path('/srv/slam-research/grf/map-denoise/cache/uv'), 4),
             bounded_names(Path('/srv/slam-research/grf/map-denoise/cache/pathnet-v5-uv'), 4),
             bounded_names(Path('/srv/slam-research/grf/ai4s-toolkit/cache'), 4),
             bounded_names(Path('/var/cache/apt/archives'), 1)]
    for scan in scans:
        for hit in scan['hits']:
            path = Path(hit)
            if path.name.lower() == 'colmap' and path.is_file():
                paths.append(path)
    found = sorted(set(str(p.resolve()) for p in paths if p.is_file() and os.access(p, os.X_OK)))
    versions = [command([p, '-h']) for p in found[:3]]
    return {'path_lookup': shutil.which('colmap'), 'executables': found,
            'status': 'AVAILABLE_BACKEND_UNVERIFIED' if found else 'NOT_FOUND_IN_SEARCH_SCOPE',
            'help_only_probes': versions, 'research_environments': envs, 'bounded_scans': scans,
            'apt_policy': command(['apt-cache', 'policy', 'colmap']),
            'apt_package_metadata': command(['apt-cache', 'show', 'colmap']),
            'installed_debian_package': command(['dpkg-query', '-W', '-f=${Status} ${Version}\n', 'colmap']),
            'limitation': 'Opaque pip cache objects, unreadable paths, other users and arbitrary trees were not searched.'}


def gpu() -> dict:
    if not shutil.which('nvidia-smi'):
        return {'status': 'RESOURCE_STATE_UNKNOWN', 'allocation_authorized': False}
    device = command(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'])
    # Process names can contain browser command lines; retain only PID and allocation size.
    processes = command(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                         '--format=csv,noheader,nounits'])
    busy = bool(processes.get('stdout', '').strip())
    known = device.get('returncode') == 0 and processes.get('returncode') == 0
    return {'status': ('OCCUPIED_DO_NOT_LAUNCH' if busy else 'NO_COMPUTE_PROCESS_AT_SNAPSHOT') if known else 'RESOURCE_STATE_UNKNOWN',
            'device_query': device, 'process_query': processes,
            'allocation_authorized': False, 'processes_terminated': False,
            'note': 'A momentary 0% utilization or no compute PID is not permission to take a display GPU.'}


def development_inputs() -> list[dict]:
    records = []
    seen_hashes = {}
    for meta_path in sorted((V28 / 'real_results').glob('*__native/construction.json')):
        meta = json.loads(meta_path.read_text())
        names = meta['views']
        image_root = DATA / f"scan{meta['scene_id']}" / 'image'
        checks = []
        for name in names:
            image = image_root / name
            if image not in seen_hashes:
                seen_hashes[image] = sha(image) if image.is_file() else None
            checks.append({'name': name, 'path': str(image), 'exists': image.is_file(),
                           'sha256_matches_V28': seen_hashes[image] == meta['image_hashes'][name]})
        conditions_same = True
        for condition in ('plus3', 'minus3'):
            sibling = json.loads((meta_path.parent.parent / (meta['roi_id'] + '__' + condition) / 'construction.json').read_text())
            conditions_same &= sibling['views'] == names and sibling['image_hashes'] == meta['image_hashes']
        sparse = DATA / f"scan{meta['scene_id']}" / 'sparse/0'
        all_pass = len(names) == 5 and len(set(names)) == 5 and conditions_same and all(x['sha256_matches_V28'] for x in checks)
        records.append({'roi_id': meta['roi_id'], 'scene_id': meta['scene_id'],
                        'evidence_class': 'EXPOSED_DEVELOPMENT_ONLY',
                        'reference': names[0], 'sources': names[1:],
                        'image_checks': checks, 'conditions_share_images_and_view_order': conditions_same,
                        'sparse_path': str(sparse),
                        'sparse_files': {name: {'exists': (sparse / name).is_file(),
                                              'sha256': sha(sparse / name) if (sparse / name).is_file() else None}
                                         for name in ('cameras.bin', 'images.bin', 'points3D.bin')},
                        'image_manifest_status': 'PASS' if all_pass else 'FAIL',
                        'colmap_five_view_adapter': 'NOT_IMPLEMENTED',
                        'patch_match_cfg_plan': '\n'.join(line for name in names for line in (name, ', '.join(n for n in names if n != name))) + '\n'})
    return records


def stage_plan() -> list[dict]:
    case = str(ROOT / 'baseline_staging/LOCKED_CASE')
    return [
        {'stage': 'five_view_adapter', 'status': 'NOT_IMPLEMENTED', 'command': None,
         'requires': 'Locked five-view names + known cameras + exact half-resolution pixels + input-only depth bounds + point track consistency'},
        {'stage': 'image_undistorter', 'status': 'NOT_RUN',
         'argv_template': ['COLMAP_PINNED_BINARY', 'image_undistorter', '--image_path', case + '/images',
                           '--input_path', case + '/sparse_fixed', '--output_path', case + '/dense', '--output_type', 'COLMAP', '--max_image_size', 'LOCKED_HALF_RESOLUTION_MAX_DIMENSION'],
         'requires': 'Adapter round-trip projection checks; only the five selected cameras/images and tracks triangulated from these five photos in sparse_fixed; resolved version/help'},
        {'stage': 'lock_patch_match_config', 'status': 'NOT_RUN', 'command': None,
         'requires': 'Write explicit names (never __auto__) to dense/stereo/patch-match.cfg; lock fusion.cfg; hash undistorted image/calibration outputs'},
        {'stage': 'patch_match_stereo', 'status': 'NOT_RUN',
         'argv_template': ['COLMAP_PINNED_BINARY', 'patch_match_stereo', '--workspace_path', case + '/dense',
                           '--workspace_format', 'COLMAP', '--PatchMatchStereo.geom_consistency', 'true',
                           '--PatchMatchStereo.gpu_index', '0', '--PatchMatchStereo.max_image_size', 'LOCKED_HALF_RESOLUTION_MAX_DIMENSION',
                           '--PatchMatchStereo.num_threads', '1', '--PatchMatchStereo.cache_size', '2',
                           '--PatchMatchStereo.depth_min', 'LOCKED_INPUT_ONLY_Z_MIN',
                           '--PatchMatchStereo.depth_max', 'LOCKED_INPUT_ONLY_Z_MAX'],
         'requires': 'Verified CUDA build, resource reservation, frozen input-only depth interval; no parameter tuning on confirmation'},
        {'stage': 'stereo_fusion', 'status': 'NOT_RUN',
         'argv_template': ['COLMAP_PINNED_BINARY', 'stereo_fusion', '--workspace_path', case + '/dense',
                           '--workspace_format', 'COLMAP', '--input_type', 'geometric',
                           '--StereoFusion.num_threads', '4', '--StereoFusion.cache_size', '2',
                           '--output_path', case + '/dense/fused.ply'],
         'requires': 'Complete geometric depth maps; independently fixed track/overlap policy; no silent min_num_pixels change'},
        {'stage': 'coordinate_and_evaluation_adapter', 'status': 'NOT_IMPLEMENTED', 'command': None,
         'requires': 'Apply frozen input-derived scale_mat exactly once, fixed ROI/support, raw output + no-hit/failure counts, seal geometry before GT access'},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-development-inputs', action='store_true', help='Only audit software/resources, without hashing old scene images')
    args = parser.parse_args()
    if socket.gethostname() != 'liekkas' or Path(__file__).resolve().parent != ROOT or Path.cwd().resolve() != ROOT:
        raise SystemExit('TARGET_MISMATCH: expected host liekkas and exact closeout workspace as cwd/script location')
    result = {'schema': 1, 'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'host': socket.gethostname(), 'workspace': str(ROOT), 'python': sys.executable,
              'platform': platform.platform(), 'baseline_result_status': 'NOT_RUN',
              'readiness_status': 'INCOMPLETE', 'software': local_software(), 'gpu': gpu(),
              'proposed_upstream_pin': PIN,
              'development_input_audit': [] if args.skip_development_inputs else development_inputs(),
              'planned_stages': stage_plan(),
              'comparison_scope': {'official_native_output': 'SYSTEM_REFERENCE: fixed calibration and same five photos, but no GeoSVR mesh prior in its depth estimator',
                                   'V28_inputs': 'Same five photographs/calibration plus existing GeoSVR geometry',
                                   'optional_fixed_point_lift': 'NOT_IMPLEMENTED; future custom adapter, not an official COLMAP algorithm'},
              'new_scene_evidence': 'NONE', 'gpu_computation_executed': False,
              'environment_modifications': False, 'external_downloads': False,
              'result_interpretation': 'Preflight and input-manifest checks only. NOT_RUN is no quality observation and is not a scientific failure.'}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
