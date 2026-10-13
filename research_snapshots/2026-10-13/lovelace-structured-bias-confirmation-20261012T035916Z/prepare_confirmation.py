"""Separate one-shot generation, never imported by the fitting process."""
import sys
sys.dont_write_bytecode = True
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-structured-bias-confirmation-20261012T035916Z')
FROZEN = ROOT.parent / 'lovelace-structured-view-bias-20261012T022500Z'
SHARED = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z'
SEED, PREFIX, BETA = 61004, 'shared-appearance-v1', Q(1, 100)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def transform(image, decode, encode):
    """Unclipped base + bias, then exactly one clamp; no pre-clamp."""
    return [encode(min(Q(1), max(Q(0), decode(y) + (-BETA, Q(0), BETA)[i // 144])))
            for i, y in enumerate(image)]


def prepare():
    if socket.gethostname() != 'liekkas' or ROOT.resolve() != EXPECTED:
        raise RuntimeError('requires exact registered host/root')
    review = json.loads((ROOT / '.aris/REVIEW_GATE.json').read_text())
    if review.get('status') != 'PASS':
        raise RuntimeError('fresh source review must pass before unseen generation')
    for path, digest in review['files'].items():
        if sha(path) != digest:
            raise RuntimeError('reviewed source changed: ' + path)
    frozen_lock = json.loads((FROZEN / 'outputs/structured-v1/SOURCE_LOCK.json').read_text())
    for path, digest in frozen_lock['files'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: ' + path)
    data = ROOT / 'data'
    if data.exists():
        raise FileExistsError('one-shot preparation will not overwrite data')
    shared = module('confirmation_frozen_generator', SHARED / 'run_shared.py')
    observations, tasks, truths = [], [], []
    started = time.perf_counter()
    for case, branch, colors in shared.CASES:
        for truth in shared.TRUTHS:
            public, private = shared.generate_observation(case, branch, colors, truth, SEED,
                radius=Q(1, 5), prefix=PREFIX)
            oid = f'c{len(observations) + 1:03d}'
            observations.append({'observation_id': oid, 'unit_id': 'clip', 'clipped': True,
                'radius': public['radius'], 'observation': transform(public['observation'], shared.decode, shared.encode)})
            private = {k: v for k, v in private.items() if k not in ('calibration_noise', 'calibration_noise_linf')}
            truths.append({**private, 'observation_id': oid, 'unit_id': 'clip', 'clipped': True,
                'image_kind': 'biased_clip', 'actual_view_bias': list(map(shared.encode, (-BETA, Q(0), BETA)))})
            for state, offset in enumerate(shared.OFFSETS):
                tid = f'{oid}s{state}'
                tasks.append({'task_id': tid, 'base_task_id': tid, 'observation_id': oid,
                    'incumbent': shared.encode(truth + offset)})
    if (len(observations), len(tasks), len(truths)) != (18, 162, 18):
        raise RuntimeError('wrong registered generation population')
    dump(data / 'observations.json', observations)
    dump(data / 'tasks.json', tasks)
    dump(data / 'private/truth.json', truths)
    dump(data / 'GENERATION.json', {'seed': SEED, 'prefix': PREFIX, 'observations': 18, 'tasks': 162,
        'truths': 18, 'transform': 'add fixed view bias to unclipped base then clamp once',
        'auxiliary_samples_used': 0, 'seconds': time.perf_counter() - started})
    print(json.dumps({'prepared': str(data), 'observations': 18, 'tasks': 162}), flush=True)


if __name__ == '__main__':
    prepare()
