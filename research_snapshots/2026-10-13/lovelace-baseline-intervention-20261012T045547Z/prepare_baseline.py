"""One-shot paired image generation in a process separate from inference."""
import sys
sys.dont_write_bytecode = True
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import socket
import time
import baseline_geometry as geometry

ROOT = Path(__file__).resolve().parent
EXPECTED = Path('/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z')
SHARED = ROOT.parent / 'lovelace-shared-appearance-20261011T052300Z'
FROZEN = ROOT.parent / 'lovelace-structured-view-bias-20261012T022500Z'
SEED, PREFIX, BETA = 61005, 'shared-appearance-v1', Q(1, 100)
BASELINES = (Q(70), Q(80))

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

def render_pair(shared, branch, truth, colors, private, baseline):
    offsets = tuple(map(shared.decode, private['actual_offsets']))
    noise = tuple(map(shared.decode, private['image_noise']))
    image = geometry.point_rgb(branch, truth, offsets, colors, baseline)
    return [shared.encode(min(Q(1), max(Q(0), value + error + (-BETA, Q(0), BETA)[i // 144])))
            for i, (value, error) in enumerate(zip(image, noise))]

def sample_tape(shared, case, branch, colors, truth, seed):
    """Same first434 historical draws, without unused material samples."""
    tape = f'{PREFIX}:{case}:{Q(1,5)}:{truth}:{seed}'
    rng = random.Random(tape)
    offsets = (Q(1,5)*rng.choice(shared.WEIGHTS), Q(0), Q(1,5)*rng.choice(shared.WEIGHTS))
    noise = tuple(Q(1,100)*rng.choice(shared.WEIGHTS) for _ in range(432))
    return {'case':case,'rear_branch':branch,'truth':shared.encode(truth),'colors':list(map(shared.encode,colors)),
            'seed':seed,'random_tape':tape,'actual_offsets':list(map(shared.encode,offsets)),
            'image_noise':list(map(shared.encode,noise)),'noise_linf':shared.encode(max(map(abs,noise)))}

def build_population(shared, seed):
    """Fixture-testable; formal seed is invoked only by prepare after review."""
    observations, tasks, truths = [], [], []
    for case, branch, colors in shared.CASES:
        for truth in shared.TRUTHS:
            private = sample_tape(shared,case,branch,colors,truth,seed)
            pair_id = f'p{len(observations)//2+1:03d}'
            for baseline in BASELINES:
                oid, unit = pair_id + 'b' + str(baseline), 'baseline_' + str(baseline)
                observations.append({'observation_id': oid, 'pair_id': pair_id, 'unit_id': unit,
                    'baseline': str(baseline), 'clipped': True, 'radius': '1/5',
                    'observation': render_pair(shared, branch, truth, colors, private, baseline)})
                truths.append({**private, 'observation_id': oid, 'pair_id': pair_id, 'unit_id': unit,
                    'baseline': str(baseline), 'clipped': True, 'image_kind': 'biased_clip',
                    'actual_view_bias': list(map(shared.encode, (-BETA, Q(0), BETA)))})
                for state, offset in enumerate(shared.OFFSETS):
                    tasks.append({'task_id': f'{oid}s{state}', 'base_task_id': f'{pair_id}s{state}',
                        'observation_id': oid, 'incumbent': shared.encode(truth+offset)})
    return observations, tasks, truths

def prepare():
    if socket.gethostname() != 'liekkas' or ROOT.resolve() != EXPECTED:
        raise RuntimeError('requires exact registered host/root')
    review = json.loads((ROOT / '.aris/REVIEW_GATE.json').read_text())
    if review.get('status') != 'PASS':
        raise RuntimeError('fresh source review must pass before unseen generation')
    for path, digest in review['files'].items():
        if sha(path) != digest:
            raise RuntimeError('reviewed source changed: '+path)
    lock = json.loads((FROZEN / 'outputs/structured-v1/SOURCE_LOCK.json').read_text())
    for path, digest in lock['files'].items():
        if sha(path) != digest:
            raise RuntimeError('frozen source changed: '+path)
    data = ROOT / 'data'
    if data.exists():
        raise FileExistsError('one-shot preparation will not overwrite data')
    shared = module('baseline_frozen_generator', SHARED / 'run_shared.py')
    started = time.perf_counter()
    observations, tasks, truths = build_population(shared, SEED)
    if (len(observations), len(tasks), len(truths)) != (36, 324, 36):
        raise RuntimeError('wrong registered generation population')
    dump(data / 'observations.json', observations)
    dump(data / 'tasks.json', tasks)
    dump(data / 'private/truth.json', truths)
    dump(data / 'GENERATION.json', {'seed': SEED, 'prefix': PREFIX, 'observations': 36, 'tasks': 324,
        'truths': 36, 'transform': 'same pose/noise tape; explicit baseline point image + noise + fixed view bias; clamp once',
        'auxiliary_samples_used': 0, 'paired_noise': True, 'baselines_mm': ['70','80'],
        'seconds': time.perf_counter()-started})
    print(json.dumps({'prepared': str(data), 'observations': 36, 'tasks': 324}), flush=True)

if __name__ == '__main__':
    prepare()
