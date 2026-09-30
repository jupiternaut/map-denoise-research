"""Acquire preselected unseen DTU inputs only; never decode reference geometry.

Exact target: liekkas / colmap-transfer-20260930T180000Z. 118 and 122 are
historical reserves, selected before inspecting their photos or any scores.
All outputs are exclusive-create; the historical source archive is read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path('/srv/slam-research/grf/map-denoise/runs/colmap-transfer-20260930T180000Z')
DATA = Path('/srv/slam-research/grf/map-denoise/datasets')
ARCHIVE = DATA / 'closeout-confirmation-v1/downloads/dtu.tar.gz'
OLD_MANIFEST = DATA / 'closeout-confirmation-v1/INPUT_MANIFEST.json'
META = DATA / 'external-confirmation-20260928T213020Z/METADATA.json'
OLD_SELECTION = DATA / 'external-confirmation-20260928T213020Z/SCENES.json'
SCENES = (118, 122)
VIEWS = ('0007', '0016', '0017', '0022', '0035')
TAILS = tuple(f'images/{v}.png' for v in VIEWS) + (
    'cameras.npz', 'sparse/0/cameras.bin', 'sparse/0/images.bin')
TOTAL_LIMIT = 100_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def exposure_audit():
    roots = ['/home/grf/Documents/Codex', '/srv/slam-research/grf/map-denoise/runs']
    globs = ['*.md', '*MANIFEST*.json', '*LOCK*.json', '*SCENES*.json',
             '*RESULTS*.csv', '*METRICS*.csv']
    command = ['rg', '--files', *roots]
    for glob in globs:
        command += ['-g', glob]
    command += ['-g', '!**/.git/**', '-g', '!**/' + ROOT.name + '/**']
    files = sorted(subprocess.check_output(command, text=True).splitlines())
    pattern = re.compile(r'(?i)(?:scan[_ -]?(118|122)(?!\d)|stl(118|122)(?!\d)'
                         r'|"(?:scene|scene_id|sid)"\s*:\s*(118|122)(?!\d))')
    hits, skipped = [], []
    for filename in files:
        path = Path(filename)
        if path.stat().st_size > 16 * 1024**2:
            skipped.append({'path': filename, 'reason': 'over 16 MiB text limit'})
            continue
        body = path.read_text(errors='replace')
        matches = []
        for number, line in enumerate(body.splitlines(), 1):
            if pattern.search(line):
                matches.append({'line': number, 'text': line[:500]})
        for match in re.finditer(r'"scenes"\s*:\s*\[([^]]+)\]', body):
            if set(map(int, re.findall(r'\b\d+\b', match.group(1)))) & set(SCENES):
                matches.append({'line': body.count('\n', 0, match.start()) + 1,
                                'text': match.group(0)[:500]})
        if matches:
            hits.append({'path': filename, 'sha256': sha(path), 'matches': matches})
    return {'roots': roots, 'filename_globs': globs, 'files_considered': len(files),
            'file_list_sha256': hashlib.sha256('\n'.join(files).encode()).hexdigest(),
            'matching_files': hits, 'skipped': skipped,
            'limitation': 'No recorded experimental use in bounded local text search; '
                          'not proof about unrecorded activity or other machines.'}


def main():
    if socket.gethostname() != 'liekkas' or Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Wrong host or absolute run identity')
    for name in ('SCENE_SELECTION.json', 'SCENE_SELECTION.md', 'INPUT_MANIFEST.json'):
        if (ROOT / name).exists():
            raise FileExistsError('Will not overwrite existing acquisition: ' + name)
    metadata = json.loads(META.read_text())
    selection = json.loads(OLD_SELECTION.read_text())
    manifest = json.loads(OLD_MANIFEST.read_text())
    expected = [r for r in manifest['files'] if r['path'] == str(ARCHIVE)]
    if len(expected) != 1:
        raise RuntimeError('Archive source identity not unique')
    expected = expected[0]
    if (expected['bytes'], expected['sha256']) != (metadata['archive']['bytes'], metadata['archive']['sha256']):
        raise RuntimeError('Historical manifests disagree')
    if sha(OLD_MANIFEST) != metadata['archive']['historical_manifest_sha256']:
        raise RuntimeError('Historical manifest changed')
    if sha(META) != selection['metadata_sha256']:
        raise RuntimeError('Historical metadata changed')
    if ARCHIVE.stat().st_size != expected['bytes'] or sha(ARCHIVE) != expected['sha256']:
        raise RuntimeError('Source archive bytes changed')
    decisions = {r['scene']: r['status'] for r in selection['decisions']}
    if any(decisions[s] != 'RESERVE_PRIORITY_NOT_REACHED' for s in SCENES):
        raise RuntimeError('Chosen scenes were not untouched historical reserves')
    member_expected = {}
    for scene in SCENES:
        record = metadata['archive']['scenes'][str(scene)]
        if not record['ready']:
            raise RuntimeError('Input metadata incomplete')
        members = {v['relative']: v for v in record['members']}
        for tail in TAILS:
            entry = members[tail]
            member_expected[entry['member']] = {'scene': scene, 'relative': tail,
                                                 'bytes': entry['bytes']}
    if sum(r['bytes'] for r in member_expected.values()) >= TOTAL_LIMIT:
        raise RuntimeError('Input extraction cap exceeded')
    audit = exposure_audit()
    # Historical reserve lists are expected. Any actual new scene-specific result
    # detected here is a stop condition for the parent to inspect, not auto-replaced.
    if audit['matching_files']:
        raise RuntimeError('Scene mention found; inspect before acquisition: ' +
                           json.dumps(audit['matching_files'], ensure_ascii=False))
    binding = {str(p): sha(p) for p in (Path(__file__), OLD_MANIFEST, META, OLD_SELECTION)}
    chosen = {'created_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
              'root': str(ROOT), 'scenes': list(SCENES), 'reference_view': '0022',
              'source_views': ['0016', '0035', '0017', '0007'],
              'selection_rule': 'Use the two previously reserved, metadata-complete scenes '
                                '118 and 122, in their historical priority order; no score selection.',
              'historically_used_dtu': [24, 37, 40, 55, 65, 69, 83, 97, 105, 106, 110, 114],
              'extra_conservative_exclusions': [1, 2, 63],
              'prior_selection_status': {str(s): decisions[s] for s in SCENES},
              'audit': audit, 'source_bindings': binding,
              'photos_seen_at_selection': False, 'reference_content_accessed': False,
              'confirmation_scope': 'new scenes within the same DTU acquisition family; '
                                    'not cross-sensor or cross-acquisition generalization'}
    save(ROOT / 'SCENE_SELECTION.json', chosen)
    with (ROOT / 'SCENE_SELECTION.md').open('x', encoding='utf-8') as f:
        f.write('# 冻结迁移场景：DTU 118 / 122\n\n'
                '选择依据：两者是上一轮六场景确认实验的预先登记候补，状态均为 '
                '`RESERVE_PRIORITY_NOT_REACHED`。本次按原候补顺序使用，不查看新照片、'
                '参照或方法分数再挑选，不根据结果换场景。\n\n'
                '已使用场景至少包括 24、37、40、55、65、69、83、97、105、106、110、114；'
                '文本中另出现 1、2、63，保守排除。限定历史全文检索未发现 118/122 '
                '既往实验记录；检索范围、文件列表哈希及限制见 JSON。这不是对其他机器'
                '或未记录活动的保证。\n\n'
                '照片和相机只从已封存的 GeoSVR DTU 输入归档提取，每场景固定 '
                '0022 参考视图及 0016/0035/0017/0007 源视图。保留 supplied world_mat '
                '物理单位相机，后续需单独核对像素约定；不提取 points3D、不载入参照。\n\n'
                '这属于同 DTU 采集族中的新场景迁移，不是跨传感器确认。\n')
    extracted, seen = [], set()
    with tarfile.open(ARCHIVE, 'r|gz') as archive:
        for member in archive:
            if member.name not in member_expected:
                continue
            info = member_expected[member.name]
            pp = PurePosixPath(member.name)
            if pp.is_absolute() or '..' in pp.parts or not member.isfile():
                raise RuntimeError('Unsafe selected archive member')
            if member.name in seen or member.size != info['bytes']:
                raise RuntimeError('Duplicate or changed member')
            seen.add(member.name)
            dest = ROOT / 'inputs' / f"scan{info['scene']}" / info['relative']
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, dest.open('xb') as out:
                copied = 0
                while chunk := src.read(1 << 20):
                    copied += len(chunk)
                    if copied > info['bytes']:
                        raise RuntimeError('Archive member exceeded declared size')
                    out.write(chunk)
            if copied != info['bytes']:
                raise RuntimeError('Short archive member')
            extracted.append({'member': member.name, 'path': str(dest),
                              'bytes': copied, 'sha256': sha(dest)})
    if seen != set(member_expected):
        raise RuntimeError('Missing locked inputs')
    save(ROOT / 'INPUT_MANIFEST.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
         'host': socket.gethostname(), 'scenes': list(SCENES), 'files': extracted,
         'total_bytes': sum(r['bytes'] for r in extracted), 'archive': expected,
         'source_bindings': binding, 'selection_sha256': sha(ROOT / 'SCENE_SELECTION.json'),
         'points3D_extracted': False, 'reference_content_accessed': False})
    print('INPUTS_READY', len(extracted), sum(r['bytes'] for r in extracted), flush=True)


if __name__ == '__main__':
    main()
