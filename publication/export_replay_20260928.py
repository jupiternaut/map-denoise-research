"""Export an immutable, bounded public subset of the seven 2026-09-26 labs.

Run on the original host with --dry-run to inspect the allowlist, or without it
to create new files. Existing files must match byte-for-byte and are never
overwritten. This copies evidence; it does not execute research code, load
pickled models, rewrite source paths, or reproduce the original experiments.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path('/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1')
SOURCE_ROOT = Path('/home/grf/Documents/Codex/2026-09-26')
RUN_ROOT = Path('/srv/slam-research/grf/map-denoise/runs')
SNAPSHOT = Path('research_snapshots/2026-09-26')
MANIFEST = Path('publication/REPLAY_20260928_MANIFEST.json')
ORIGIN = 'https://github.com/jupiternaut/map-denoise-research.git'
LABS = (
    ('relative-gain-lab-20260926T004220Z', None),
    ('reserved-evidence-lab-20260926T011717Z', None),
    ('boundary-cross-lab-20260926T014111Z', None),
    ('joint-revision-lab-20260926T080710Z', None),
    ('visibility-revision-lab-20260926T085529Z', 'visibility-revision-20260926T085529Z'),
    ('continuous-step-lab-20260926T100617Z', 'continuous-step-20260926T100617Z'),
    ('multisurface-field-lab-20260926T102842Z', 'multisurface-field-20260926T102842Z'),
)
ALLOW_SUFFIXES = {'.md', '.py', '.json', '.csv', '.txt', '.toml', '.png', '.svg', '.pdf'}
CACHE_PARTS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
               'plot-cache', 'mplconfig', '.cache', '.venv', 'venv', 'envs', 'node_modules'}
DENSE_SUFFIXES = {'.ply', '.pcd', '.npz', '.npy', '.e57', '.las', '.laz'}
RUN_SECTIONS = {'evaluation', 'evidence', 'inference', 'training', 'figures',
                'posthoc_field_capacity', 'smoke', 'observation_smoke'}
MODEL_PATHS = {'training/models.joblib', 'training/policies.joblib'} | {
    f'training/arms/{arm}/models.joblib'
    for arm in ('base_shallow', 'base_rich', 'geometry_shallow', 'geometry_rich',
                'interaction_shallow', 'interaction_rich')
}
LIMITS = {'file_bytes': 5_000_000, 'total_bytes': 80_000_000,
          'files': 1_600, 'json_files': 1_250}
CREDENTIAL_PATTERNS = {
    'private_key': rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----',
    'github_token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'credential_url': rb'https?://[^\s/@:]+:[^\s/@]+@',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'provider_api_key': rb'\b(?:sk-proj-|sk-ant-)[A-Za-z0-9_-]{20,}',
    'slack_token': rb'\bxox[baprs]-[A-Za-z0-9-]{20,}',
    'sensitive_literal_assignment': (
        rb'(?i)\b(?:password|passwd|api_key|access_token|client_secret)\b'
        rb'[\s\x22\x27]*[:=]\s*[\x22\x27][^\x22\x27\r\n]{8,}[\x22\x27]'
    ),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def verify_target():
    if socket.gethostname() != 'liekkas' or ROOT != EXPECTED_ROOT:
        raise RuntimeError('Target identity mismatch; original host and repository are required')
    if Path(git('rev-parse', '--show-toplevel')) != ROOT:
        raise RuntimeError('Unexpected Git repository root')
    if git('branch', '--show-current') != 'main':
        raise RuntimeError('Unexpected branch; expected main')
    if git('remote', 'get-url', '--all', 'origin').splitlines() != [ORIGIN]:
        raise RuntimeError('Unexpected origin fetch URL')
    if git('remote', 'get-url', '--push', '--all', 'origin').splitlines() != [ORIGIN]:
        raise RuntimeError('Unexpected origin push URL')


def path_bindings(path):
    """Read literal Path declarations without importing or executing the labs."""
    bindings = {}
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != 'Path' or not call.args:
            continue
        first = call.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if not Path(first.value).is_absolute():
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = first.value
    return bindings


def walk_files(root):
    """Inventory regular files and links, including hidden names, without following links."""
    for folder, dirs, filenames in os.walk(root, followlinks=False):
        for name in sorted(dirs):
            path = Path(folder) / name
            if path.is_symlink():
                yield path
        dirs[:] = sorted(name for name in dirs if not (Path(folder) / name).is_symlink())
        for name in sorted(filenames):
            yield Path(folder) / name


def exclusion_reason(path, rel, kind):
    if path.is_symlink():
        return 'symbolic_link_not_followed; link retained at source'
    if not stat.S_ISREG(path.lstat().st_mode):
        return 'non_regular_file_not_copied'
    if any(part in CACHE_PARTS for part in rel.parts):
        return 'cache_or_environment; regenerable, outside public research subset'
    if path.name.startswith('.env') or path.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}:
        return 'sensitive_filename; excluded from public subset'
    if path.suffix.lower() in DENSE_SUFFIXES:
        return 'dense_geometry_or_per_point_array; full replay requires original local data'
    if kind == 'run_evidence' and rel.parts[0] not in RUN_SECTIONS:
        return 'run_section_outside_explicit_evidence_allowlist'
    if path.suffix.lower() == '.joblib':
        return None if rel.as_posix() in MODEL_PATHS else 'model_not_explicitly_allowlisted'
    if path.suffix.lower() not in ALLOW_SUFFIXES and path.name not in {'LICENSE', 'COPYING', 'NOTICE'}:
        return 'extension_not_allowlisted; retained at source'
    if path.suffix.lower() in {'.png', '.svg', '.pdf'} and not any(
            part in {'figures', 'figures_capacity'} for part in rel.parts[:-1]):
        return 'image_outside_report_figure_directories; photos are not published'
    return None


def credential_findings(path):
    # Byte scanning does not execute serialized models. Only path/category are reported.
    content = path.read_bytes()
    return [{'source': str(path), 'category': category}
            for category, pattern in CREDENTIAL_PATTERNS.items() if re.search(pattern, content)]


def safe_target(rel):
    target = ROOT / rel
    if not target.is_relative_to(ROOT) or '..' in rel.parts:
        raise RuntimeError('Destination is outside the verified repository')
    for part in (target, *target.parents):
        if part == ROOT:
            break
        if part.is_symlink():
            raise RuntimeError(f'Destination contains a symbolic link: {rel}')
    return target


def build_manifest():
    rows, excluded, scopes, findings, lab_records = [], [], [], [], []
    for lab, run in LABS:
        source = SOURCE_ROOT / lab
        if not source.is_dir() or source.resolve() != source:
            raise RuntimeError(f'Unexpected source root: {source}')
        bindings = path_bindings(source / 'common.py')
        expected_out = str(RUN_ROOT / run) if run else None
        if bindings.get('OUT') != expected_out:
            raise RuntimeError(f'common.py OUT mapping differs: {source}')
        records = [(source, SNAPSHOT / lab, 'lab_workspace')]
        if run:
            run_source = RUN_ROOT / run
            if not run_source.is_dir() or run_source.resolve() != run_source:
                raise RuntimeError(f'Unexpected run root: {run_source}')
            records.append((run_source, SNAPSHOT / lab / 'run_evidence', 'run_evidence'))
        lab_records.append({
            'lab': lab, 'source': str(source), 'snapshot': str(SNAPSHOT / lab),
            'artifact_source': expected_out or str(source),
            'artifact_location_verified_from': str(source / 'common.py'),
            'common_py_sha256': sha(source / 'common.py'),
            'literal_absolute_path_bindings': bindings,
            'layout': 'external OUT copied under run_evidence' if run else 'outputs co-located with source',
        })
        for root, destination, kind in records:
            selected, omitted = [], []
            for path in sorted(walk_files(root)):
                rel = path.relative_to(root)
                size = path.lstat().st_size
                reason = exclusion_reason(path, rel, kind)
                base = {'source': str(path), 'bytes': size, 'lab': lab, 'source_kind': kind}
                if reason:
                    entry = {**base, 'reason': reason}
                    if path.is_symlink():
                        entry['link_target'] = os.readlink(path)
                    omitted.append(entry)
                    continue
                if size > LIMITS['file_bytes']:
                    raise RuntimeError(f'Allowlisted file exceeds byte limit; review required: {path}')
                findings.extend(credential_findings(path))
                rename = rel.name == 'AGENTS.md'
                target_rel = destination / (rel.with_name('SOURCE_AGENTS.md') if rename else rel)
                target = safe_target(target_rel)
                digest = sha(path)
                if target.exists() and (not target.is_file() or sha(target) != digest):
                    raise RuntimeError(f'Refusing overwrite of different existing file: {target_rel}')
                selected.append({**base, 'path': target_rel.as_posix(), 'sha256': digest,
                                 'transformation': 'rename_only_AGENTS_to_SOURCE_AGENTS' if rename else 'none'})
            rows.extend(selected)
            excluded.extend(omitted)
            scopes.append({
                'source': str(root), 'snapshot': destination.as_posix(), 'source_kind': kind,
                'inventoried_files': len(selected) + len(omitted),
                'included_files': len(selected), 'included_bytes': sum(r['bytes'] for r in selected),
                'excluded_files': len(omitted), 'excluded_bytes': sum(r['bytes'] for r in omitted),
                'included_source_inventory_sha256': canonical_sha(selected),
            })
    rows.sort(key=lambda item: item['path'])
    excluded.sort(key=lambda item: item['source'])
    if findings:
        print(json.dumps({'credential_review_required': findings}, ensure_ascii=False))
        raise RuntimeError('Credential-pattern findings require review before export')
    counts = Counter(Path(row['path']).suffix for row in rows)
    total_bytes = sum(row['bytes'] for row in rows)
    if len(rows) > LIMITS['files'] or counts['.json'] > LIMITS['json_files'] or total_bytes > LIMITS['total_bytes']:
        raise RuntimeError('Public subset exceeds explicit size/count limits; review scope before exporting')
    if len({row['path'] for row in rows}) != len(rows):
        raise RuntimeError('Duplicate snapshot destinations')
    script = Path(__file__).resolve()
    return {
        'schema_version': 1,
        'publication_date': '2026-09-28',
        'scope': 'Public source, frozen small models, protocols, audits, plots and tabular evidence for seven 2026-09-26 replay labs; not a full data backup or a portable experiment bundle',
        'research_status': 'Development scans 24/37; already exposed replay scans 55/65/69, not independent unseen confirmation. Diagnostic oracles remain labelled in historical reports.',
        'target': {'host': 'liekkas', 'repository': str(ROOT), 'origin': ORIGIN, 'branch': 'main'},
        'source_immutable': True,
        'source_verification': 'Every included source file is SHA-256 checked before and after copying; exported bytes must match. No research experiment was rerun by this exporter.',
        'historical_instructions': 'Every AGENTS.md is copied byte-for-byte as SOURCE_AGENTS.md; historical instructions are evidence, not active publication instructions.',
        'historical_seals': 'Original seals/manifests are unchanged and may refer to excluded dense data or absolute source paths. This manifest, not historical full-run seals, defines the public subset.',
        'portability': 'Original source paths are preserved. Loading archived models or rerunning research scripts requires the original dependencies and path adaptation; models were not deserialized during export.',
        'license_note': 'No new licence is granted by this export. Existing notices are preserved; third-party datasets and source photographs are excluded.',
        'selection': {
            'allowed_suffixes': sorted(ALLOW_SUFFIXES),
            'explicit_model_relative_paths': sorted(MODEL_PATHS),
            'external_run_sections': sorted(RUN_SECTIONS),
            'figure_directories': ['figures', 'figures_capacity'],
            'cache_environment_parts': sorted(CACHE_PARTS),
            'dense_suffixes_excluded': sorted(DENSE_SUFFIXES), 'limits': LIMITS,
            'all_exclusions_listed_individually': True,
        },
        'credential_scan': {'status': 'no_pattern_findings', 'scope': 'all included source-file bytes',
                            'categories': sorted(CREDENTIAL_PATTERNS), 'findings': [],
                            'limitation': 'Pattern scanning is not proof that no secret exists; binary models are scanned as bytes without loading.'},
        'external_dependencies': [
            {'source': '/srv/slam-research/grf/map-denoise/datasets',
             'status': 'not_in_this_snapshot', 'reason': 'DTU/reference geometry, camera calibration, meshes and source images needed for full replay; dataset redistribution excluded'},
            {'source': '/home/grf/Documents/Codex/2026-09-22/v28-surface-experts-20260922T125505Z',
             'status': 'prior_source_and_dense_artifacts_not_reexported', 'reason': 'Frozen A/B geometry, cached features, scene adapters and earlier constructors; consult prior repository snapshots and original data'},
            {'source': '/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z',
             'status': 'prior_source_and_dense_artifacts_not_reexported', 'reason': 'Frozen confirmation geometry/features and v28_closeout package; a prior public source subset exists under research_snapshots/2026-09-22'},
            {'source': '/srv/slam-research/grf/map-denoise/envs/open3d-019',
             'status': 'environment_not_copied', 'reason': 'Original Python/numerical environment; research imports include NumPy, SciPy, scikit-learn, Open3D, joblib, threadpoolctl and Matplotlib'},
            {'source': 'per-file paths in excluded', 'status': 'retained_at_original_sources',
             'reason': 'Per-point PLY/NPZ/NPY data are required for full inference/evaluation replay; cached plots/fonts are regenerable'},
        ],
        'labs': lab_records,
        'source_scopes': scopes,
        'summary': {'files': len(rows), 'bytes': total_bytes, 'json_files': counts['.json'],
                    'models': counts['.joblib'], 'excluded_files': len(excluded),
                    'excluded_bytes': sum(row['bytes'] for row in excluded),
                    'suffix_counts': dict(sorted(counts.items())),
                    'files_inventory_sha256': canonical_sha(rows)},
        'generated_artifacts': [{'path': script.relative_to(ROOT).as_posix(), 'bytes': script.stat().st_size,
                                 'sha256': sha(script), 'role': 'reproducible publication exporter, not historical research source'}],
        'manifest_self_reference': 'This manifest is excluded from its own file hashes. Publication prose and independent validation tools are separately reviewed Git changes.',
        'files': rows, 'excluded': excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='validate and show counts without writing')
    args = parser.parse_args()
    verify_target()
    manifest = build_manifest()
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    target_manifest = safe_target(MANIFEST)
    if target_manifest.exists() and target_manifest.read_bytes() != payload:
        raise RuntimeError('Existing manifest differs; refusing overwrite')
    if not args.dry_run:
        for row in manifest['files']:
            source, target = Path(row['source']), safe_target(Path(row['path']))
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise RuntimeError(f'Exported byte mismatch: {row["path"]}')
        for row in manifest['files']:
            source = Path(row['source'])
            if source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
                raise RuntimeError(f'Source changed during export: {row["source"]}')
        if not target_manifest.exists():
            with target_manifest.open('xb') as stream:
                stream.write(payload)
    print(json.dumps({'mode': 'dry-run' if args.dry_run else 'exported_and_verified',
                      **manifest['summary'], 'manifest': MANIFEST.as_posix(),
                      'manifest_bytes': len(payload)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
