"""Append a compact V19–V22 publication snapshot, never overwrite run evidence."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('/srv/slam-research/grf/map-denoise/runs')
SOURCES = (
    RUNS / 'multiscan-pilot-v1/parallel-v19-9ob96cyr',
    RUNS / 'multiscan-pilot-v1/candidate-repair-v20-qndibmbc',
    RUNS / 'real-closure-v21-dhb1ebbx',
    RUNS / 'reconstruction-v22-qayc8gft',
)


def main():
    dest = ROOT / 'evidence/progress_v22'
    dest.mkdir(exist_ok=False)
    records = []

    def copy(source, target):
        assert source.is_file() and not source.is_symlink(), source
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), target
        shutil.copyfile(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        records.append(dict(source=str(source), path=str(target.relative_to(ROOT)),
                            bytes=target.stat().st_size, sha256=digest))

    # Compact tables and provenance, excluding per-point parameters and predictions.
    keywords = ('RESULTS', 'SUMMARY', 'SLICES', 'AUDIT', 'REVIEW', 'VERIFICATION',
                'LOCK', 'SEALED', 'TIMING', 'ORACLE_DIAGNOSTIC', 'SELECTION')
    for folder in SOURCES:
        for p in sorted(folder.iterdir()):
            if (p.is_file() and p.suffix in ('.json', '.csv')
                    and any(word in p.name.upper() for word in keywords)
                    and p.stat().st_size < 2 * 1024**2):
                copy(p, dest / 'runs' / folder.name / p.name)
    oracle = SOURCES[1] / 'posthoc_raw_R/COMPLETE_ARCHIVED_POOL_ORACLE.json'
    copy(oracle, dest / 'runs' / SOURCES[1].name / 'posthoc_raw_R' / oracle.name)
    data = RUNS.parent / 'datasets/reconstruction-v22-scan37'
    for p in sorted(data.glob('DOWNLOAD*.json')):
        copy(p, dest / 'data_provenance/scan37' / p.name)
    # Run locks include top-level documentation. Preserve its pre-publication bytes
    # before updating the GitHub landing pages; scientific source stays untouched.
    for name in ('README.md', 'REPOSITORY_SNAPSHOT.md'):
        copy(ROOT / name, dest / 'pre_publication_docs' / name)
    manifest = dict(scope='V19–V22 compact evidence; not full raw-output backup',
                    files=records, total_bytes=sum(r['bytes'] for r in records))
    with (dest / 'EXPORT_MANIFEST.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(dict(files=len(records), bytes=manifest['total_bytes'])))


if __name__ == '__main__':
    main()
