"""Compare historical stage-1 package to existing first-stage artifacts."""
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT.parent / 'Lovelace-mathematical-closure-stage1-20261008.zip'


def main():
    results = {}
    with zipfile.ZipFile(ARCHIVE) as packed:
        for name in packed.namelist():
            if name.endswith('/'):
                continue
            path = PROJECT / name
            results[name] = path.exists() and packed.read(name) == path.read_bytes()
    historical_manifest_names = [name for name in results if name != 'README.md']
    passed = all(results[name] for name in historical_manifest_names)
    report = {'schema': 1, 'passed': passed,
              'archive': ARCHIVE.name, 'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              'stage1_files_byte_equal': {name: results[name] for name in historical_manifest_names},
              'allowed_metadata_change': {'README.md': 'added stage-2 navigation; original remains in stage-1 ZIP'},
              'scope': 'all historical stage-1 source, evidence, report, review and package manifest remain byte-identical'}
    (PROJECT / 'continuous_world/outputs/confirmation-v1/STAGE1_PRESERVATION.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': passed, 'unchanged_historical_files': len(historical_manifest_names),
                      'unexpected_changes': [name for name in historical_manifest_names if not results[name]]}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
