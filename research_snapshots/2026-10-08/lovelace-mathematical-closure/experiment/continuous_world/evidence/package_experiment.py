"""Archive both experiment stages without altering historical sources/evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT.parent / 'Lovelace-mathematical-closure-stage2-20261008.zip'
MANIFEST = PROJECT / 'STAGE2_PACKAGE_MANIFEST.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if ARCHIVE.exists():
        raise FileExistsError('Preserve published package; use a fresh version name.')
    files = sorted(path for path in PROJECT.rglob('*') if path.is_file()
                   and '__pycache__' not in path.parts and path.suffix != '.pyc'
                   and path != MANIFEST)
    manifest = {'schema': 1, 'scope': 'complete local mathematical experiment tree, stages 1 and 2',
                'scientific_runtime': 'Python standard library; confirmed Python 3.14.0',
                'optional_presentation_rebuild': 'Matplotlib; see continuous_world/presentation/requirements.txt; offline HTML/SVG already supplied',
                'historical_manifest': 'PACKAGE_MANIFEST.json is the original stage-1 receipt; use this manifest for the current tree',
                'excluded': ['generated Python bytecode'],
                'files': {path.relative_to(PROJECT).as_posix():
                          {'bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())}
                          for path in files}}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files.append(MANIFEST)
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as packed:
        for path in sorted(files):
            packed.write(path, path.relative_to(PROJECT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as packed:
        bad = packed.testzip()
        all_equal = all(packed.read(path.relative_to(PROJECT).as_posix()) == path.read_bytes() for path in files)
        if bad is not None or not all_equal or len(packed.namelist()) != len(files):
            raise RuntimeError('Package bytes/CRC validation failed')
    receipt = {'schema': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'archive': ARCHIVE.name, 'bytes': ARCHIVE.stat().st_size,
               'sha256': digest(ARCHIVE.read_bytes()), 'file_count': len(files),
               'all_entries_match_workspace_bytes': all_equal, 'zip_crc_passed': bad is None,
               'manifest_sha256': digest(MANIFEST.read_bytes()),
               'scope': manifest['scope']}
    receipt_path = ARCHIVE.with_suffix('.receipt.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
