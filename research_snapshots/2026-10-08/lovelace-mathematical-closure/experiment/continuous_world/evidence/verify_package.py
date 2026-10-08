"""Verify fresh extraction, bundled asset and independent checker from a new path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT.parent / 'Lovelace-mathematical-closure-stage2-20261008.zip'
EXTRACTED = PROJECT.parent / 'stage2-package-check-20261008'


def main():
    if EXTRACTED.resolve().parent != ARCHIVE.resolve().parent:
        raise ValueError('Extraction must stay directly within the named Lovelace directory')
    EXTRACTED.mkdir(exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as packed:
        for name in packed.namelist():
            if not (EXTRACTED / name).resolve().is_relative_to(EXTRACTED.resolve()):
                raise ValueError('Archive path escapes extraction root')
        packed.extractall(EXTRACTED)
    manifest = json.loads((EXTRACTED / 'STAGE2_PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    files_equal = all(hashlib.sha256((EXTRACTED / name).read_bytes()).hexdigest() == row['sha256']
                      for name, row in manifest['files'].items())
    code = ('from pathlib import Path; import json; '
            'from continuous_world.verify_continuous import verify_archive; '
            'r=verify_archive(Path("continuous_world/outputs/confirmation-v1").resolve()); '
            'print(json.dumps({"passed":r["passed"],"failures":r["failures"],"counts":r["counts"]})); '
            'raise SystemExit(0 if r["passed"] else 1)')
    run = subprocess.run([sys.executable, '-B', '-c', code], cwd=EXTRACTED,
                         text=True, encoding='utf-8', capture_output=True)
    report = {'schema': 1, 'passed': files_equal and run.returncode == 0,
              'extraction_root': str(EXTRACTED), 'manifest_files_match_after_fresh_extraction': files_equal,
              'independent_checker_from_extracted_tree_exit': run.returncode,
              'checker_result': json.loads(run.stdout) if run.returncode == 0 else run.stdout,
              'stderr': run.stderr, 'scope': 'fresh-path bundled source and geometry asset verification; no original GLB dependency'}
    receipt = ARCHIVE.with_suffix('.delivery-check.json')
    receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
