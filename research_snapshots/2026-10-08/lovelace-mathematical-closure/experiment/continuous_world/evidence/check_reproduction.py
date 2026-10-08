"""Compare frozen-source stage-2 rerun; no experiment implementation imports."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL = PROJECT / 'continuous_world/outputs/confirmation-v1'
REPLAY = PROJECT / 'continuous_world/outputs/replay-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = {}
    for name in ('BASELINE_FORECASTS.json', 'FORECAST_LOCK.json', 'scenes.jsonl', 'controls.json'):
        first, second = ORIGINAL / name, REPLAY / name
        files[name] = {'byte_equal': first.read_bytes() == second.read_bytes(),
                       'sha256': digest(first), 'replay_sha256': digest(second)}
    first = json.loads((ORIGINAL / 'summary.json').read_text(encoding='utf-8'))
    second = json.loads((REPLAY / 'summary.json').read_text(encoding='utf-8'))
    times = {'original_seconds': first.pop('elapsed_seconds'),
             'replay_seconds': second.pop('elapsed_seconds')}
    summary_equal = first == second
    first_protocol = json.loads((ORIGINAL / 'PROTOCOL.json').read_text(encoding='utf-8'))
    second_protocol = json.loads((REPLAY / 'PROTOCOL.json').read_text(encoding='utf-8'))
    for protocol in (first_protocol, second_protocol):
        for model in protocol['models']:
            model.pop('source_asset', None)
    protocol_equal = first_protocol == second_protocol
    source_first = json.loads((ORIGINAL / 'SOURCE_LOCK.json').read_text(encoding='utf-8'))
    source_second = json.loads((REPLAY / 'SOURCE_LOCK.json').read_text(encoding='utf-8'))
    source_equal = source_first['hashes'] == source_second['hashes']
    passed = all(row['byte_equal'] for row in files.values()) and summary_equal and protocol_equal and source_equal
    report = {'schema': 1, 'passed': passed,
              'scope': 'same frozen algorithm and same archived confirmation conditions; not independent new scenes',
              'mathematical_data': files, 'summary_equal_excluding_elapsed_seconds': summary_equal,
              'protocol_equal_excluding_absolute_asset_path': protocol_equal,
              'algorithm_source_hash_maps_equal': source_equal,
              'metadata_expected_to_differ': ['UTC source receipt', 'elapsed wall time', 'absolute asset path'],
              'timing': times}
    (ORIGINAL / 'REPRODUCIBILITY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
