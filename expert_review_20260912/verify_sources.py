"""Read existing results for the expert dossier; does not run/change algorithms."""
import csv,json,hashlib,statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RUNS=Path('/srv/slam-research/grf/map-denoise/runs')
SOURCES={
 'E01':Path('/home/grf/Documents/Codex/2026-09-04/fable-5-1-qce-0-home/outputs/最小数学系统与逐级消融.md'),
 'E02':Path('/home/grf/Documents/Codex/2026-09-10/map-denoise-v0/local_operator_v1/README.md'),
 'E03':Path('/home/grf/Documents/Codex/2026-09-10/map-denoise-v0/parallel_geometry_v5/REPORT.md'),
 'E04':Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/REPORT.md'),
 'E05':PROJECT/'exploration_v4/REPORT.md',
 'E06':PROJECT/'exploration_v8/REPORT.md',
 'E07':PROJECT/'exploration_v10/REPORT.md',
 'E08':PROJECT/'exploration_v11/REPORT.md',
 'E09':PROJECT/'exploration_v13/REPORT.md',
 'E10':PROJECT/'exploration_v14/REPORT.md',
 'E11':PROJECT/'published_outputs_v2/REPORT.md',
 'E12':PROJECT/'published_outputs_v3/REPORT.md',
 'E13':PROJECT/'observation_model_v1/REPORT.md',
 'D10':RUNS/'multiscan-pilot-v1/effect-v14-taip2cld/confirmation_RESULTS.csv',
 'D11':RUNS/'published-outputs-v2/scan24/GEOMETRY_RESULTS.json',
 'D12':RUNS/'published-outputs-v3/RESULTS.json',
 'D13':RUNS/'observation-plane-v1-izu7fff7/RESULTS.json',
 'D14':RUNS/'observation-model-v1-7riqb9c4/RESULTS.json',
 'D15':RUNS/'observation-plane-v1-izu7fff7/AUDIT.json',
 'D16':RUNS/'published-outputs-v2/room/RENDER_RESULTS.json',
}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    index={key:dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for key,p in SOURCES.items()}
    with SOURCES['D10'].open() as f:rows=list(csv.DictReader(f))
    aggregates={}
    for method in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==method and r['status'] in ('OK','APPLY')]
        assert len(rr)==144,(method,len(rr))
        aggregates[method]=dict(n=len(rr),mae_mm=statistics.mean(float(r['surface_accuracy_mean_mm']) for r in rr),
            rms_mm=statistics.mean(float(r['matched_point_rms_mm']) for r in rr))
    scan24=json.loads(SOURCES['D12'].read_text())['summary']
    current=json.loads(SOURCES['D13'].read_text())
    candidate=[r for r in current['summary'] if r['method']=='valid1_local1_normal_all']
    baseline=[r for r in json.loads(SOURCES['D14'].read_text())['summary'] if r['method']=='identity']
    for r in candidate:
        if r['kind']=='normal_shift':assert abs(r['mean_mm']-2.0365838181135234)<1e-10
    assert abs(next(r['mean_mm'] for r in scan24 if r['method']=='identity')-.4980698627003903)<1e-10
    assert len(rows)==1152
    checks=dict(scope='Read-only arithmetic/report-source check, not independent algorithm rerun',
        v14_aggregates=aggregates,scan24_aggregates=scan24,observation_identity=baseline,
        observation_candidate=candidate,ambiguity=current['ambiguity'],latest_audit=json.loads(SOURCES['D15'].read_text()))
    for key,p in SOURCES.items():assert digest(p)==index[key]['sha256']
    (HERE/'EVIDENCE_INDEX.json').write_text(json.dumps(index,indent=2,ensure_ascii=False))
    (HERE/'CLAIM_CHECKS.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False))
    print(json.dumps(dict(verified_source_files=len(index),v14=aggregates),ensure_ascii=False))

if __name__=='__main__':main()
