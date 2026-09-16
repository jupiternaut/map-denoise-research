"""Reproducible six-row, per-patch aggregated data for the native chart preview."""
import json
from pathlib import Path
import sqlite3
import sys

SQL = """
SELECT a.scene, a.field, COUNT(*) AS n_patches,
       AVG(a.accuracy_mm) AS accuracy_mm,
       AVG(i.accuracy_mm - a.accuracy_mm) AS mae_gain_mm,
       AVG(a.recall) * 100.0 AS recall_percent,
       AVG(a.recall - i.recall) * 100.0 AS recall_change_pp,
       AVG(a.actual_budget_mm) AS actual_budget_mm
FROM v24_results AS a
JOIN v24_results AS i ON a.case_id=i.case_id AND i.method='native__identity'
WHERE a.kind='budget' AND a.allocation='uniform'
  AND a.requested_budget_mm=0.025
GROUP BY a.scene,a.field
ORDER BY a.scene,a.field
"""


def main():
    dest = Path(sys.argv[1]); data_path = dest/'RESULTS.json'
    rows = json.loads(data_path.read_text())
    conn = sqlite3.connect(':memory:'); conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE v24_results (scene TEXT, case_id TEXT, method TEXT, '
                 'field TEXT, kind TEXT, allocation TEXT, requested_budget_mm REAL, '
                 'actual_budget_mm REAL, accuracy_mm REAL, recall REAL)')
    conn.executemany('INSERT INTO v24_results VALUES (?,?,?,?,?,?,?,?,?,?)',
                     [(r['scene'],r['case'],r['method'],r['field'],r['kind'],r['allocation'],
                       r.get('requested_budget_mm'),r.get('actual_budget_mm'),r['accuracy_mm'],r['recall'])
                      for r in rows if r['status']=='OK'])
    result = [dict(r) for r in conn.execute(SQL)]
    assert len(result)==6 and all(abs(r['actual_budget_mm']-.025)<1e-12 for r in result)
    payload = dict(title='V24：等位移预算下的表面距离变化',
                   subtitle='三种修正场的排序依场景而变；距离改善需同时查看覆盖代价。',
                   source=dict(label='V24 / 42 exposed patches / frozen output evaluation',path=str(data_path),
                               query=dict(engine='SQLite (in memory)',language='sql',sql=SQL,
                                          description='同片输入与输出距离差，按场景及修正场等权平均；0.025 mm预算未触发上限。',
                                          tables_used=['v24_results'],
                                          metric_definitions={'mae_gain_mm':'mean_patch(identity MAE - candidate MAE), positive=lower error; mm',
                                                              'recall_change_pp':'mean_patch(candidate recall - identity recall)*100; percentage points'})),
                   table=dict(rows=result,row_count=6,truncated=False),
                   chart=dict(type='bar',fields=dict(x={'field':'scene'},y={'field':'mae_gain_mm'},color={'field':'field'}),
                              options=dict(grouping='grouped',orientation='vertical')),
                   display=dict(baseline=0,controls=True,unit='mm',x_axis_title='场景（已暴露诊断）',y_axis_title='MAE 降低量（mm，越高越好）'))
    with (dest/'chart_payload.json').open('x') as f: json.dump(payload,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(payload,ensure_ascii=False))


if __name__=='__main__': main()
