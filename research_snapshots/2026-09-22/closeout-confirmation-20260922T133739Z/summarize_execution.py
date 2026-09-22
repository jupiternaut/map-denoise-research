"""Render frozen-scene evaluation without changing models or selecting a winner."""
import csv,json
from pathlib import Path
from scene_adapter import save,sha
ROOT=Path(__file__).resolve().parent
def main():
    d=json.loads((ROOT/'evaluation/SUMMARY.json').read_text());p=d['confirmation'];s=d['per_scene']
    rows=list(csv.DictReader((ROOT/'evaluation/METRICS.csv').open()))
    assert len(rows)==560 and len([r for r in rows if r['role']=='confirmation'])==420
    conditions=['native','minus1','plus1','minus3','plus3'];names={'native':'原始','minus1':'−1 mm','plus1':'+1 mm','minus3':'−3 mm','plus3':'+3 mm'}
    def improvement(c,sid=None):
        x=p[c] if sid is None else s[str(sid)][c]
        return 1-x['post_A_keep']['source_MSE_mm2']/x['identity']['source_MSE_mm2']
    native=dict(scene_improvements={str(i):improvement('native',i) for i in (55,65,69)},
                pooled_improvement=improvement('native'),
                recall_deltas={str(i):s[str(i)]['native']['post_A_keep']['recall_1mm']-s[str(i)]['native']['identity']['recall_1mm'] for i in (55,65,69)},
                p95_ratios={str(i):s[str(i)]['native']['post_A_keep']['source_p95_mm']/s[str(i)]['native']['identity']['source_p95_mm'] for i in (55,65,69)})
    native['pass']=(all(x>0 for x in native['scene_improvements'].values()) and native['pooled_improvement']>=.05
                    and all(x>=-.01 for x in native['recall_deltas'].values()) and all(x<=1.05 for x in native['p95_ratios'].values()))
    recovery={c:dict(pooled_improvement=improvement(c),scene_improvements={str(i):improvement(c,i) for i in (55,65,69)}) for c in ('minus3','plus3')}
    for c,r in recovery.items():r['pass']=r['pooled_improvement']>=.1 and all(x>0 for x in r['scene_improvements'].values())
    diagnostic={}
    for c in conditions:
        diagnostic[c]={}
        for arm in ('A_all','post_A_keep','random_A_keep'):
            rr=[r for r in rows if r['role']=='confirmation' and r['condition']==c and r['arm']==arm]
            n=sum(int(r['n_source']) for r in rr)
            diagnostic[c][arm]=dict(n=n,improved_fraction=sum(float(r['improved_fraction'])*int(r['n_source']) for r in rr)/n,
                harmed_fraction=sum(float(r['harmed_fraction'])*int(r['n_source']) for r in rr)/n,
                benefit_mm2=sum(float(r['benefit_sum_mm2']) for r in rr),harm_mm2=sum(float(r['harm_sum_mm2']) for r in rr),
                accepted_fraction=sum(float(r['accepted_fraction'])*int(r['n_rows']) for r in rr)/sum(int(r['n_rows']) for r in rr))
    save(ROOT/'EXPERIMENT_DECISION.json',dict(native=native,recovery=recovery,diagnostics=diagnostic,
        official_baseline='NOT_RUN',metrology_uncertainty_quantified=False,default='identity' if not native['pass'] else 'not_promoted_pending_external_baseline_and_metrology'))
    text=['# 收尾实验首轮执行报告','',
      '2026-09-22，主机 liekkas。固定主方法 post_A_keep，固定训练模型和协议；没有按结果替换主方法。',
      'scan40 为工程适配；scan55/65/69 为三个未参与本轮开发的同来源确认场景。',
      '80 个输入案例 / 560 行结果已完成；独立确认60案例/420行。所有构造输出封存后才获取并打开参考。','',
      '## 1. 主结果：固定源点 MSE（mm²）','',
      '下表先按 ROI 等权再按场景等权，改善率是总体均值之比；正值为改善。',
      '| 输入 | identity | 主方法 A/KEEP | 相对改善 |','|---|---:|---:|---:|']
    for c in conditions:text.append(f"| {names[c]} | {p[c]['identity']['source_MSE_mm2']:.6f} | {p[c]['post_A_keep']['source_MSE_mm2']:.6f} | {100*improvement(c):+.2f}% |")
    text+=['','## 2. 逐场景：不能用均值隐藏失败','',
      '| 场景 | 原始改善 | −1 mm | +1 mm | −3 mm | +3 mm |',
      '|---|---:|---:|---:|---:|---:|']
    for sid in (55,65,69):text.append('| '+str(sid)+' | '+' | '.join(f'{100*improvement(c,sid):+.2f}%' for c in conditions)+' |')
    text+=['','## 3. 七个预定臂全部报告','',
      '下表仍为确认集固定源 MSE；B 与其他臂不得替换预先指定主臂。',
      '| 方法 | 原始 | −1 mm | +1 mm | −3 mm | +3 mm |','|---|---:|---:|---:|---:|---:|']
    for arm in p['native']:text.append('| '+arm+' | '+' | '.join(f"{p[c][arm]['source_MSE_mm2']:.6f}" for c in conditions)+' |')
    text+=['','## 4. 改善与损伤比例','',
      '改善/损伤阈值为距离变化超过0.1mm，以下按全部固定源支持点行汇总；其余含KEEP与小变化。',
      '这是点行诊断，不是独立场景样本数。接受比例的分母是全部输入行，不等于实际非零移动比例。',
      '| 输入 | 方法 | 改善比例 | 损伤比例 | 接受比例 |','|---|---|---:|---:|---:|']
    for c in conditions:
        for arm,v in diagnostic[c].items():text.append(f"| {names[c]} | {arm} | {v['improved_fraction']:.2%} | {v['harmed_fraction']:.2%} | {v['accepted_fraction']:.2%} |")
    text+=['','## 5. 覆盖、集合误差与尾部','',
      '| 输入 | 方法 | E_sym mm | recall@1mm | 源点 P95 mm |','|---|---|---:|---:|---:|']
    for c in conditions:
        for arm in ('identity','post_A_keep'):
            x=p[c][arm];text.append(f"| {names[c]} | {arm} | {x['E_sym_mm']:.6f} | {x['recall_1mm']:.2%} | {x['source_p95_mm']:.6f} |")
    text+=['','## 6. 按预定标准判定','',
      f"- 原始输入验收：{'通过' if native['pass'] else '未通过'}。要求三场景都改善、总体至少5%，并满足recall和P95代价限制。",
      f"- −3 mm 恢复验收：{'通过' if recovery['minus3']['pass'] else '未通过'}。",
      f"- +3 mm 恢复验收：{'通过' if recovery['plus3']['pass'] else '未通过'}。",
      '- 官方COLMAP未运行，不能宣称超越成熟外部方法。',
      '- 尚未量化参考和坐标的计量不确定度；几何得分不等于真实薄层身份保持。',
      '- 评价采用冻结局部窗口、官方ObsMask和0.8mm参考体素；不是DTU官方整场景排行榜分数。',
      '- 三个确认场景来自相同DTU/GeoSVR链，不是跨来源或普适性证明。',
      '- 本轮不修改方法或阈值；所有失败与次要臂均保留。',
      '- 默认仍为identity，不因扰动恢复或某个次要臂的局部胜出而部署新方法。','',
      '## 7. 实施与复核','',
      '- 四场景228个输入成员解包；参考仅选择性下载12个成员。',
      '- 新adapter重放旧八ROI有序视图完全一致；四场景都通过相机/单位检查。',
      '- 5项评价器反例检查通过；16个native ROI用Open3D独立复核SciPy距离。',
      f"- 最大独立NN距离差：{max(x['max_abs_mm'] for x in d['independent_NN_checks']):.3g} mm。",
      '- 构造用CPU；三确认场景并行，计时是每场景包含七臂的完整构造，不是单独主方法的推理成本。','',
      '| 场景 | 角色 | 构造秒数 | 本进程峰值RSS MiB |','|---|---|---:|---:|']
    for sid in (40,55,65,69):
        path=ROOT/('adaptation' if sid==40 else 'confirmation')/f'scan{sid}'
        x=json.loads((path/'SUMMARY.json').read_text());text.append(f"| {sid} | {'适配' if sid==40 else '确认'} | {x['wall_seconds']:.2f} | {x['peak_own_rss_mib']:.1f} |")
    text+=['','原始结果：`evaluation/METRICS.csv`；汇总：`evaluation/SUMMARY.json`；',
      '验收细目：`EXPERIMENT_DECISION.json`；每场景目录保存ROI照片图、逐案例PLY、特征、决策与封存哈希。',
      '适配场景的数值在原始表/per_scene中单列，不混入上述确认均值。','',
      '## 8. 尚未闭合的工作','',
      '官方基线及同信息适配器、干净环境从数据到结果的一键复现尚待完成；原有方法包验证不替代这两项。',
      '先报告本轮确认事实，再据此决定是以条件恢复/失效边界收尾，还是有必要另立新方法命题。',
      '不在本轮确认结果上悄悄调参、替换场景、重新命名赢家。','']
    with (ROOT/'EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(text))
    print(json.dumps(dict(native=native,recovery=recovery),indent=2))
if __name__=='__main__':main()
