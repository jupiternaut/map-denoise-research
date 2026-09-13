"""Read the immutable result table; produce plots and a reproducible readout."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from schema import read_patch,read_evaluation

LABELS={
 'identity':'Input', 'xyz_mixture':'Old XYZ', 'fast':'Old fast',
 'open3d_icp_then_xyz':'Open3D ICP + XYZ', 'graph_local':'Local surface only',
 'graph_shared':'Shared-bias surface graph', 'measure_balanced':'Balanced transport',
 'measure_unbalanced':'Unbalanced transport', 'measure_balanced_local':'Balanced + local',
 'measure_unbalanced_local':'Unbalanced + local'}


def med(rows,key):
    values=[float(r[key]) for r in rows if r.get(key) not in ('',None)]
    return float(np.median(values)) if values else float('nan')


def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+[
        '| '+' | '.join(str(x) for x in r)+' |' for r in rows]


def run():
    parser=argparse.ArgumentParser(); parser.add_argument('run_dir',type=Path)
    args=parser.parse_args(); run_dir=args.run_dir.resolve()
    if (run_dir/'REPORT.md').exists(): raise FileExistsError('report exists; preserve it')
    rows=list(csv.DictReader((run_dir/'RESULTS.csv').open()))
    summary=json.loads((run_dir/'SUMMARY.json').read_text())
    methods=summary['methods']
    full=[r for r in rows if r['split']=='synthetic' and r['sampling']=='full' and r.get('geometry_scored')=='True' and r['ok']=='True']
    plots=run_dir/'figures'; plots.mkdir(exist_ok=False)
    case='dual_g4_s912101_b4'
    choices=[next(r for r in full if r['case']==case and r['method']==m) for m in methods]
    ev=read_evaluation(Path(choices[0]['source']))
    p,_=read_patch(Path(choices[0]['source']))
    out=[]
    for row in choices:
        with np.load(row['output']) as a: out.append(a['xyz_world_canonical']*1000.)
    zrange=(min(q[:,2].min() for q in out)-1,max(q[:,2].max() for q in out)+1)
    fig,axs=plt.subplots(2,5,figsize=(19,7),sharex=True,sharey=True)
    for ax,row,q in zip(axs.ravel(),choices,out):
        ax.scatter(q[:,0],q[:,2],s=3,c=p['scan_id'],cmap='tab10',alpha=.60,linewidths=0)
        for z,x0,x1,y0,y1 in ev['surface_rectangles_mm']:
            ax.plot([x0,x1],[z,z],color='black',lw=1.2,ls='--')
        ax.set_title(LABELS[row['method']]+'\nMAE '+f"{float(row['surface_accuracy_mean_mm']):.3f} mm",fontsize=10)
        ax.set_ylim(*zrange); ax.set_xlim(-65,65); ax.grid(alpha=.15)
        ax.set_xlabel('World x [mm]'); ax.set_ylabel('World z [mm]')
    fig.suptitle('Development example: 4 mm true step, 4 mm scan-bias RMS, 1 mm random noise\nAll 768 points shown; scan colours, dashed true surfaces; same axes',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.9)); figure=plots/'world_step_4mm.png'; fig.savefig(figure,dpi=150); plt.close(fig)

    selected=['fast','open3d_icp_then_xyz','graph_shared','measure_balanced_local','measure_unbalanced_local']
    groups=[(g,b) for g in (0.,2.,4.,8.) for b in (0.,4.)]
    values=np.array([[med([r for r in full if float(r['gap_mm'])==g and float(r['bias_rms_mm'])==b and r['method']==m],
                              'surface_accuracy_mean_mm') for g,b in groups] for m in selected])
    fig,ax=plt.subplots(figsize=(11,4.5))
    im=ax.imshow(values,cmap='magma_r',vmin=0,vmax=max(1.2,values.max()))
    ax.set_xticks(range(len(groups)),[f'gap {g:g}\nbias {b:g}' for g,b in groups])
    ax.set_yticks(range(len(selected)),[LABELS[m] for m in selected])
    for i in range(len(selected)):
        for j in range(len(groups)): ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>.6 else 'black')
    ax.set_title('Median surface error [mm], three exposed seeds per condition (lower is better)')
    fig.colorbar(im,ax=ax,label='mm'); fig.tight_layout(); fig.savefig(plots/'condition_errors.png',dpi=150); plt.close(fig)

    graph=med([r for r in full if r['method']=='graph_shared'],'surface_accuracy_mean_mm')
    old=med([r for r in full if r['method']=='fast'],'surface_accuracy_mean_mm')
    doc=['# 两端构造 V3：原型与开发实验报告','',
         '## 结论','',
         f'已得到两个可运行构造及配对消融。24 个可评分原合成案例的表面误差中位数：旧 fast {old:.6f} mm，共享偏差表面图 {graph:.6f} mm；两个中位数之比对应下降 {(1-graph/old)*100:.2f}%。这是公开开发案例上的信号，不是独立确认，也不是逐案例改善率的平均。',
         '', '关系图端更值得继续投入；不平衡运输在本轮固定预算下没有优于平衡运输。真实片区中旧方法的大幅改写明显减少，但新方法恢复到原测量云的误差仍接近 identity，尚未建立真实几何精度收益。',
         '', '## 实际交付与范围','',
         f'- 完整运行 {summary["ok"]}/{summary["rows"]} 个输出；27 原合成案例（24 唯一真值可评分、3 标量歧义）及其 12 个子采样/坐标变换诊断；2 个真实场景、6 片区×3输入。570 输出不是570场景。',
         '- 新估计器只接收 XYZ、扫描 ID、提供的 sigma；无 GT 法向、关联、逆扰动或 clean 参考。此轮未使用射线、盲估 sigma、全六自由度位姿或 CUDA。',
         f'- wall time {summary["elapsed_s"]:.3f} s，方法累计 {summary["method_total_s"]:.3f} s；Python 进程峰值 RSS {summary["peak_process_rss_kib"]} KiB（包含 Open3D 导入，不是 GPU 显存）。报告、图和独立审查不在该 wall time 内。',
         f'- {summary["protected_files"]} 个受保护历史文件哈希未变；预留种子仍未使用。',
         '', '## 算法不是同一公式加参数','',
         'A：同帧近邻差分估计共同方向；局部单元可有 1/2 个表面；不同单元通过同一扫描的共同偏差连接，交替估计局部关联、表面和扫描偏差。local_only 使用相同法向、局部拟合和支持条件，仅关闭共享偏差。',
         '', 'B：把扫描表示为实测位置、局部方向和采样密度代理质量；balanced/UOT 求跨站软匹配。匹配只驱动整扫描共同法向平移，不用运输重心替换点坐标。这样纠偏阶段严格保持同一扫描内任意点对相对向量。之后再独立测试相同 local_only 后置滤波。它仍是表面测度启发的原型，不是完整 varifold 配准或新的层论算法。',
         '', f'完整方程与有限命题：[CONSTRUCTION.md]({HERE/"CONSTRUCTION.md"})。',
         '', '## 1. 原合成全条件：表面误差','',
         '每格三种子的中位数，单位 mm，越小越好。sigma=1 mm；bias 是逐扫描共同法向偏差 RMS。']
    condition_rows=[]
    for j,(g,b) in enumerate(groups):
        condition_rows.append([f'{g:g}',f'{b:g}']+[f'{values[i,j]:.4f}' for i in range(len(selected))])
    doc+=['']+table(['真层距','偏差 RMS']+[LABELS[m] for m in selected],condition_rows)
    doc+=['', '**必须保留的反例：** 无偏差单平面旧 fast 约0.043 mm，新图端约0.160 mm；局部化牺牲了简单场景中的全局平均收益。不能写成全面优于旧方法。',
          '', f'![全部条件]({plots/"condition_errors.png"})',
          '', '## 2. 几何和成本一起看','',
          '下表对原24例分别取中位数。sample coverage 依赖原采样，不是连续表面完整度；法向角与同XY拟合间距使用GT来源组，仅作评估。']
    aggregate=[]
    for m in methods:
        rr=[r for r in full if r['method']==m]
        aggregate.append([LABELS[m],f'{med(rr,"surface_accuracy_mean_mm"):.4f}',f'{med(rr,"matched_point_rms_mm"):.4f}',
                          f'{med(rr,"reference_sample_coverage_1mm"):.3f}',f'{med(rr,"source_surface_tilt_mean_deg"):.3f}',f'{1000*med(rr,"method_seconds"):.2f}'])
    doc+=['']+table(['方法','表面 MAE mm','来源对应 RMS mm','1 mm采样覆盖','来源表面偏转°','单例 ms'],aggregate)
    doc+=['', '每例一次 warm-call、固定单线程 CPU 的探索计时；不同方法预处理和工作量不同，不是优化后性能下界。Open3D ICP 加自研后处理仅是明确对照，不能代表 JRMPC/BALM 或整个学术前沿。',
          '', '## 3. 同输入消融与观测变化','']
    paired=[]
    for sampling in ('full','density','partial_overlap'):
        for m in ('graph_local','graph_shared','measure_balanced_local','measure_unbalanced_local'):
            rr=[r for r in rows if r['sampling']==sampling and r['method']==m and r.get('geometry_scored')=='True' and r.get('bias_rms_mm')=='4.0']
            paired.append([sampling,LABELS[m],len(rr),f'{med(rr,"surface_accuracy_mean_mm"):.4f}'])
    doc+=table(['输入','方法','案例数','表面 MAE 中位数 mm'],paired)
    doc+=['', 'density/partial_overlap 只来自一个已曝光种子的四种场景；不等价于四个新数据集。只按当前观测横向位置和点序子采样，同步保留来源与评价数据。',
          '', '平衡/不平衡都使用24次内层、3次外层预算。平衡列归一残差接近零不代表行边缘也收敛；应以 metadata 里的双边残差判断。不平衡质量更少不意味着自动选对物理表面。',
          '', '## 4. 真实输入：减少改写，尚非真实去噪成功','']
    real_rows=[]
    for m in methods:
        rr=[r for r in rows if r['split']=='real' and r['method']==m]
        real_rows.append([LABELS[m]]+[f'{med([r for r in rr if r["sampling"]==s],"recovery_vs_measured_reference_rms_mm"):.4f}' for s in ('unpert','trans','trans_rot')])
    doc+=table(['方法','零注入改写 mm','平移后对原测量 RMS mm','平移+旋转后 RMS mm'],real_rows)
    doc+=['', '这里原测量云不是独立真值，真实片区也不是毫米级真薄层标注。新图端偏差注入后表现接近 identity，说明尚未证明它能恢复真实扰动。零注入少改写只排除了旧模型的大范围几何压平，不证明去噪收益。',
          '', '## 5. 台阶是否变成斜坡','',
          f'![所有方法同坐标输出]({figure})',
          '', '例子固定为 gap4 / bias4 / seed912101，所有点和所有方法使用同一坐标范围。虚线只在评估图表示已知合成表面，不提供给算法。图端仍有未校正/未支持点；不能称完全恢复。',
          '', 'GT、塌缩、斜坡、整体偏移100 mm负控见 NEGATIVE_CONTROLS.json；自报K变化不改变几何分数。所有旋转诊断只逆一次共同刚体换坐标，不逐层/逐帧对齐。',
          '', '## 研究判断','',
          '1. 可以继续投入：同帧关系估方向、局部表面与共享扫描偏差的组合产生了开发证据。跨片区共享量有作用，不只是换名。',
          '2. 不能从中推出：所有指标改善、真实几何改善、比对口联合配准方法更好。正常单面回归和计算成本已经可见。',
          '3. 运输端暂不淘汰：三轮外层有总位移预算上限，必须通过质量—计算预算对照区分求解不足和关联构造不足；若增加预算也不改善，再改关联。',
          '4. 下一次独立确认前应冻结完整方法及所有选择规则；不要用这批开发结果当未见证据。真实场景后续需要局部输入/参考对齐与独立几何证据。',
          '', '## 复现','',
          f'源码目录：`{HERE}`。冻结快照在本运行 `source/`，CSV与NPZ均保留。可用项目当前入口重新跑一个新目录：','',
          '```bash',
          'env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\',
          '  /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python \\',
          f'  {HERE/"run_exploration.py"}',
          '```','',
          '运行使用原机 liekkas 和既有绝对数据/算子路径。源码快照是审计记录，不是脱离现有数据环境即可执行的便携包。']
    (run_dir/'REPORT.md').write_text('\n'.join(doc)+'\n')
    shutil.copyfile(__file__,run_dir/'REPORT_GENERATOR.py')
    (run_dir/'REPORT_GENERATOR.sha256').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')
    print(run_dir/'REPORT.md')
    print(figure)


if __name__=='__main__':run()
