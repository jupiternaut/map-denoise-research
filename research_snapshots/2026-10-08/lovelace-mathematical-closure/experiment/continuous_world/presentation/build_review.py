"""Build the offline Chinese review and scientific SVG from confirmed archives.

This presentation helper does not import or edit the algorithm. Aggregation
uses exact Fractions; only display and plot coordinates convert to floats.
"""

from collections import defaultdict
from fractions import Fraction as Q
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "confirmation-v1"
FAMILIES = {"lovelace_triangle": "LOVELACE · 孤立 M4 面片", "rectangle": "矩形 · 解析基线"}
METHODS = {"certificate": "连续外包证书", "keep": "始终 KEEP", "nominal_grid": "名义相机网格"}
STATES = {"correct": "正确初点", "minus60": "初点 −60 mm", "plus60": "初点 +60 mm"}


def q(value):
    return Q(value["numerator"], value["denominator"]) if isinstance(value, dict) else Q(value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files():
    paths = list(ROOT.glob("*.py")) + [ROOT / "EXPERIMENT_PLAN.md", ROOT / "THEORY.md"]
    paths += [p for p in (ROOT.parent / "finite_world").rglob("*") if p.is_file()]
    return {str(p): digest(p) for p in paths}


def read_data():
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((SOURCE / "VERIFICATION.json").read_text(encoding="utf-8"))
    scenes = [json.loads(line) for line in (SOURCE / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    groups = defaultdict(list)
    counters = {"correct_keep": 0, "offset_improved": 0, "damage": 0, "nominal_correct_damage": 0}
    max_offset_error = Q(0)
    compact = []
    for scene in scenes:
        truth = q(scene["truth_depth_mm"])
        radius = str(q(scene["camera_radius_mm"]))
        hull = tuple(q(v) for v in scene["outer"]["hull"])
        trace = {node["id"]: node for node in scene["outer"]["trace"]}
        retained = [[str(q(trace[i]["lo"])), str(q(trace[i]["hi"]))] for i in scene["outer"]["retained_ids"]]
        decision_rows = []
        for decision in scene["decisions"]:
            incumbent = q(decision["incumbent"])
            output = q(decision["certificate"]["output"])
            nominal = q(decision["baselines"]["nominal_grid"])
            before, after = abs(incumbent - truth), abs(output - truth)
            if decision["state"] != "correct":
                max_offset_error = max(max_offset_error, after)
            counters["correct_keep"] += int(decision["state"] == "correct" and output == incumbent
                                             and decision["certificate"]["status"] == "KEEP")
            counters["offset_improved"] += int(decision["state"] != "correct" and after < before)
            counters["damage"] += int(after > before)
            counters["nominal_correct_damage"] += int(decision["state"] == "correct" and nominal != truth)
            for method, estimate in (("certificate", output), ("keep", incumbent), ("nominal_grid", nominal)):
                groups[(scene["kind"], radius, decision["state"], method)].append((abs(estimate - truth), hull[1] - hull[0]))
            decision_rows.append({"state": decision["state"], "incumbent": str(incumbent),
                                  "output": str(output), "nominal": str(nominal),
                                  "status": decision["certificate"]["status"],
                                  "gain_lower": str(q(decision["certificate"]["gain_lower"]))})
        compact.append({"id": scene["scene_id"], "kind": scene["kind"], "radius": radius,
                        "truth": str(truth), "seed": scene["seed"],
                        "offsets": [str(q(v)) for v in scene["actual_offsets_mm"]],
                        "epsilon": str(q(scene["epsilon"])), "hull": [str(v) for v in hull],
                        "retained": retained, "boxes": scene["outer"]["evaluated_boxes"],
                        "decisions": decision_rows})
    rows = []
    for source_row in summary["rows"]:
        key = (source_row["kind"], source_row["camera_radius_mm"], source_row["state"], source_row["method"])
        entries = groups[key]
        mae = sum((e for e, _ in entries), Q(0)) / len(entries)
        mse = sum((e * e for e, _ in entries), Q(0)) / len(entries)
        width = sum((w for _, w in entries), Q(0)) / len(entries)
        if (len(entries), mae, mse, width) != (source_row["count"], q(source_row["mae_mm"]),
                                               q(source_row["mse_mm2"]), q(source_row["mean_hull_width_mm"])):
            raise ValueError(f"Scene aggregation differs from summary for {key}")
        rows.append({**{k: source_row[k] for k in ("kind", "camera_radius_mm", "state", "method", "count", "improved", "worse", "same", "moved")},
                     "mae": str(mae), "mse": str(mse), "width": str(width)})
    counts = verification["counts"]
    if (len(scenes), sum(len(s["decisions"]) for s in scenes)) != (summary["scene_configurations"], summary["decisions"]):
        raise ValueError("Scene/decision counts differ from summary")
    if (counters["correct_keep"], counters["offset_improved"], counters["damage"]) != (
            counts["correct_KEEP"], counts["offset_improved"], counts["certificate_damage"]):
        raise ValueError("Confirmation counters differ from VERIFICATION")
    if verification["passed"] is not True or verification["failures"]:
        raise ValueError("Confirmation verification is not a passing archived check")
    return {"rows": rows, "scenes": compact, "counts": counters, "max_offset_error": str(max_offset_error),
            "scene_count": len(scenes), "decision_count": summary["decisions"],
            "verification_scope": verification["scope"],
            "source_hashes": {name: digest(SOURCE / name) for name in ("summary.json", "scenes.jsonl", "VERIFICATION.json")}}


def make_svg(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "svg.fonttype": "none",
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    radii = ("0", "1/5", "1")
    xs = [float(q(x)) for x in radii]
    for row_index, family in enumerate(FAMILIES):
        values = {}
        for radius in radii:
            group = [r for r in data["rows"] if r["kind"] == family and r["camera_radius_mm"] == radius]
            values[radius] = {}
            for method in ("certificate", "nominal_grid"):
                selected = [r for r in group if r["method"] == method and r["state"] != "correct"]
                values[radius][method] = sum(q(r["mae"]) * r["count"] for r in selected) / sum(r["count"] for r in selected)
            values[radius]["width"] = next(q(r["width"]) for r in group if r["method"] == "certificate")
        axis = axes[row_index, 0]
        for method, color, marker, linestyle in (("certificate", "#087b77", "o", "-"), ("nominal_grid", "#b96022", "s", "--")):
            ys = [float(values[r][method]) for r in radii]
            axis.plot(xs, ys, color=color, marker=marker, linestyle=linestyle, linewidth=1.7, label=METHODS[method])
            for x, y in zip(xs, ys):
                axis.annotate(f"{y:.3f}", (x, y), xytext=(0, 7 if method == "certificate" else 10),
                              textcoords="offset points", ha="center", color=color, fontsize=8)
        axis.set_title(FAMILIES[family] + " · 偏移初点 MAE", loc="left", fontweight="bold")
        axis.set_ylabel("MAE (mm)")
        axis.set_ylim(bottom=0)
        axis.legend(frameon=False, loc="upper left", fontsize=9)
        width_axis = axes[row_index, 1]
        ys = [float(values[r]["width"]) for r in radii]
        width_axis.plot(xs, ys, color="#285778", marker="o", linewidth=1.7)
        for x, y in zip(xs, ys):
            width_axis.annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        width_axis.set_title(FAMILIES[family] + " · 平均外包凸包宽度", loc="left", fontweight="bold")
        width_axis.set_ylabel("Width (mm)")
        width_axis.set_ylim(0, max(ys) * 1.22)
        for panel in (axis, width_axis):
            panel.set_xticks(xs, ["0", "0.2", "1"])
            panel.set_xlabel("侧相机横向误差半径 η (mm)")
            panel.yaxis.set_major_locator(MaxNLocator(nbins=5))
            panel.grid(axis="y", color="#e1e5e8", linewidth=.6)
            panel.set_axisbelow(True)
    fig.suptitle("连续深度与有界侧相机偏差：冻结确认集", x=.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(.06, .91, f"{data['scene_count']} 场景配置 / {data['decision_count']} 次决策；每个模型族与 η 包含 12 配置；MAE 合并 ±60 mm 初点状态。", fontsize=10, color="#43505b")
    fig.text(.06, .066, "各 η 使用不同随机噪声，不是严格配对实验；连线只连接条件均值。始终 KEEP 的偏移 MAE = 60 mm。", fontsize=9, color="#43505b")
    fig.text(.06, .041, f"外包不是置信区间；均值不是逐场景上界。偏移初点的最大证书残余绝对误差 = {float(q(data['max_offset_error'])):.4f} mm。", fontsize=9, color="#43505b")
    fig.text(.06, .017, "范围：孤立 M4 面片 / 解析矩形；540–660 mm；只改变侧相机 x；毫米为归一化模型尺度，未经物理校准。", fontsize=9, color="#43505b")
    fig.subplots_adjust(left=.075, right=.965, top=.86, bottom=.14, hspace=.45, wspace=.28)
    fig.savefig(ROOT / "results.svg", format="svg", metadata={"Date": None,
                "Description": "Derived from confirmation-v1 summary and scenes; exact Fraction aggregation; conservative outer sets, not confidence intervals."})
    plt.close(fig)
    return (ROOT / "results.svg").read_text(encoding="utf-8")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LOVELACE · 连续数学世界确认结果</title>
<style>
:root{color-scheme:light;--ink:#172e3b;--muted:#58707b;--line:#d9e2e6;--teal:#087b77;--orange:#b96022}*{box-sizing:border-box}body{margin:0;background:#f6f8f7;color:var(--ink);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:38px 24px 54px}h1{font-size:clamp(25px,4vw,39px);letter-spacing:-.035em;margin:5px 0 12px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:0 0 8px}p{margin:8px 0}a{color:#126986}small,.muted{color:var(--muted)}.eyebrow{font:12px/1.5 ui-monospace,monospace;letter-spacing:.08em;color:var(--teal)}.scope{max-width:940px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric{background:white;padding:20px;border:1px solid var(--line);border-radius:8px}.metric strong{display:block;font-size:36px;line-height:1.2;color:var(--teal)}.metric.warn strong{color:var(--orange)}section{background:#fff;border:1px solid var(--line);border-radius:8px;padding:24px;margin:20px 0}.filters{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}label{display:grid;gap:5px;font-size:13px}select{font:inherit;min-width:155px;padding:8px 10px;background:white;border:1px solid #91a5ae;border-radius:5px;color:var(--ink)}:focus-visible{outline:3px solid #35aba4;outline-offset:3px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto}.details{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.details div{border-left:3px solid #dae8e9;padding-left:12px}.details b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.callout{background:#eef7f5;border-left:3px solid var(--teal);padding:13px 16px;font-size:14px}.scope-note{background:#f8f3eb;border-left:3px solid #b47a35;padding:13px 16px;font-size:14px}.depth-plot{width:100%;height:165px;background:#fafcfc;border:1px solid #e2eaec;border-radius:5px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin-top:8px}.charts svg{width:100%;height:auto}.hashes{font-size:11px;font-family:ui-monospace,monospace;overflow-wrap:anywhere}details summary{cursor:pointer;color:#315c71}button{border:1px solid var(--line);background:white;color:var(--ink);padding:8px 12px;border-radius:5px;cursor:pointer}.intervals{font-family:ui-monospace,monospace;font-size:12px;max-height:110px;overflow:auto}.tag{display:inline-block;font-size:12px;border:1px solid #c4dfd9;color:var(--teal);padding:2px 7px;border-radius:4px}footer{font-size:13px;color:var(--muted)}@media(max-width:720px){main{padding:24px 14px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:18px}.details{grid-template-columns:repeat(2,1fr)}th,td{padding:8px}select{min-width:130px}}
</style>
<main><div class="eyebrow">LOVELACE / MATHEMATICAL CLOSURE / STAGE 2</div>
<h1>从离散候选，走向连续的安全修正</h1>
<p class="scope">冻结确认集将深度放在连续区间 <b>540–660 mm</b>，并允许侧相机横向位置在误差半径内连续变化。证书先保留不能排除的深度外包，只有整段外包都保证平方误差改善时才移动。</p>
<div class="metrics"><div class="metric"><strong>__KEEP__</strong>正确初点保持原值</div><div class="metric"><strong>__IMPROVE__</strong>±60 mm 偏移初点改善</div><div class="metric"><strong>__DAMAGE__</strong>证书输出伤害</div><div class="metric warn"><strong>__NOMINAL_DAMAGE__</strong>名义网格伤害正确初点</div></div>
<p class="muted">__SCENES__ 场景配置，__DECISIONS__ 次决策；上述合计包含 LOVELACE 面片与解析矩形两个模型族。每个模型族 / 半径有 12 场景，三种初点共享同一观测。偏移初点的最大证书残余绝对误差为 <b>__MAX_ERROR__ mm</b>；平均 MAE 不代表每个场景都达亚毫米精度。</p>
<div class="scope-note"><b>适用边界：</b>LOVELACE 部分只隔离已有 M4 网格的一张三角面，矩形另作解析基线；统一不透明常色，只包含侧相机 <i>x</i> 位置偏差，参考相机固定。毫米为归一化模型尺度，未经真实物理校准。结果不表示完整角色重建，也不覆盖旋转、光照、材质或真实相机误差。</div>
<section><h2>按条件检查结果</h2><div class="filters"><label>模型族<select id="family"></select></label><label>侧相机误差半径 η<select id="radius"></select></label><label>初点状态<select id="state"></select></label></div>
<div class="table-wrap"><table><thead><tr><th>方法</th><th class="num">决策数</th><th class="num">MAE / mm</th><th class="num">改善 / 伤害</th><th class="num">平均外包宽度 / mm</th></tr></thead><tbody id="statistics"></tbody></table></div>
<p class="muted">外包宽度属于证书搜索，基线没有各自的外包。显示值为四舍五入；悬停查看归档中的精确有理数。</p>
<label>选定场景<select id="scene"></select></label><p id="scene-description" class="muted"></p>
<div class="details"><div><small>验收真值</small><b id="truth"></b></div><div><small>输入初点</small><b id="initial"></b></div><div><small>证书输出</small><b id="output"></b></div><div><small>证书决策</small><b id="status"></b></div></div>
<svg id="depthplot" class="depth-plot" viewBox="0 0 1000 165" role="img" aria-label="选定场景的真值、初点、输出和保留深度区间"></svg>
<div class="legend"><span>蓝色 T：验收真值</span><span>橙色 I：初点</span><span>绿色 C：证书输出</span><span>紫色 N：名义网格输出</span><span>实绿色段：未排除区间</span></div>
<p id="scene-error"></p><p id="hull" class="muted"></p><details><summary>查看全部未排除区间与证书下界</summary><p id="gain" class="muted"></p><div id="intervals" class="intervals"></div></details>
<div class="callout"><b>外包不是置信区间。</b>未排除不等于存在一组相机参数能解释该深度；保守区间可能包含伪候选。真值只有在声明的几何、相机和观测误差预算覆盖它时才保证保留。证书在这个外包上采取保守动作。</div>
</section>
<section><h2>可导出的科研图</h2><p class="muted">MAE 合并正、负 60 mm 偏移状态；外包宽度按场景配置平均。两种模型族分开绘制，半径按数值排序。不同半径使用不同随机噪声，因此不是严格配对实验；图中连线只连接条件均值，不能单独解释成相机误差半径的因果效应。</p><div class="charts">__SVG__</div><a href="results.svg" download>下载独立 SVG</a></section>
<section><h2>结果与核验的来源</h2><p><span class="tag">归档分区与标量证据：通过</span></p><p>本页读取 confirmation-v1 的 <a href="outputs/confirmation-v1/VERIFICATION.json">VERIFICATION.json</a>，其核验复用几何模型核，未宣称独立重写几何证明。页面构建另外用场景逐项重聚合，核对摘要中的 MAE、MSE、外包宽度及计数。</p>
<p><a href="outputs/confirmation-v1/summary.json">确认集摘要</a> · <a href="outputs/confirmation-v1/scenes.jsonl">场景与决策原始记录</a> · <a href="outputs/replay-v1/summary.json">另一次回放的原始摘要</a> · <a href="outputs/confirmation-v1/REPRODUCIBILITY.json">重跑核验原始报告</a> · <a href="outputs/confirmation-v1/AUDIT.json">源码审计原始报告</a> · <a href="REPORT.zh.md">中文报告</a> · <a href="THEORY.md">理论与范围</a> · <a href="EXPERIMENT_PLAN.md">冻结实验计划</a></p>
<p class="muted">本页的通过标签只对应所读取的 VERIFICATION；源码审计与重跑结论请查看各自原始报告。另有违反观测误差预算的控制例，演示覆盖失效后可能伤害真值；该例不计入 216 次确认决策。</p>
<details><summary>源数据 SHA-256</summary><div class="hashes">__HASHES__</div></details></section>
<footer>图表的小数仅用于展示；算法及归档使用精确 Fraction。构建脚本：presentation/build_review.py。整个页面可离线打开，无网络依赖。</footer></main>
<script id="data" type="application/json">__DATA__</script>
<script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent),names={lovelace_triangle:'LOVELACE · 孤立 M4 面片',rectangle:'矩形 · 解析基线'},states={correct:'正确初点',minus60:'初点 −60 mm',plus60:'初点 +60 mm'},methods={certificate:'连续外包证书',keep:'始终 KEEP',nominal_grid:'名义相机网格'};
const el=id=>document.getElementById(id),num=x=>{const[a,b='1']=String(x).split('/');return Number(a)/Number(b)},fmt=(x,d=4)=>num(x).toFixed(d),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function options(id,items){el(id).replaceChildren(...items.map(([value,label])=>{const o=document.createElement('option');o.value=value;o.textContent=label;return o}))}
options('family',Object.entries(names));options('radius',[['0','0 mm'],['1/5','0.2 mm'],['1','1 mm']]);options('state',Object.entries(states));el('state').value='minus60';
function updateFilter(){const family=el('family').value,radius=el('radius').value,state=el('state').value;const rows=data.rows.filter(r=>r.kind===family&&r.camera_radius_mm===radius&&r.state===state);el('statistics').innerHTML=['certificate','keep','nominal_grid'].map(method=>{const r=rows.find(r=>r.method===method);return `<tr><td>${methods[method]}</td><td class="num">${r.count}</td><td class="num" title="${esc(r.mae)}">${fmt(r.mae)}</td><td class="num">${r.improved} / ${r.worse}</td><td class="num" title="${esc(r.width)}">${method==='certificate'?fmt(r.width):'—'}</td></tr>`}).join('');const scenes=data.scenes.filter(s=>s.kind===family&&s.radius===radius).sort((a,b)=>num(a.truth)-num(b.truth)||a.seed-b.seed);options('scene',scenes.map(s=>[s.id,`真值 ${fmt(s.truth,3)} mm · seed ${s.seed}`]));updateScene()}
function updateScene(){const s=data.scenes.find(s=>s.id===el('scene').value),d=s.decisions.find(d=>d.state===el('state').value);for(const[id,v]of[['truth',s.truth],['initial',d.incumbent],['output',d.output]]){el(id).textContent=fmt(v)+' mm';el(id).title=v}el('status').textContent=d.status;el('scene-description').textContent=`实际相机偏差 [${s.offsets.join(', ')}] mm；ε = ${s.epsilon}；计算 ${s.boxes} 个深度盒。真值仅作为验收字段。`;el('scene-error').textContent=`绝对误差：初点 ${fmt(String(Math.abs(num(d.incumbent)-num(s.truth))))} mm → 证书 ${fmt(String(Math.abs(num(d.output)-num(s.truth))))} mm；名义网格 ${fmt(String(Math.abs(num(d.nominal)-num(s.truth))))} mm。`;el('hull').textContent=`外包凸包 [${fmt(s.hull[0])}, ${fmt(s.hull[1])}] mm，宽度 ${(num(s.hull[1])-num(s.hull[0])).toFixed(4)} mm；${s.retained.length} 个未排除叶区间。`;el('gain').textContent=`归档平方误差改善下界：${d.gain_lower} mm²（${fmt(d.gain_lower)}）；KEEP 时不要求强行移动。`;el('intervals').textContent=s.retained.map(p=>`[${p.join(', ')}]`).join(' ∪ ');const vals=[540,660,num(s.truth),num(d.incumbent),num(d.output),num(d.nominal)],lo=Math.min(...vals)-8,hi=Math.max(...vals)+8,x=v=>50+900*(v-lo)/(hi-lo);let svg=`<rect x="${x(540)}" y="52" width="${x(660)-x(540)}" height="35" fill="#edf1f2"/><line x1="50" x2="950" y1="88" y2="88" stroke="#9cadb5"/>`;for(const p of s.retained)svg+=`<rect x="${x(num(p[0]))}" y="60" width="${Math.max(.8,x(num(p[1]))-x(num(p[0])))}" height="20" fill="#80c7b8"/>`;for(const[value,label,color,y]of[[num(s.truth),'T','#285778',36],[num(d.incumbent),'I','#b96022',112],[num(d.output),'C','#087b77',135],[num(d.nominal),'N','#725294',153]])svg+=`<line x1="${x(value)}" x2="${x(value)}" y1="49" y2="92" stroke="${color}" stroke-width="2"/><text x="${x(value)}" y="${y}" fill="${color}" text-anchor="middle" font-size="13">${label} ${value.toFixed(3)}</text>`;svg+=`<text x="${x(540)}" y="102" text-anchor="middle" fill="#6e8089" font-size="11">540</text><text x="${x(660)}" y="102" text-anchor="middle" fill="#6e8089" font-size="11">660 mm</text>`;el('depthplot').innerHTML=svg}
for(const id of['family','radius','state'])el(id).addEventListener('change',updateFilter);el('scene').addEventListener('change',updateScene);updateFilter();
</script></html>'''


def main():
    before = protected_files()
    data = read_data()
    svg = make_svg(data)
    # Remove only the standalone SVG declaration/doctype before embedding it.
    svg_inline = svg[svg.index("<svg"):]
    page = TEMPLATE
    replacements = {"__KEEP__": data["counts"]["correct_keep"], "__IMPROVE__": data["counts"]["offset_improved"],
                    "__DAMAGE__": data["counts"]["damage"], "__NOMINAL_DAMAGE__": data["counts"]["nominal_correct_damage"],
                    "__SCENES__": data["scene_count"], "__DECISIONS__": data["decision_count"],
                    "__MAX_ERROR__": f"{float(q(data['max_offset_error'])):.4f}",
                    "__SVG__": svg_inline,
                    "__HASHES__": "".join(f"<p>{html.escape(name)}: {value}</p>" for name, value in data["source_hashes"].items()),
                    "__DATA__": json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")}
    for marker, value in replacements.items():
        page = page.replace(marker, str(value))
    (ROOT / "review.html").write_text(page, encoding="utf-8")
    if before != protected_files():
        raise RuntimeError("Protected algorithm/plan/theory/stage-one files changed during presentation build")
    print(json.dumps({"status": "built", "review": str(ROOT / "review.html"), "svg": str(ROOT / "results.svg"),
                      "counts": data["counts"], "source_aggregation": "exactly_matches_summary",
                      "protected_files_unchanged": len(before)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
