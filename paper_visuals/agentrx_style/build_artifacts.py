"""Build source-faithful LaTeX tables/code panels and assemble the figure atlas.

This script reuses frozen published scores; it never runs or changes the filter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'evidence/progress_v22/runs/reconstruction-v22-qayc8gft'
METHODS = [
    ('identity', 'Identity'), ('v18', 'Frozen V18'), ('apss2', 'APSS2'),
    ('rimls2', 'RIMLS2'), ('local_plane64', 'Local plane, $k=64$'),
    ('quadratic64', 'Local quadratic, $k=64$'),
    ('multiscale_full', 'Multiscale, full update'),
    ('multiscale_consensus', 'V22 consensus (primary)'),
    ('multiscale_matched', 'Matched-RMS constant damping'),
]
PREAMBLE = r'''\documentclass[border=4pt,varwidth=170mm]{standalone}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\setmonofont{Consolas}[Scale=0.9]
\usepackage{amsmath,amssymb,xcolor,booktabs,array,graphicx,fvextra}
\usepackage{listings}
\definecolor{ink}{HTML}{252A30}
\definecolor{blue}{HTML}{3778A8}
\definecolor{orange}{HTML}{CB7736}
\definecolor{muted}{HTML}{60666C}
\definecolor{rulegray}{HTML}{BCC4CA}
\color{ink}
\setlength{\parindent}{0pt}
\newcommand{\figtitle}[2]{{\fontsize{12}{14}\selectfont\bfseries #1}\par\vspace{3pt}{\fontsize{8.5}{11}\selectfont\color{muted}#2}\par\vspace{9pt}}
\begin{document}
\begin{minipage}{167.1883mm}
'''
END = '\n\\end{minipage}\n\\end{document}\n'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    (HERE / 'data').mkdir(exist_ok=True)
    rows = json.loads((RUN / 'confirmation_RESULTS.json').read_text(encoding='utf-8'))
    summary = json.loads((RUN / 'SUMMARY.json').read_text(encoding='utf-8'))['results']['confirmation']
    assert len(rows) == 216
    assert len({r['case'] for r in rows}) == 24
    means = {}
    for method, _ in METHODS:
        selected = [r for r in rows if r['method'] == method]
        assert len(selected) == 24 and len({r['case'] for r in selected}) == 24
        assert all(r['status'] == 'OK' for r in selected)
        means[method] = {}
        for key in ['accuracy_mm', 'completeness_mm', 'recall', 'fscore', 'displacement_rms_mm']:
            value = math.fsum(r[key] for r in selected) / len(selected)
            assert abs(value - summary['summary'][method][key]) < 1e-12
            means[method][key] = value
    (HERE / 'data/table_means_verified.json').write_text(json.dumps(means, indent=2), encoding='utf-8')
    body = PREAMBLE + r'''\figtitle{Scan37: geometry and coverage across nine methods}{One new scene; 24 disjoint source-vertex patches, 1,024 input points per patch.}
{\fontsize{9}{12}\selectfont
\setlength{\tabcolsep}{5.3pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
\textbf{Method} & \shortstack{\textbf{MAE}$\downarrow$\\mm} & \shortstack{\textbf{Recall}$\uparrow$\\\% at 1 mm} & \shortstack{\textbf{F-score}$\uparrow$\\at 1 mm} & \shortstack{\textbf{Move RMS}\\mm} \\
\midrule
'''
    for method, label in METHODS:
        vals = means[method]
        items = [label, f"{vals['accuracy_mm']:.6f}", f"{100*vals['recall']:.3f}", f"{vals['fscore']:.6f}", f"{vals['displacement_rms_mm']:.6f}"]
        if method == 'multiscale_consensus':
            items = [r'\textcolor{blue}{\textbf{' + v + '}}' for v in items]
        body += ' & '.join(items) + ' \\\\\n'
    body += r'''\bottomrule
\end{tabular*}}\par
\vspace{8pt}
{\fontsize{8.5}{11.5}\selectfont
\textcolor{blue}{Blue identifies the preselected primary method, not the best score.}
Every metric is averaged equally over the 24 patches. Evaluation uses the fixed observed local region;
scored point counts can differ from the 1,024-point input. Recall measures reference coverage within 1 mm.
These are local patch results, not official full-scene DTU scores.}\par
\vspace{9pt}\par
{\fontsize{9}{12}\selectfont\textbf{Paired reading.}
Against identity, the primary method reduces mean MAE by 0.598\%, while recall decreases by
0.333 percentage points. Matched-RMS constant damping has a slightly lower MAE than the primary method.}
''' + END
    (HERE / 'fig03_table.tex').write_text(body, encoding='utf-8')
    code_path = ROOT / 'reconstruction_v22/operator.py'
    lines = code_path.read_text(encoding='utf-8').splitlines(keepends=True)
    excerpt = ''.join(lines[118:138])
    assert 'anchor = quadratic[control_k]' in excerpt and 'consensus = points +' in excerpt
    (HERE / 'data/operator_excerpt.py').write_text(excerpt, encoding='utf-8')
    comparison = {'identity': summary['comparisons']['identity']}
    (HERE / 'data/identity_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n', encoding='utf-8')
    code = PREAMBLE + r'''\figtitle{V22: the implemented multiscale correction}{Verbatim source excerpt: reconstruction\_v22/operator.py, lines 119--138.}
\lstset{language=Python,basicstyle=\ttfamily\fontsize{8}{10.7}\selectfont,
  keywordstyle=\color{blue},commentstyle=\color{muted},stringstyle=\color{orange},
  breaklines=true,breakatwhitespace=true,breakautoindent=true,
  postbreak=\mbox{\textcolor{muted}{\tiny$\hookrightarrow$}\space},
  keepspaces=true,showstringspaces=false,columns=fullflexible,
  frame=single,rulecolor=\color{rulegray},framerule=0.35pt,framesep=6pt,
  xleftmargin=6pt,xrightmargin=6pt,numbers=none,tabsize=4}
\lstinputlisting{data/operator_excerpt.py}
\vspace{5pt}
{\fontsize{10}{13}\selectfont
The applied displacement is $\alpha_i d_i\boldsymbol n_{i,64}$, where
\[\alpha_i=\frac{v_{n,i}}{v_{n,i}+v_{p,i}+v_{s,i}+\varepsilon_i}.\]
Shared observations are not treated as independent evidence: prediction variance is not divided
by the number of scales. The factor $\alpha$ is heuristic damping, not a calibrated probability or a rejection gate.}
\vspace{9pt}\par
{\fontsize{9}{12}\selectfont\textbf{Published comparison against identity}\quad\textcolor{muted}{SUMMARY.json / results.confirmation.comparisons}}
\vspace{4pt}
\VerbatimInput[fontsize=\footnotesize,frame=single,rulecolor=\color{rulegray},
  framesep=6pt,breaklines=true,breakanywhere=true,tabsize=2]{data/identity_comparison.json}
\vspace{5pt}\par
{\fontsize{8.5}{11}\selectfont\texttt{passed: false} means the predeclared composite improvement criterion was not met;
it does not mean execution failed. Gains and recall differences in this JSON are fractions, not percentages.}
''' + END
    (HERE / 'fig04_code.tex').write_text(code, encoding='utf-8')
    source_files = [ROOT/'reconstruction_v22/operator.py', ROOT/'reconstruction_v22/PROTOCOL.md',
                    ROOT/'reconstruction_v22/REPORT.md', RUN/'confirmation_RESULTS.json', RUN/'SUMMARY.json', RUN/'FINAL_AUDIT.json']
    provenance = {
        'repository': 'https://github.com/jupiternaut/map-denoise-research',
        'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_files': [{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p)} for p in source_files],
        'rows':len(rows),'cases':24,'methods':9,
        'aggregation':'equal arithmetic mean of the 24 patch-level metrics per method',
        'code_excerpt':{'path':'reconstruction_v22/operator.py','first_line':119,'last_line':138,'text_sha256':digest(HERE/'data/operator_excerpt.py')},
        'validation':'Saved metrics recomputed for presentation; original geometric experiment not rerun.',
    }
    (HERE/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Verified 216 saved score rows, 45 summary means, and exact 20-line source excerpt.')


def compile_tex(tectonic: str, name: str) -> None:
    proc = subprocess.run([tectonic, '-X', 'compile', name, '--keep-logs'], cwd=HERE,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8',errors='replace')
    (HERE/'build-logs').mkdir(exist_ok=True)
    (HERE/'build-logs'/f'{Path(name).stem}.txt').write_text(proc.stdout,encoding='utf-8')
    if proc.returncode:
        print(proc.stdout[-5000:])
        raise RuntimeError(f'TeX compilation failed: {name}; see build-logs.')
    print(f'Compiled {name}; full compiler diagnostics saved in build-logs.')


def export_pdf(path: Path) -> dict:
    import fitz
    with fitz.open(path) as doc:
        assert len(doc)==1, (path,len(doc))
        p=doc[0]
        p.get_pixmap(matrix=fitz.Matrix(2.4,2.4),alpha=False).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(p.get_svg_image(text_as_path=False),encoding='utf-8')
        return {'file':path.name,'pages':len(doc),'page_size_pt':[p.rect.width,p.rect.height],
                'embedded_images':len(p.get_images()),'vector_paths':len(p.get_drawings()),
                'text_chars':len(p.get_text()),'fonts':[f[3] for f in p.get_fonts()]}


def atlas() -> None:
    import fitz
    images = ['fig01_pipeline.pdf', 'fig02_confirmation.pdf', 'fig03_table.pdf', 'fig04_code.pdf']
    titles = ['01  从局部曲面到连续阻尼','02  逐片区变化与几何覆盖权衡','03  九种方法的完整数值对照','04  真实代码与未通过的升级判据']
    captions = [
        'V22 为每个查询点建立排除自身的多尺度邻域，各自拟合局部二次曲面。不同尺度的修正投影到 k64 法向后融合，再施加连续阻尼。独立参考仅用于输出封存后的评分；该算法没有“拒绝即回退”的安全门控。',
        'scan37 新确认场景的 24 个片区。逐片区变化均与同片区 identity 配对；主候选与相同整体位移 RMS 的常数阻尼同时展示。汇总均为片区等权均值，不能当作全场景官方 DTU 分数，也不代表独立场景数量为 24。',
        '216 条已发布记录（24 片区 × 9 方法）直接重算。距离单位为 mm，召回率阈值为 1 mm。蓝色仅标识事前主候选。MAE 的轻微下降伴随召回和 F-score 代价，且主候选没有胜过同位移 RMS 常数阻尼。',
        '上方原样摘录 operator.py 第 119–138 行，折行箭头只用于排版；下方来自已发布 SUMMARY.json 的实际比较。passed=false 表示未达到预设综合升级条件，并非程序执行失败。源码保留了“相关尺度不能虚构独立方差收益”的说明。',
    ]
    doc=fitz.open()
    for i,(name,title,caption) in enumerate(zip(images,titles,captions)):
        page=doc.new_page(width=595.276,height=841.89)
        page.insert_font(fontname='CN',fontfile='C:/Windows/Fonts/simsun.ttc')
        page.insert_text((42,43),'MAP DENOISE RESEARCH  /  V22',fontsize=9,color=(.22,.47,.66))
        page.insert_text((42,72),title,fontname='CN',fontsize=19,color=(.15,.17,.19))
        page.draw_line((42,85),(553,85),color=(.73,.76,.79),width=.5)
        with fitz.open(HERE/name) as src:
            page.show_pdf_page(fitz.Rect(42,101,553,691),src,0,keep_proportion=True)
        left=page.insert_textbox(fitz.Rect(42,711,553,797),caption,fontname='CN',fontsize=10.2,lineheight=1.55,color=(.23,.25,.27))
        assert left>=0, ('caption overflow',i,left)
        page.insert_text((42,819),'SOURCE SNAPSHOT  a7b362e8  |  13 SEP 2026  |  STORED RESULTS, RE-PLOTTED',fontsize=7,color=(.4,.43,.45))
        page.insert_text((542,819),str(i+1),fontsize=8,color=(.4,.43,.45))
    doc.subset_fonts()
    doc.save(HERE/'map-denoise-visual-atlas.pdf', garbage=4, deflate=True)
    for i,p in enumerate(doc):
        p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(HERE/f'atlas-page-{i+1}.png')
    from PIL import Image, ImageOps, ImageDraw
    thumbnails=[]
    for i in range(4):
        im=Image.open(HERE/f'atlas-page-{i+1}.png').convert('RGB')
        im.thumbnail((596,842))
        thumbnails.append(im)
    preview=Image.new('RGB',(1240,1740),'#e8ebee')
    for i,im in enumerate(thumbnails):
        preview.paste(im,(16+(i%2)*616,16+(i//2)*860))
    preview.save(HERE/'overview.png')
    doc.close()
    print('Built four-page vector PDF atlas and overview.png')


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--tectonic',default='tectonic')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--assemble-only',action='store_true')
    args=p.parse_args()
    if not args.assemble_only:
        prepare()
        if args.prepare_only:return
        for f in ['fig03_table.tex','fig04_code.tex']:
            compile_tex(args.tectonic,f)
        qa=[export_pdf(HERE/f) for f in ['fig03_table.pdf','fig04_code.pdf']]
        (HERE/'table-code-qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    if (HERE/'fig01_pipeline.pdf').exists() and (HERE/'fig02_confirmation.pdf').exists():
        atlas()


if __name__=='__main__':main()
