# Figure 1: V22 point-only reconstruction pipeline

## Outputs and scope

- `fig01_pipeline.tex`: editable TikZ source.
- `fig01_pipeline.pdf`: 170.000 × 177.998 mm, one page, vector shapes and searchable text.
- `fig01_pipeline.svg`: vector preview/export; text is converted to paths to preserve appearance.
- `fig01_pipeline.png`: raster preview rendered from the PDF at 2.2× (158.4 dpi).
- `fig01_pipeline.log`: retained Tectonic compilation log.

The input JSON is explicitly labelled **configuration illustration**. Its three field names (`requested_ks`, `n_points`, `leave_self_out`) are genuine diagnostics fields; the illustrated values correspond to the frozen default scales and the 1024-point experiment patches. This panel is not represented as a saved per-point run diagnostic.

## Source trace

All lines refer to this repository snapshot, not to V18 or V20:

- `reconstruction_v22/operator.py:95–118`: point-only interface, input validation, neighborhood size caps and query-row exclusion.
- `reconstruction_v22/operator.py:42–78`: independent PCA and quadratic fits at each scale.
- `reconstruction_v22/operator.py:119–138`: common-normal projection, inverse-variance weights, continuous heuristic shrinkage, final normal-only update.
- `reconstruction_v22/operator.py:145–160`: input-only diagnostics.
- `reconstruction_v22/PROTOCOL.md:18,23`: same source points and 1024-point patches.
- `reconstruction_v22/run.py:114–122`: phase outputs are sealed before independent reference scoring.

There is no rejection/safety gate or curvature-triggered passthrough in this diagram. Independent reference geometry connects only to evaluation. The algorithm is a classical MLS-family construction with heuristic uncertainty shrinkage, not a calibrated posterior or a proven safety guarantee.

## Validation performed

- Compiled with Tectonic 0.17.0; compilation exited 0.
- No overfull/underfull boxes or missing-character warnings were found in the retained log.
- Windows Times New Roman and Consolas were embedded; main serif font sizes are approximately 8, 9 and 10 TeX points.
- PyMuPDF verified one page, 170 mm width, no embedded raster images, 41 vector drawing groups, and no text spans outside the page.
- Rendered and visually checked the PNG. The configuration-illustration caption was moved below its box to avoid touching the border, then rebuilt and checked again.
- Compilation prints a non-fatal Fontconfig configuration message and Windows-font path portability warnings. The resulting PDF contains the intended Times New Roman and Consolas fonts.

## Suggested Chinese caption

图1｜V22逐点局部曲面重建与多尺度修正。每个查询点在排除自身的32、64和128近邻中分别建立PCA坐标并拟合二次曲面，将各尺度修正投影至64近邻法向后加权融合，再按残差方差、预测方差代理和尺度分歧施加连续阻尼。输出保留点数与源行序，仅沿局部法向移动。灰色支路记录输入侧诊断；独立DTU参考仅在输出封存后的评估阶段使用。该阻尼为启发式，不构成校准可信度或安全保证。输入JSON为配置示意。

## Rebuild

From this folder:

```powershell
& 'C:\Users\gengr\Documents\Paper-Figure-Toolkit\runtime\tectonic-0.17.0\tectonic.exe' --keep-logs fig01_pipeline.tex
```

Render with the system Python and PyMuPDF:

```python
from pathlib import Path
import fitz
doc = fitz.open('fig01_pipeline.pdf')
page = doc[0]
page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False).save('fig01_pipeline.png')
Path('fig01_pipeline.svg').write_text(page.get_svg_image(text_as_path=True), encoding='utf-8')
```
