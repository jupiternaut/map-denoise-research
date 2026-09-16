# 报告交付与核验

主机 liekkas，日期 2026-09-15。本目录仅制作 E0-D 正式报告，不运行新科学实验。

正文为 CPR2_E0D_TECHNICAL_REPORT.md，Word 版为 CPR2_E0D_TECHNICAL_REPORT.docx。共 12 页、7 张表、4 个原生 Word 数学公式块。第 1 至 12 页已逐页查看最新渲染，无发现文字裁切、表格溢出或公式缺字。qa 内的 PDF 和页面图是排版核验中间产物。

证据检查：旧 E0 87 文件及 E0-D 283 文件，共 370 文件 SHA-256 与制作报告前相同。EVIDENCE_SNAPSHOT.json 保存完整清单。REPORT_RECHECK.json 保存对现有输出的重新汇总；没有生成新观测、重新调参或改写原结果。

关键核验值：BH 17/18 配置零拒绝；投影 24/24 优于 ARGMIN，3/24 优于 identity；连续边界覆盖内累计有害事件 348 到 0；守卫阻止 3600 个有害候选，也阻止 3600 个有益去重影候选。报告环境指标重算最大差 4.440892098500626e-16 mm，低于沿用的 1e-12 mm 容差。

数学统计复审收紧了『样本不足已被排除』的原草稿措辞，最终只陈述 2000 至 32000 已考察范围内未稳定恢复检出。守卫复审补入单次回退后仍可能出现新收缩的限制。

正式文件 SHA-256：

- Markdown：32e7f21b0731a2f447a1103531911f1efa9fe5950c9968d2a828c5cc87d2fe53
- DOCX：d8e851b465c1a48e4177c17d6c3db271dda1bfc7fcb4f5e7880716b2e103fa68

build_report.py 是报告生成与指标重算脚本，不导入或执行实验模块。使用报告用捆绑解释器：

`/home/grf/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`

文档渲染使用捆绑 LibreOffice，由 documents 技能的 render_docx.py 调用；未使用桌面 LibreOffice。重新生成 Word 后必须重新渲染和目视核验，且文件哈希会变化。报告生成没有安装包、使用 GPU、创建后台任务或上传 GitHub。
