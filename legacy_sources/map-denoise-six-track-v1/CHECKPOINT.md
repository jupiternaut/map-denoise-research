# 检查点与交付包

## 原检查点：保留且校验通过

- 路径：`/srv/slam-research/grf/map-denoise/checkpoints/20260911-1404-pre-paper/research-state.tar.gz`
- 本轮开始与结束均通过原 `SHA256SUMS` 校验。
- 旧项目、旧论文中纳入清单的 201 个文件前后哈希无变化，记录见 `BASELINE_SOURCES.json`、`FINAL_VALIDATION.json`。
- 本轮没有执行恢复、覆盖或删除。回到旧研究状态时直接使用原目录；无需覆盖本轮目录。

## 新检查点：本轮源码、报告与全部原始结果

新工作目录：`/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1`。
新运行目录：`/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600`。

交付归档目录：`/srv/slam-research/grf/map-denoise/checkpoints/20260911-six-track-v1`。

- [源码、报告与结果包](/srv/slam-research/grf/map-denoise/checkpoints/20260911-six-track-v1/six-track-source-results.tar.gz)
- [逐文件清单](/srv/slam-research/grf/map-denoise/checkpoints/20260911-six-track-v1/FILES.json)
- [归档 SHA256](/srv/slam-research/grf/map-denoise/checkpoints/20260911-six-track-v1/SHA256SUMS)
- [归档内逐文件核验](/srv/slam-research/grf/map-denoise/checkpoints/20260911-six-track-v1/PACKAGE_VERIFICATION.json)

包内 `workspace/` 对应本轮源码与报告，`results/` 对应全部原始输入、输出和指标。
报告保留本机绝对链接；换机器后需要相应修改。外部源码、权重、既有 Python 环境、旧项目依赖不重复打包；这不是独立可运行的容器镜像。

归档只用于保留与搬运本轮结果。重新执行时应解包到新的明确目录，不能覆盖已有工作；冻结清单与原始失败结果应保留。
