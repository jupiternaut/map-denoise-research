# 2026-09-16 公开快照

目标：`jupiternaut/map-denoise-research`，原 `main` 分支；保留远端科研图册更新。
这是**代码与研究证据快照，不是磁盘完整备份，也不是新算法部署**。

## 收录与排除

- 收录 V23/V24/真实支持域源码与指标、V25 源码与结果、CPR 构造、E0–E0-F、Hermes 与 open-world、有限精确搜索实验。
- 保留负结果、协议、更正、主要 CSV/JSON、测试代码、绘图与核验记录。
- 排除原始点云/照片/模型包、NPY/NPZ/PLY、环境缓存、重复 replay、逐输出照片覆盖图、TLC JAR 和第三方论文全文。
- 文件级路径、SHA-256 和排除原因见 `SNAPSHOT_20260916.json`。其中 source 是原机来源，不是可在 GitHub 点击下载的地址。
- 历史源内容逐字节保存；AGENT/AGENTS 文件仅改名为 SOURCE_AGENT/SOURCE_AGENTS，避免旧任务指令在公开仓库生效。
- 原 `SOURCE_LOCK` / `SHA256SUMS` 是**原运行证据**，可能包含未发布文件；公开子集应使用本目录 manifest 验证，不能假装旧完整清单均可满足。
- 本次未添加或更改软件许可；第三方数据的许可与下载要求继续适用。

## 复现层级

**有限精确搜索实验：** 标准库实现；可在其公开目录运行只读测试：

```bash
cd research_snapshots/2026-09-16/pnp-oracle-lab-20260916T112018Z
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -v
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest audit_checks -v
```

重跑数值实验使用该目录 `COMMANDS.md` 的新输出目录命令。历史 `verify.py` 额外检查原机 host、
旧 Hermes 绝对路径和哈希，不是跨机器的一键入口。`audit_checks.py` 作为脚本会更新收据；上述 unittest 形式不改收据。

**几何/E0/Hermes：** 多处仍依赖原机绝对路径、兄弟工作区或未发布数组，按各自 COMMANDS/README 配置。
不宣称仅克隆本仓库即可重现全部历史。发布时的本机测试通过也不等于冷启动可移植性验证。

检查公开导出内容（不需要原机数据）：

```bash
python3 -B publication/check_snapshot.py
```

`export_20260916.py` 是原机打包工具，仅在 liekkas、源目录齐备时使用；它拒绝覆盖不同内容。
发布验证记录见 `VALIDATION_20260916.json`（只覆盖本次执行的检查，不替代历史实验验证）。
