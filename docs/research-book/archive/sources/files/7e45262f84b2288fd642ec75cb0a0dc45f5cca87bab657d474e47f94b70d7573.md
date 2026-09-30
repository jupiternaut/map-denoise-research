# 发给 Windows Codex 的提示词

复制下面这段，在已经解压的本交接目录启动一个新 Codex 任务：

---

请读取本目录 AGENTS.md、README.md、TASK.md、ARCHITECTURE.md、PORTING_NOTES.md 和 ACCEPTANCE.md，实施 V28 的 Windows/WSL2 RTX 5080 工程移植。

请直接实施，而不是再交一份计划。先核对当前 Windows 主机、实际工作目录、WSL2 发行版及 RTX 5080，再验证 BUNDLE_MANIFEST。优先使用本机已存在的 WSL2 Ubuntu 与项目内venv；不要静默转去其他电脑，也不要全局安装驱动、WSL或重启系统。若需要系统级改动，请给最小必要操作并等我确认。

reference/ 是固定CPU算法和模型，只读。主线是：先做可运行SDK/CLI和CPU goldens，然后把光度评分迁到GPU，下游候选选择、插值和冻结selector保持CPU语义。大胆优化分块、缓存和张量并行，但不改几何算法/特征/模型来获得表面上的速度或精度优势。

按M0–M5连续推进，阶段通过就继续，不必每一步询问。显卡忙时先做CPU代码和测试；按约定等待，不能杀其他进程。无可用GPU或缺真实数据时，交付已经完成的代码、日志、明确状态与下一条可执行命令，而不是空等或假称通过。

保证测试有非零修正的案例，不仅测试全KEEP。给出CPU/GPU分数、有效掩码、候选选择、模型特征、最终路由和坐标逐层对照；通过后才报告性能。没有实测不要预报加速倍数。GPU结果不得使用GT做选择。

本轮先完成SDK/CLI、可取消任务和结果存储；GUI/TUI/MCP只留清晰适配边界，不要因此开一个通用Agent平台。不要重训模型、重做几何科研、下载未批准的大数据包或推送GitHub。完成请给最短运行命令、结果目录、实际GPU性能和未完成事项。

---

## Windows 侧只读核验示例

以下是现有系统命令，不是项目CLI。路径和发行版须由实际输出确认：

```powershell
hostname
Get-Location
wsl.exe --list --verbose
nvidia-smi
```

在已确认的WSL终端：

```bash
hostname
pwd
uname -a
python3 --version
nvidia-smi
python3 verify_bundle.py
```

若WSL中nvidia-smi不在PATH，按NVIDIA文档检查 `/usr/lib/wsl/lib/nvidia-smi`，不是安装Linux驱动。缺少python3/WSL时先报告，不假定用户已同意系统安装。

本包无需重新clone即可阅读reference。若要复核来源，可只读获取GitHub固定提交并核对SHA，不切到main最新状态。`PACKAGING_VALIDATION.json`只证明liekkas上的打包核验，不是Windows/5080执行结果。
