# V28 Windows / WSL2 RTX 5080 实施交接包

状态：**架构与任务已编写；CPU 参考包随附；CUDA 后端尚未实现，也未在 Windows 验证。**

目标是把已有方法交付为可复现工具，并加速重复的光度评分，不是重做研究算法。

## 怎么交给 Windows Codex

1. 把本文件夹或同名 ZIP 传到 Windows 并完整解压，在该文件夹打开 Codex。
2. 把 `START_WINDOWS.md` 的启动提示词发给它。
3. Codex 先核对 Windows 主机、WSL 发行版和 RTX 5080；缺 WSL/驱动时先报告，不修改系统。
4. 推荐在**经核对的同一台 Windows 主机的 WSL2 Ubuntu**建立独立工程。具体路径由 Windows 实际环境决定，不能套用原机 `/home/grf`。

ZIP 无数据集、无私钥、无 Python 环境；CPU 参考包及冻结模型包含在 `reference/package/`。项目源自：

- 仓库：https://github.com/jupiternaut/map-denoise-research
- 固定提交：`2cf73a80f5e41a5d126ebdc8321c0cd6c61f24f9`
- 固定子目录：`research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z`
- [GitHub 原始实验报告](https://github.com/jupiternaut/map-denoise-research/blob/2cf73a80f5e41a5d126ebdc8321c0cd6c61f24f9/research_snapshots/2026-09-22/closeout-confirmation-20260922T133739Z/EXPERIMENT_REPORT.md)

## 阅读顺序

| 文件 | 作用 |
|---|---|
| `AGENTS.md` | 范围、权限、停止条件 |
| `TASK.md` | 分阶段实施与交付定义 |
| `ARCHITECTURE.md` | 模块、API、执行状态与扩展边界 |
| `PORTING_NOTES.md` | 源码依据、数值语义与 GPU 迁移细节 |
| `ACCEPTANCE.md` | 可执行验收契约；哪些命令尚待实现 |
| `START_WINDOWS.md` | 用户可直接复制的启动提示词和环境核验 |
| `reference/` | 从固定 Git 对象导出的 CPU 代码、模型及历史证据 |
| `verify_bundle.py` / `BUNDLE_MANIFEST.json` | 无需第三方依赖的文件哈希核验 |

## 第一版边界

必须完成：CPU reference + CUDA score backend + SDK + CLI + 状态/取消/产物 + 对照测试 + RTX 5080 实测报告。

可以自由探索：评分分块、缓存、向量化、内存复用；先以 PyTorch 实现，再按实测决定是否写 CUDA/Triton 内核。

后续接口：GUI / TUI / MCP 使用同一应用 API，不能各自实现算法；第一版不要求全部开发。插件只预留计算后端协议，不建插件市场或通用 Agent 平台。

不能混淆：本方法需要几何、照片和相机标定，不是仅输入任意 PLY 的无条件滤波器。CPU/GPU 等价与提速不等于几何精度提高。现有证据是在同来源新场景受控 ±3 mm 偏移上改善；原始输入 MSE 恶化 8.12%，不得在产品说明中删掉。

此包在原机只进行了来源、完整性与参考代码检查；Windows 环境、GPU 兼容性、性能均交由目标主机实测。不替用户推送新分支或安装显卡驱动。
