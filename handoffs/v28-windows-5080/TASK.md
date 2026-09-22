# 实施任务：V28 Windows / WSL2 GPU 工具化

## 完成目标

在用户指定的 RTX 5080 主机上，把固定 CPU 算法包装为独立可安装 Python SDK / CLI；实现 CUDA 光度评分后端，保留 CPU 法向/插值、候选选择与冻结收益预测器。验证数值、决定和输出一致，再测速度与资源。

这是一项**工程移植**，不要求突破 native 几何质量。优先 WSL2 Ubuntu；Windows 原生仅为兼容性支线，不并行建设两套后端。两者接口相同。

## M0：身份、参考和可安装性

1. `python verify_bundle.py` 验证包。检查 reference 清单及固定 commit，不读取无关历史项目。
2. 发现实际 WSL 发行版和 GPU；记录硬件、系统、路径、Python 与驱动。不要从机器命名猜配置。
3. 建项目内虚拟环境。优先 Python 3.12，核验固定 CPU 依赖能否安装；安装支持实际 Blackwell GPU 和驱动的 PyTorch 构建，保留安装来源/版本/包锁。
4. 不安装驱动；无需自行编译内核的 PyTorch 原型不应先安装整套系统 CUDA toolkit。是否需要编译器由后续实测决定。
5. 运行冻结 runtime 测试。2 个需要旧 V28_SOURCE 的回放若无材料可跳过，但单列 NOT_RUN；不能报告“7/7 全通过”。

交付 ENVIRONMENT.json、REFERENCE_CHECK.json、DEPENDENCY_LOCK、首次测试日志。环境不具备 GPU 也可完成 CPU 工程。

## M1：CPU 应用包装与测试数据

- 在新命名空间 `map_recovery` 实现应用 API 与 CLI，reference 包只读。输入规范见 ARCHITECTURE。
- 保留全部行和单位，构造返回候选、features、state；selector 返回输出与 masks。算法处理本地文件必须经过 adapter，核心不能自己猜数据集路径。
- 建确定性合成场景与**非平凡**几何：有纹理平面、边界遮挡、弱纹理、多峰重复纹理、近切向面、正负 ray 偏移。至少两个场景应产生非零 raw 修正，至少一个应产生 non-KEEP 选择；未触发则声明此层测试未覆盖，补夹具而非改模型。
- 输出 CPU golden（包括完整 scores、validity、anchor IDs、候选/特征/模型分数/masks/点坐标），由冻结参考生成、哈希封存。
- 实现小型离线 demo：用户无数据时能跑；不把 demo 精度算成新科研结果。

## M2：CUDA 评分后端（正确性优先）

- 先实现与 `score_surfacelets` 同接口的 `TorchCudaScorer`。相机规范化与法向保留 CPU，GPU 做 ray/plane 投影、双线性采样、共享有效像素 ZNCC；下游 `solve_surfacelets` 与 selector 仍用 CPU 原实现。
- 从 FP64 几何和归约开始，按 CPU 规则在 scores 出口转 FP32；禁用 TF32/AMP，固定候选顺序。先对齐，再优化；FP64 可能慢，需实测。
- 使用原生张量算子和明确边界掩码；不要默认用 grid_sample 的 padding/坐标/NaN 语义代替 SciPy。
- batch_size 改变只切 anchor 计算，不改变全点邻域、anchor 集或插值权重。OOM 缩小 batch 重新计算同一块，不能跳点、减假设或减视图。
- 按 ACCEPTANCE 做 score、选点、特征、最终输出的逐层差分。结果不能只打印最终均值。

## M3：优化与 RTX 5080 性能

- profile 后优化最大时间占比；缓存五视图、相机矩阵与 patch 网格，复用每 normal 的投影采样；数据传输按批聚合。
- 允许试不同 batch、算子融合、FP32 试验；FP32 是显式候选后端，先按同一验收通过，不能静默替换严格实现。
- 不默认写手工 CUDA/MLIR/Triton；只有 profiling 确认高价值瓶颈、依赖可维护且 M2 通过时才投入。
- 对同一数据/候选数/模型测 CPU 单线程参考、合理 CPU 多线程包装（若实现）、CUDA：首次冷启动、预热、评分、端到端、transfer、显存/主机 RAM。
- 有 GPU 实测就报告，无加速也如实交付。正确但慢是“CUDA_IMPLEMENTED / NO_SPEEDUP”，不是假称达成加速，也不自动否定算法。

## M4：任务生命周期与可用性

- `run` 前台运行，其他终端能按 run ID 查询状态、请求取消；取消在分块边界生效。保存取消状态，不发布半成品点云为成功结果。
- 已完成的 ROI 可复用检查点；未完成 ROI 重算。resume 校验输入、代码、配置、模型 hash；不在第一版承诺 GPU 内存或任意指令级恢复。
- errors 用稳定 code + message，日志分级；钩子只围绕固定事件，不修改核心数值。CI 执行门禁，不依赖可关闭的本地 hook。
- Windows PowerShell launcher 只把参数交给确认过的 WSL 发行版；命令行含空格/中文路径需测试，不拼接 eval，不误用另一发行版。主工程/热数据优先放 WSL 文件系统。

## M5：有限真实回放、交付与停止

- 有用户提供的合法真实输入时，按同一 inputs 在 CPU/GPU 回放一个小 ROI；新主机未提供真实数据则 synthetic 通过，REAL_REPLAY=NEEDS_INPUT，不造数据、不隐式下载 3.9 GB。
- 原确认场景已曝光，回放用于工程对照，不是新的独立确认。由独立 evaluator 做可选原指标核验，不能调模型。
- 打包源码、安装锁、运行示例、CPU/GPU 差分报告、BENCHMARK.csv、结果示例和未完成清单。模型许可/数据条款未核对前，不擅自再授权或公开原始数据。
- 本机建小步提交可以；GitHub push 需用户另行指令。不修改历史提交或原发布包。

## 不作为本轮完成前置条件

完整 GUI/TUI/MCP、插件商城、通用 Agent runtime、COLMAP 重建安装、模型重训、让 native 指标变好、所有历史版本复跑。
若 M0–M5 已完成，交付后停下；不要为了长程而自动换课题。
