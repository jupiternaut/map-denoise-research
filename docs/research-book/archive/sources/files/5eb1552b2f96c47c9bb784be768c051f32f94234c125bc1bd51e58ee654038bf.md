# 设计架构：共享能力 API + CPU 参考 + CUDA 评分

## 1. 当前事实与新增设计

**已有**：`v28_closeout.construct` / `apply_arrays`、scikit-learn 冻结模型、四方向法向×五支持域、CPU ZNCC、点级候选与 KEEP 决策、研究用数据适配器。

**待实现**：独立 `map_recovery` 包、通用输入清单、CLI/任务状态、CUDA scorer、安装/验收工具。下面的接口、目录和状态机都是设计，不是已经存在的功能。

**后续可选**：GUI、TUI、MCP。GUI 不是算法主入口；SDK 不调用 CLI 再解析文本；所有入口调用同一 ApplicationService。

```text
本轮：CLI ─┐                 后续：GUI / TUI / MCP
           ├── ApplicationService（共享公开契约） ◀─────┘
Python SDK ┘    │ validate / run / status / cancel / resume
               │
        RunStore + 生命周期事件 + 配置/资源检查
               │
        InputAdapter（几何 + 5照片/相机；不读GT）
               │
        CPU 准备：单位检查、法向、anchor、插值权重
               │
        EvidenceScorer ── CPUReferenceScorer
               └──────── TorchCudaScorer（本轮GPU边界）
               │               scores → CPU
        原 solve_surfacelets → 原特征展开/插值 → 原模型
               │
        原 KEEP / A / B 决策 → 坐标/状态/审计产物
               │
        单独 evaluate（可读取独立参考，不逆流进入推理）
```

这只是应用分层，不是 OS 内核态/用户态隔离。Python backend 是受信扩展，默认具有进程权限；第一版不能宣称沙箱。

## 2. 目录建议

```text
project/
  AGENTS.md
  pyproject.toml
  src/map_recovery/
    api.py                  # 唯一应用能力入口
    contracts.py            # SceneSpec / RunConfig / RunResult / Error
    cli.py
    adapters/arrays.py      # 已规范化的数组输入
    adapters/manifest.py    # 相对路径、单位、视图、图像
    pipeline.py            # prepare -> score -> finalize -> select
    backends/base.py        # EvidenceScorer Protocol
    backends/cpu.py
    backends/torch_cuda.py
    jobs/store.py           # 一个run一个目录，状态/事件/检查点
    jobs/events.py
    jobs/resources.py      # 只检查/等待，不杀进程
  reference/package/       # 冻结v28_closeout只读
  tests/{unit,contract,parity,integration}/
  fixtures/{synthetic,golden}/
  examples/
  scripts/{doctor.ps1,run-wsl.ps1}
  runs/                     # 忽略，不进入源码版本库
  reports/
```

拆分 `prepare/finalize` 可以在新模块复用原函数和顺序，但先以原 `construct` 验证 CPU wrapper；禁止全局 monkeypatch 函数来切 backend（并发时会串状态）。`graph_field.solve_field` 不是当前主路径，不纳入第一轮 GPU 迁移。

## 3. 核心契约

### SceneSpec

- `schema_version = 1`、唯一 scene_id。
- `points`：有限 `[N,3]`、N>=3、物理毫米、行号稳定。支持 `.npy` 或 NPZ（`allow_pickle=False`）；PLY 是适配器可选输入，必须记录如何保留行和其他属性。
- `reference` 加 `sources[4]`：每相机有 2D grayscale 图像、物理 `P[3,4]`、`center[3]`，source 顺序固定。
- 声明 `image_encoding`：`prepared_gray_v28` 表示已经半分辨率/投影变换；`rgb_u8_original` 则按冻结规则只预处理一次。禁止根据尺寸猜是否要再减半。
- 初版要求无畸变针孔投影。未提供可靠去畸变/尺度变换的输入返回 UNSUPPORTED_CAMERA 或 UNIT_MISMATCH，不能猜米/mm。单位换算必须同步修改 points、center 和 P 的世界坐标表示。
- 可选 source_vertex_ids / ROI；不接受 GT 字段供推理使用。完整网格输入时本轮只更新顶点，不声称拓扑优化；若输出网格要保留 faces 并验证索引，未知属性不应静默丢失。
- 文件路径以 manifest 所在目录为基准；不使用原机绝对路径。输入只读，输出必须新目录，禁止覆盖源文件。

### RunConfig

- `backend`: cpu / cuda / auto；`precision`: reference64 / experimental32。
- `method`: identity / post_A_keep（主）/ post_AB_keep（次）以及命名明确的诊断臂。
- 方法是用户明确选择，GPU版不自动把主臂换成“效果最好”的次臂。未经选择的产品演示默认 identity，另存修正候选；benchmark 显式选择 post_A_keep。
- `batch_size`: 起始32，只控制评分anchor分块；`gpu_budget_gib`: 默认4或用户明确值；`wait_gpu_seconds`: 默认1200。
- seed 固定20260922；冻结算法配置单独有 hash，不用执行参数覆盖。
- `cuda` 请求不可用则明确错误；`auto` 可回退CPU，但 `backend_requested/backend_effective/fallback_reason` 必须出现在结果和性能表。

### API（建议签名，尚待实现）

```python
validate_scene(scene: SceneSpec) -> ValidationReport
run(scene: SceneSpec, config: RunConfig, output_dir: Path) -> RunResult
get_status(run_id: str, store: Path) -> RunStatus
cancel(run_id: str, store: Path) -> CancelReceipt
resume(run_id: str, store: Path) -> RunResult
```

`run` 同步但先创建 run_id 并写结构化启动事件；CLI progress 输出 stderr，stdout 只输出 JSON 事件或最后结果，不让SDK解析控制台。未来 submit/daemon 可包装此执行器，本轮不要求常驻服务。

### EvidenceScorer

```python
score_surfacelets(points, normal_bank, offsets, reference, sources,
                  batch_size=32) -> dict
```

返回与原函数同名同形数组：scores float32 `[4,N_anchor,20,49]`，缺测为 NaN；directions、candidates、offsets、ref_variance、ref_fraction、ref_valid 及原假设元数据。FP64中间值按原语义处理。下游统一在 CPU 调原求解与模型，避免同时改变两个模块。

## 4. 生命周期、状态与副作用

```text
CREATED -> VALIDATING -> [WAITING_GPU] -> RUNNING -> SUCCEEDED
                    \-> FAILED                 \-> FAILED
RUNNING / WAITING_GPU -> CANCEL_REQUESTED -> CANCELLED
FAILED / CANCELLED / WAITING_GPU --显式resume且hash一致--> VALIDATING
```

- 状态来自 RunStore，不属于窗口。RunStore 保存 config/input/code/model/env hashes、状态时间戳、最后完整ROI及未完成原因。
- 单writer锁；events.jsonl 单writer顺序追加、带序号。state.json 临时文件完成后原子替换；崩溃后的不完整末行检测，不伪装成功。
- 取消是合作式停止，不是撤销此前所有副作用；输出先写 `.partial`，成功后原子发布。不得删除原输入或把中断文件当最终结果。
- 重启通过新 attempt ID 复用已封存ROI，history不改写；失败CUDA块未完成即整体重算。没有hash一致不自动复用。
- `hooks`: before_run、after_stage、on_artifact、on_failure；计时/测试/审计订阅事件。可观察hook不得改数组/接受规则；hook超时/错误单列，不无限递归。关键输入和发布检查在宿主内不可跳过。
- 未来 MCP: tools 对应 validate/run/status/cancel/results，资源对应只读报告；不暴露任意shell；网络鉴权/多用户隔离另行设计。

## 5. 部署与依赖

推荐 Windows -> WSL2 Ubuntu 用户目录 -> 项目venv -> PyTorch CUDA -> Windows NVIDIA驱动 -> RTX5080。
CPU依赖以 reference requirements/模型锁为准；Pillow作为原图adapter依赖，Open3D仅旧DTU mesh loader可选，不应变成评分SDK硬依赖。PyTorch作为 cuda extra，CPU安装不应被迫下载CUDA轮子。

官方资料（实施时核对当前版本，不把下面URL当作执行授权）：

- [NVIDIA WSL 用户指南](https://docs.nvidia.com/cuda/wsl-user-guide/)：WSL使用Windows驱动，不装Linux显示驱动。
- [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)：选择实际平台/驱动支持的构建，锁版本和下载源。
- [NVIDIA GPU 能力表](https://developer.nvidia.com/cuda/gpus)：核对实际GPU能力，不根据 `nvidia-smi CUDA Version` 推断已安装toolkit或所有包都兼容。

### 真实数据与适配器

本包不包含照片/稠密点云。随附的旧 `scene_adapter.py` 用于理解尺度和预处理，不可原样拿49张DTU图的限定当通用接口。新工程先支持显式5视图manifest；大包下载或Liekkas数据转移由用户决定。没有真实数据并不阻塞 synthetic/CPU/GPU 工程测试，但真实回放必须标未执行。
