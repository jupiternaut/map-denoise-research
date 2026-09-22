# 验收：把“能安装、同算法、用GPU、真正加速”分开

本文件的 `map-recovery` CLI 和 `tests/parity` 等是**待实现验收契约**，不是声称当前已存在。reference测试与verify_bundle.py现在可执行。除明确标为历史记录外，所有PASS必须来自目标主机实际日志。

## G0 完整性和环境

```text
python verify_bundle.py
python reference/tests/test_runtime.py
```

第二条在项目venv安装固定reference依赖后执行；5个runtime测试应通过。2个旧归档回放无V28_SOURCE时明确SKIP，不阻塞工程初验，也不算通过。

ENVIRONMENT.json必须含：Windows主机、WSL发行版、发行版版本、Linux hostname、路径映射、GPU名称/UUID/compute capability、driver、python/pip依赖、torch及其CUDA构建、实际CUDA张量运算结果。单有torch.cuda.is_available或nvidia-smi不能代替实际kernel smoke。

## G1 CPU 包装一致

同一目标环境中，新CPU pipeline与未改的reference直接运行对照；禁止以新pipeline生成自身golden。Linux历史结果可作为额外对照，不要求跨BLAS/系统误称逐位完全相同。

- shape、source顺序、anchor IDs、假设、choice、validity、masks严格相同。
- 相同实现的CPU包装，所有输出须一致；若拆分引入差异，先定位，不让GPU来背锅。
- 零纹理/all-invalid最终输出为identity；至少两种纹理夹具有非零raw更新，至少一组有selected更新。反例/失败fixture保留。
- 输入变更/模型hash变更不能复用旧检查点；不读GT也不打开reference目录之外的数据集真值。

## G2 数值与离散决策

以下是**预设工程等价门槛，不是已测精度或统计保证**。比较在共同finite位置进行，NaN/inf布局另查。

| 项目 | 严格版初始验收 |
|---|---|
| input行、anchor、source/hypothesis顺序 | 完全相同 |
| score/ref_valid及全部下游validity | 完全相同 |
| finite score max绝对差 | <=1e-6；同时报告P50/P95/P99/最大值 |
| ref_fraction、计数 | 完全相同 |
| ref_variance | atol1e-10, rtol1e-6 |
| hypothesis/choice、zero fallback | 完全相同 |
| 插值offset、A/B坐标、最终坐标 | max绝对坐标差<=1e-5 mm；逐点欧氏差也报告 |
| features | atol1e-5, rtol1e-5；缺失标志完全相同 |
| 预测收益 | atol1e-6 mm², rtol1e-5，并要求route不变 |
| 主/次/random路由、KEEP位 | 完全相同；KEEP原坐标不变 |

两个score在1e-6内却选择不同，仍判G2 FAIL；不能用平均误差掩盖离散翻转。必须给最小case、阈值/同分差距和中间tensor定位。CPU直接调用、CPU拆分、CUDA分别列，区分包装错误和GPU误差。

float32优化版可以作为实验后端；若未通过，不默认启用，保留严格版。不能看GT后放宽验收。必要容差修订单独提出，保留原验收结果。

测试矩阵至少含：

1. 平坦纹理、阶梯/周期纹理、线性渐变图；已知双线性采样值。
2. 边界坐标0/W-1及±5e-10/±2e-9；NaN像素/坐标；无效投影深度。
3. 共享像素计数阈值两侧、方差阈值两侧；所有候选无效。
4. 成本精确同分、近同分、±offset同模、全零；CPU候选选择单测和端到端GPU都检查。
5. model gain=负/零/正，AB完全同分；无修改全部KEEP。
6. 两种source排序（分别与自己的CPU一致），修改reserved source不能改变fit提案。
7. batch=1/7/32及尾块非整除；重复运行3次，输入/输出不受上次状态污染。
8. OOM缩batch/取消/恢复；无GPU请求cuda明确报错；auto回退显式记录。

## G3 命令与生命周期（实现后必须执行）

建议CLI：

```text
map-recovery doctor --json
map-recovery demo --write fixtures/demo
map-recovery validate --scene fixtures/demo/scene.json --json
map-recovery run --scene fixtures/demo/scene.json --backend cpu --method post_A_keep --out runs/cpu
map-recovery run --scene fixtures/demo/scene.json --backend cuda --precision reference64 --method post_A_keep --out runs/cuda
map-recovery compare --left runs/cpu --right runs/cuda --out reports/parity.json
map-recovery benchmark --scene fixtures/demo/scene.json --backends cpu,cuda --warmup 3 --repeats 10 --out reports/benchmark
map-recovery status --run-id ID --store runs --json
map-recovery cancel --run-id ID --store runs --json
map-recovery resume --run-id ID --store runs
```

`demo`若目录存在拒绝覆盖；run先落run ID与事件，status/cancel可以另开终端。首版SDK同步run足够，不要求daemon。命令名称可以统一调整，但必须同步文档与测试，不留“命令看起来存在”却无法运行的示例。

## G4 性能口径

- 精度等价通过后报告速度；相同输入、模型、offset数、hypothesis数、点数、图像分辨率、CPU线程。
- 评分device计时用CUDA events或同步边界；端到端wall clock包含CPU准备、传输、评分、CPU求解/模型。另报含I/O的用户总耗时，不能混作同一种加速比。
- 冷启动至少一遍，warmup3、正式10遍，报告median/P95/全部记录；不混用预热CPU与冷GPU。
- 记录transfer、torch peak allocated/reserved、系统显存占用及测量来源、CPU峰值RSS的实际进程范围。Windows后台占用不可算本进程峰值。
- CPU single-thread参考与合理CPU多线程若有都列出；仅比单线程时标明，不能声称击败最优CPU。
- G4的完成要求是可复现测量，不设未经依据的10倍速度门槛。GPU正确但不快就报告不快。

## G5 真实回放与交付

真实数据未提供：G0–G4可以PASS，G5_REAL_REPLAY=NEEDS_INPUT，最终状态 PARTIAL_VALIDATED；不以synthetic冒充真实场景。
提供数据：至少一个带纹理真实ROI，同一输入CPU/CUDA完成G2；完整研究560项不是移植首次验收前置条件。

必要文件：README、固定依赖锁、ENVIRONMENT.json、REFERENCE_CHECK.json、PARITY.json（逐层）、BENCHMARK.csv、TEST_RESULTS.json、CHECKPOINT.md、示例输入/输出和可复现命令。测试结果绑定code/input/model/config哈希；代码再改不能沿用旧绿灯。

完成标签分别为 CPU_READY、CUDA_IMPLEMENTED、PARITY_PASS、PERFORMANCE_MEASURED、REAL_REPLAY_PASS；禁止用一个“全部完成”掩盖未测项。
