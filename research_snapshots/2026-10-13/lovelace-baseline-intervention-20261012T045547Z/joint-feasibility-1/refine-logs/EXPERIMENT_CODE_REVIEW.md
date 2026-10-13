审查结论：PASS，same-family / provisional。无未解决 BLOCKING；此前发现的 budget 叶计数口径差异已修复。

已核实 host=`liekkas`，精确根为 `/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1`。本审查只读协议、根源码及必要历史源码；未读取 private/truth、正式结果，未运行正式批次，未编辑文件。

逐项结论：

- **NONBLOCKING／通过：原叶与约束恢复。** [run_polytope.py:62](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/run_polytope.py:62) 使用原 lo/hi、校准界、九材料界及 joint 三偏差界，固定 δ=1/10；只遍历原 retained_ids，不新增分裂或传播。与历史 `variation_solver.py` 的 retained_tolerance／unresolved_budget 记录语义一致。
- **NONBLOCKING／通过：证书数学与失败处理。** [polytope_filter.py:133](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/polytope_filter.py:133) 独立重算非负有理权重、λA、λb、带符号 box-min，必须严格 m>B；不要求 λA=0。LP失败、非有限结果、无精确证据返回 UNKNOWN。精确见证检查全部行和全部变量界。
- **NONBLOCKING／通过：仅证书删叶与正确 union。** [run_polytope.py:78](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/run_polytope.py:78) 保留 UNKNOWN；历史 union 确实只按 filtered retained_ids 取叶。trace、剩余叶边界不改，空 hull 沿用 INCOMPATIBLE 且输出原初态。
- **NONBLOCKING／通过：baseline及信息顺序。** [run_polytope.py:185](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/run_polytope.py:185) 核对全部旧动作字段；193–198行完成648新＋648旧＋324KEEP保存、读回及唯一计数后才调用 private 评价。前置 private 字节只经冻结哈希路径读取，不解析或进入选择。
- **NONBLOCKING／通过：独立核验。** [verify_polytope.py:155](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/verify_polytope.py:155) 使用历史独立几何和必要行实现，不导入新 filter／runner；193行起逐叶核对身份、证书／见证、原trace和union。旧评估复用函数与 box/polytope 标签兼容；动作、CSV、JSON及指标均独立重算。
- **NONBLOCKING／通过：同一原叶覆盖。** [verify_polytope.py:418](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/verify_polytope.py:418) 检查生成分支内同叶真深度、九材料及 joint 三偏差，重验真实参数满足必要行，并要求原全部此类见证叶存活，强于总体 hull 覆盖。
- **NONBLOCKING／通过：预算、分母、统计。** [execute_polytope.py:18](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/execute_polytope.py:18) main与verify共享一个3600秒绝对期限，子进程仅获剩余时间；两端 BUDGET_SCOPE 相同。核验固定72／5103／1620及主40等分母，统计口径吻合；不以科学增益为机械通过条件。
- **NONBLOCKING／已修复：budget元数据。** [verify_polytope.py:178](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/verify_polytope.py:178) 现按完整冻结trace计 unresolved_budget_leaves，与历史union一致；[test_verify_polytope.py:163](/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/joint-feasibility-1/test_verify_polytope.py:163) 覆盖删除budget叶及错误计数拒绝。本矩阵仍独立要求budget叶为0。

最终 SHA-256（以下文件均位于上述精确根）：

```text
execute_polytope.py      a3969f591579156570fba8491dd31ee81f69d3599f147df2d321975b16ca901d
polytope_filter.py       094e982cad973c5dc633758a874272e833da4b47a8d7b7e938e20efc566f031f
run_polytope.py          0572bd17e3e8314c5b4adc405e1f53f961d0cd5c051fd51521dc91c7bb83e993
test_polytope_filter.py  f21a7362f57071f2ca3813ad030894df7c642e71330f9c95959574efd7587d52
test_polytope_pipeline.py 23cbbc6e134b4a495cbbb26d922043b8ecf645e9defe712bc640dd5df1fcb1f7
test_verify_polytope.py  831319c9a35c158a4eb9f4cc7d471daf290d78b48b0b69e9b2bdf1acc7a89cfc
verify_polytope.py       96ec770db9788122e77a78223878f3f84c051c4615b87537ae3f2f0055c95a83
AGENTS.md               e42b0b0d7cec720d3c8f5752caf0b49461c1a7e60a93ba0b1351f475caec91c9
PROTOCOL.md             3e3daf03adc456aba21af30b4a37587fc4c068b7e41080fba2b214d171a1d8a8
REGISTRATION.json       5e16b04735a96b89aa5e278a2b00b5b6035dd1893484d8527507b069f3b248f6
```

本结论是冻结实现的源码语义审查，不是正式运行结果验证，也不是真实物理 soundness 的外部证明。FEASIBLE_WITNESS仅证明冻结盒级必要线性松弛非空。
