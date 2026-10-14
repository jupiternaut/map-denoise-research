# A15：最终边界四盒离线体检

本脚本只读保存的求解状态及 A14 公开观测，不读真值、输入扰动、任务分组或评价结果。它不参与在线算法，不改变主实验的计算预算或决策。诊断成本另行记录。

## 边界对象

从 `state.cells` 选出所有触及最终 `lower` 或 `upper` 的单元，而不是宽度小于某阈值的任意已访问单元。同端点上所有分支均保留；同观测、同分支、同二进制浮点端点的单元跨来源去重，所有来源继续登记。记录原始 `checked`、`terminal`、停止原因；不把未检查或超时单元伪称为已达到容差。

## 每盒的证据

每个最终边界单元的 0、1、2、3 四盒全部执行，绝不因先前盒有解或未知提前终止。每盒运行在独立子进程中，上限 3 秒，父进程超时终止并记为 unknown。共享模块导入在父进程一次完成，但计入父进程成本。

1. 调原始 A14 `_TileContext.check_cell` 寻找并精确核验不可行证书。
2. 未得到矛盾时，在剩余预算中构造必要 LP 的数值内部点／顶点候选，再以有理数逐行核验所有约束和变量界；近似活跃集可提示有理基求解，但不能授权“可行”。
3. 使用原始 A14 独立验证器重建约束行，再复验矛盾或可行见证。独立验证失败降为 unknown。

严格分为 `exact_infeasible`、`exact_necessary_lp_feasible`、`unknown`。数值求解状态单独记录；浮点成功而找不到精确见证仍属 unknown。

**必要 LP 真可行不等于物理成像模型真可行，也不证明继续切相机域一定能排除该单元。** 四盒全部不可行说明当前保留单元可被这个离线资源配置排除；它本身仍不能把在线漏排归因为某一个独立机制。

## 调用

```bash
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B boundary_tests.py
/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B boundary_diagnostic.py \
  --batch /srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z/outputs/formal-v1 \
  --arms joint --tile-seconds 3 \
  --output /srv/slam-research/grf/map-denoise/runs/lovelace-proof-reuse-20261014T074405Z/outputs/boundary-coarse.json
```

`--batch` 可重复给出，支持批目录及直接 `jobs` 目录；同单元只执行一次。新实验的 arm 名若不同，显式传入对应名称。`--max-cells 1` 只作冒烟测试，结果明确登记未覆盖全量，不当正式结论。输出只能新建在本轮目录，不能覆盖已存在文件。主实验运行期间不要并行启动体检，以免污染耗时对照。

本文件是实现说明；完成数值运行前不声称任何边界分类比例。
