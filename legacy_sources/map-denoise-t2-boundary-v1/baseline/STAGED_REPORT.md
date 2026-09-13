# 廉价同信息对照：先估法向，再做冻结 scalar 滤波

文件：`normal_then_profile.py`。冻结 SHA256：
`077a03589c575d7fda921470aa5c61eb1c441b9d9586f27bd6abe44207215733`。
既有 `plane_profile.py` 与 `scalar_reference.py` 哈希核对未改变。

方法：在目标 ROI 内分别去掉每帧 XYZ 均值，普通最小二乘回归 common z slope；
不使用层标签。构建以估计法向为第三轴的正交旋转，把输入变到新坐标，调用冻结
`scalar_profile_hard` 或旧 `joint_forced`，最后转回三维原坐标。

导出两臂：`within_frame_normal_then_profile`、`within_frame_normal_then_old`。
接口同约定。旋转无长度尺度变化；提供的 scalar sigma 保留原值。
这相当于把它当作近似法向噪声/各向同性标准差，并非从未知完整协方差得到了精确噪声变换。
输入并未提供完整噪声协方差，不使用 GT 补出它。

## 小测结果

沿用 plane 小测同一 seed=71331、gap=6 mm、sigma=1 mm、8帧×160点、imbalance=.9。
3族×3角度×2方法，共18个保存输出。以下法向 MAE，mm：

| 场景 | staged-profile | staged-old-joint | 联合 plane（原保存结果） |
|---|---:|---:|---:|
| dual 0° | 0.15054 | 0.14665 | 0.06905 |
| dual 3° | 0.13825 | 0.13408 | 0.06843 |
| dual 6° | 0.12637 | 0.12181 | 0.06785 |
| raycast 0° | 0.85563 | 0.80912 | 0.08620 |
| raycast 3° | 0.85591 | 0.81000 | 0.08666 |
| raycast 6° | 0.85794 | 0.81089 | 0.08717 |
| single 0° | 0.03727 | 0.03727 | 0.03727 |
| single 3° | 0.03550 | 0.03550 | 0.03550 |
| single 6° | 0.03375 | 0.03375 | 0.03375 |

因此，单面法向误差的大收益并不需要复杂联合模型，一次廉价回归即可。
当层标签与横向位置相关时，单平面回归会把层间差异吃进倾斜，raycast 尤其明显；
联合方法在这组具体对照中有进一步收益。**这不证明联合求解在数学上必需，
也不代表更强的分层后顺序法向估计无法取得同样结果。**

staged-profile约5–6 ms/1280点；共同估计约40 ms。可据此比较实际收益与代价。
18例输入未被修改，输出有限、R R^T 与单位阵误差均小于1e-12，先保存后评价。

结果：`/srv/slam-research/grf/map-denoise/runs/t2-boundary-v1-20260911-1704/baseline-staged/results.json`。
本轮不再调整冻结算子，交根线程统一条件和新种子比较。
