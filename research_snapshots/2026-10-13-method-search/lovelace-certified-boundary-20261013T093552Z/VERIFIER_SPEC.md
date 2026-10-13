# A13 独立证书核验规范

目标主机 `liekkas`；工作目录 `/srv/slam-research/grf/map-denoise/runs/lovelace-certified-boundary-20261013T093552Z`，已核对现有符号链接解析至同名实验目录。核验器仅依赖 Python 标准库，不导入 `cert_kernel`、搜索器、优化器、其他方法或真值。输入仅为观测与证书／搜索状态。规范冻结的物理定义来自 A12 `task_bundle/TASK.md` 和公开 `forward.py`；重建使用理想有理数像素边界，不把浮点 renderer 系数当成精确模型。

## 输入语义和限制

`photo` 为 432 个 JSON 数值，按视图、像素、RGB 排列；`camera.u` 为 3 项；`appearance.coupons` 为 **平铺 9 项**、`references` 为 3 项。元数据必须与冻结定义相同：`Xf=0,Zf=1000,survey_bound=localization_bound=.01`，外观 `bias_bound=.01,transfer_bound=noise_bound=.005`。其他数值、非有限值或不支持的格式不会获得排除凭据。相机或外观参考导致空区间时，本版拒绝凭据并让调用方保留区间，未另设矛盾证书类型。

观测源是冻结有理数生成器的 `float(exact(value))`，后经 JSON 写出／读回保持同一 binary64 值；主协调者已核实 A12 `data_builder.py` 的 `float_tree` 和照片转换路径，并沿用到新数据。核验器对每个有限读数 x 使用

`[(prev_binary64(x)+x)/2, (x+next_binary64(x))/2]`

作为转换前精确数的**闭**包围区间，所有运算均为 `Fraction`。闭区间包含舍入中点，两端并不根据 ties-to-even 删除；这是保守放松。照片、coupon 和 reference 再与 `[0,1]` 相交。这既不把显示十进制直接假设成真实观测，也不把 binary64 读数假设为转换前的精确值。结论要求上述最近舍入来源；任意来源十进制文本、截断或其他未知量化过程不自动在本规范内。物理常数 `.01,.005,.1,.2` 采用任务规定的精确有理数，与观测舍入误差区分。

证书端点采用浮点搜索端点的精确 binary64 有理值，写作规范约分的 `n/d` 字符串；这是被检验的闭区间本身，不额外解释为某个未经记录的理想分割点。状态中的数值端点按其精确 binary64 值比较，绝不加浮点容差填补缺口。

## 独立重建的必要条件

中相机中心恒为 0。若真实相机读数 `u1∈[l1,h1]`，共享标靶误差满足 `k∈[-.01,.01]∩[-1000h1-.01,-1000l1+.01]`。侧相机中心区间为 `[-1000hv-kmax-.01,-1000lv-kmin+.01]` 与标称中心 `±.2` 的交集。消元后保留必要区间，允许原有相关性丢失；因此只可能放宽可行域。

像素为 `[-1/5+i/20,-1/5+(i+1)/20]×[-2/25+2j/75,-2/25+2(j+1)/75]`，面积 `1/750`。对前景深度闭区间及每个相机中心闭区间，边缘 `(X-C)/z,Y/z` 的极值在端点角组合精确求得。每条左／下边最大值与右／上边最小值构成内矩形，反向极值构成外矩形。空内矩形面积为零。后景在深度 720，分支为 none、partial、full；所有矩形坐标同任务定义。

前景左右材料的像素面积分别以其内／外矩形求界。后景可见面积按集合包含关系

`Rinner \ Fouter ⊆ Ractual \ Factual ⊆ Router \ Finner`

得到下界 `area(pixel∩Rinner)-area(pixel∩Rinner∩Fouter)` 及上界 `area(pixel∩Router)-area(pixel∩Router∩Finner)`。这些精确式保留同一后景矩形在面积差中的关系，避免独立面积相减破坏外包围。矩形与像素交的宽高用精确 max/min 计算，先遮挡再面积积分。

变量顺序固定为 `d0_0,...,d2_2,b0,b1,b2,ba0,ba1,ba2`；`d=1-c` 属于 `[0,1]`，照片偏置 `b∈[-.01,.01]`，外观偏置 `ba` 独立。reference 的未饱和单边约束与 `[-.01,.01]` 相交成为 ba 的 box。照片的每个未饱和通道读数产生 `I+b≥lo-.01` 或 `I+b≤hi+.01`；三个通道取交集得 `[L,U]`。包含 0 的源区间不生成下界，包含 1 的源区间不生成上界。

行号按以下确定顺序从 0 开始，所有行统一表示 `A x ≤ rhs`：

1. v=0..2、k=0..2：coupon 下界适用时 `d[v,k]-ba[v]≤1-lo+.01`，随后上界适用时 `-d[v,k]+ba[v]≤hi-1+.01`。
2. k=0..2、v=0..2、w=0..v-1：先 `d[v,k]-d[w,k]≤.1`，再反向行。
3. 每个 v：p=0..47 的所有有限照片下界行 `Σ Amin d-b≤1-L`；再 p=0..47 的所有有限上界行 `-Σ Amax d+b≤U-1`。

所有精确物理可行点必满足这些行与 box；反方向不成立。核验器按此独立枚举行号，只为非零乘子支持的照片行构造面积，以减少认证成本。`reconstruct_constraints` 可供审计完整物化同一组独立必要行。

## 证书和矛盾判定

证书字段：`schema='a13-rational-box-farkas-v1'`，`observation_sha256`，`branch`，`lo/hi`，`variable_order`，整数 `row_count`，`multipliers=[[row_index,'n/d'],...]`，`margin='n/d'`。稀疏乘子必须严格正、行号唯一且在范围内；零项直接省略。所有有理数必须已约分，整数也写作 `n/1`。

观测摘要为只选择 `photo,camera,appearance` 后的 `json.dumps(...,sort_keys=True,separators=(',',':'),allow_nan=False)` 的 UTF-8 SHA256。它绑定本次读取的数值和元数据，辅助发现观测误换。摘要不是签名，也不声称某凭据只可能适用于一个世界；修改未使用约束并重新绑定摘要后，同一数学证明仍可能有效，测试明确允许这种情况。

将乘子 y 与独立重建行相乘求和，得 `a=Σ yA`、`β=Σ yrhs`。box 上线性函数的精确最小值为 `m=Σ a_j*(lower_j if a_j≥0 else upper_j)`。只有 `m-β>0` 且与序列化 margin **完全相等**时接受。这一规则直接证明 box 内不可能同时满足必要条件；无需浮点求解器状态，也无需浮点 epsilon。返回 `{valid: bool,reason: str,...}`。失败返回原因，调用方不得排除。

`verify_certificate(observation,record)` 接受证书本身或含 `certificate` 的 exclusion wrapper；后者的 `branch/lower/upper` 必须与证书精确相等。`verified:true` 标签不能代替核验。

## 搜索状态与完整覆盖

`verify_state` 消费 `cells`、`exclusions`、`lower/upper`（或可选 `hull`），独立核验每个实际排除凭据。每个分支内，保留区间与排除区间排序后必须从 540 精确连续覆盖至 660；共享闭端点允许，任何正长度重叠或任意小的未记录缺口均拒绝。端点可重复保留，单点区间不视为正长度重叠。整体 `lower/upper` 和存在时的 `hull` 必须等于保留集合的精确最小／最大端点。至少保留一个区间，避免空集合仍用投影规则。

显式 `fallback_reason`、`fallback` 或约定 `reason`（`encoding_or_solver_error`、`empty_enclosure_identity_fallback`）且三个分支均保留完整 `[540,660]` 时，允许旧排除记录与全域回退重叠；仍逐条核验这些历史排除记录，返回 `partitioned:false,conservative:true`。其他状态不能借 fallback 标签绕过覆盖检查。未通过的提案应放在 `certificate_attempts`，不应混入 `exclusions`。

此检查覆盖已保存状态的空间记账，不证明运行期间从未使用未记录的临时排除；最终预测须另由冻结 decoder 与保存状态核对。它也不证明超出冻结合成模型的真实成像、数据来源真实或有用精度。保留区间是必要条件的外包围，不是存在兼容世界的证明。

## 验证记录

测试脚本使用独立直接点世界矩形积分生成观测，不导入 builder。覆盖 12 个有理数可行世界（三分支、端点和非整数深度）分别在点区间及全域的所有必要行成立；附加独立照片／外观偏置、全白饱和可行例、舍入源区间包含性。手工暗像素的单行凭据在 z=600、none 分支有可解析 margin `71/200-1/2^1075`；系数、符号、margin、变量顺序、行号、观测绑定、几何／分支篡改均拒绝。分区覆盖测试包含一个 binary64 ULP 的缺口和重叠、wrapper 不匹配及全域回退。

运行使用现有 `/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B` 与 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。初次测试因测试辅助函数尚不接受 Fraction 端点及测试意外改变行布局而失败，CPU 0.0487 s；修复后前 8 项通过，CPU 0.2685 s。后续增加独立偏置测试，9 项全部通过，CPU 0.3036 s；最终要求 state 明示 hull 后，9 项再次全部通过，CPU 0.3150 s。四次测试 body 合计 0.9359 CPU s，不包含标准库导入开销。主协调者另已确认 builder 联调和首个开发状态通过本核验器。独立作者累计开发执行远低于分配的 80 CPU s。未读取 holdout 真值，未修改旧方法或数据。
