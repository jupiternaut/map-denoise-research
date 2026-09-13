# V5 真实输入迁移

本表是恢复已有注入量到原测量云，不是独立真实几何精度。低改写/低自响应也不证明去噪。

192/192 输出成功；复用同一24输入、6片区、2场景；sigma=2 mm。

|模式|方法|XYZ恢复中位RMS mm|构造法向 mm|构造切向 mm|零注入改写 mm|
|---|---|---:|---:|---:|---:|
|zero|identity|0.000000|0.000000|0.000000|0.000000|
|zero|pool_independent|0.678011|0.363194|0.265824|0.678011|
|zero|pool_compatible|0.717192|0.386438|0.346085|0.717192|
|zero|shared_group_slope|1.099828|0.387512|0.648729|1.099828|
|zero|node_intercepts|0.705655|0.377502|0.301475|0.705655|
|zero|graph_shared|0.842999|0.363417|0.267543|0.842999|
|zero|measure_unbalanced|0.204497|0.150513|0.040376|0.204497|
|zero|official_gicp|10.727074|5.085639|3.322927|10.727074|
|normal_translation|identity|3.500000|3.500000|0.000000|0.000000|
|normal_translation|pool_independent|3.524791|3.499508|0.313258|0.678011|
|normal_translation|pool_compatible|3.535175|3.499475|0.389485|0.717192|
|normal_translation|shared_group_slope|3.593930|3.500075|0.648936|1.099828|
|normal_translation|node_intercepts|3.529492|3.499491|0.347814|0.705655|
|normal_translation|graph_shared|3.528699|3.499531|0.319240|0.842999|
|normal_translation|measure_unbalanced|2.446451|2.438339|0.056504|0.204497|
|normal_translation|official_gicp|10.167634|4.772354|3.322927|10.727074|
|tangent_translation|identity|3.500000|0.000000|3.500000|0.000000|
|tangent_translation|pool_independent|3.455469|0.372453|3.418798|0.678011|
|tangent_translation|pool_compatible|3.463845|0.397943|3.421875|0.717192|
|tangent_translation|shared_group_slope|3.566714|0.398244|3.500363|1.099828|
|tangent_translation|node_intercepts|3.462723|0.395080|3.421364|0.705655|
|tangent_translation|graph_shared|3.509522|0.372520|3.472847|0.842999|
|tangent_translation|measure_unbalanced|3.349387|0.162145|3.335580|0.204497|
|tangent_translation|official_gicp|10.727074|5.085639|3.180256|10.727074|
|small_rotation|identity|3.500000|3.493785|0.207428|0.000000|
|small_rotation|pool_independent|3.521818|3.492904|0.490352|0.678011|
|small_rotation|pool_compatible|3.530700|3.492796|0.556922|0.717192|
|small_rotation|shared_group_slope|3.668385|3.493364|0.701479|1.099828|
|small_rotation|node_intercepts|3.525612|3.492850|0.522710|0.705655|
|small_rotation|graph_shared|3.529019|3.492952|0.511668|0.842999|
|small_rotation|measure_unbalanced|3.519993|3.493785|0.208530|0.204497|
|small_rotation|official_gicp|9.754645|5.617259|3.269593|10.727074|

逐片区结果、支持率、当前输入PCA与候选轴分歧见 RESULTS.csv；配对增量见 AGGREGATES.json。
逐点moved_mask与候选支持mask分开保存。旧接口没有逐点mask时明确缺失，不能用低移动量补造。
GICP是官方求解器的输入锚站包装，无点级去噪；原始变换和输入定义平移规范在输出JSON中。
measure仍为原3步扫描常量纠偏。功能与pool/graph不同，不作无差别去噪排名。
未用参考或构造法向重置算法，没有逐扫描/逐层/参考对齐，没有新增独立真实场景。
总wall 25.446s；方法调用累计 19.717s。单线程CPU但其他任务可能争用。
源码、整个V4原run和输入hash前后检查；VERIFICATION.json重读所有输出并独立复算分解分数。
