# 裁剪、宽界与传播：冻结实验协议

日期2026-10-12（Asia/Shanghai）；host liekkas。唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-observation-bridge-20261011T232502Z。
用户目标：保留有效几何、修复不足。来源是上批NEXT_EXPERIMENT.md已设计A+B，本批只实施这两块，不运行C/D/E/F新模型。

## 主张与证据

C1：以配对设计分别测量裁剪、宽观测误差界、固定宽界下偏差及其交互，解释旧未裁剪115次修复与新β=.01/裁剪32次修复之间的差异。允许零效应与反向结果，不预设谁主导；结算依赖同图像/同界的匹配，不把不同单元差异当独立可加百分比。
C2：固定β=.01新图及δ=.02，只改4/8/16传播轮数，检验是否还有该局部范围内的计算余量；一致不等于精确求解或不可辨识。

## 来源与身份

旧source只读：
BASE=/srv/slam-research/grf/map-denoise/runs/lovelace-shared-appearance-20261011T052300Z/outputs/confirmation-v1。
ABLATION=/srv/slam-research/grf/map-denoise/runs/lovelace-shared-ablation-20261011T055200Z/outputs/ablation-v1。
BIAS=/srv/slam-research/grf/map-denoise/runs/lovelace-view-bias-20261011T092300Z/outputs/view-bias-v1。
复用R0epsilon为ABLATION的global_gray完整511盒结果，CBdelta为BIAS中β=1/100的global_shared_wide结果；验证谱系、图像、参数、初态、已有VERIFY，不伪装新确认。

## 八配置矩阵

R原始已加噪未裁剪图、P裁剪到[0,1]、b=(-.01,0,+.01)仅生成端，epsilon=.01、delta=.02。

| unit_id | 图像 | epsilon | rounds | 求解来源 |
|---|---|---|---:|---|
| R0epsilon | R | 1/100 | 8 | 历史复用 |
| C0epsilon | P(R) | 1/100 | 8 | 新求解 |
| R0delta | R | 1/50 | 8 | 新求解 |
| C0delta | P(R) | 1/50 | 8 | 新求解 |
| RBdelta | R+b | 1/50 | 8 | 新求解，未裁剪机制诊断 |
| CBdelta | P(R+b) | 1/50 | 8 | 历史复用 |
| CBdelta4 | P(R+b) | 1/50 | 4 | 新求解 |
| CBdelta16 | P(R+b) | 1/50 | 16 | 新求解 |

每配置原54观测、486初态，其中432偏移、54正确、72偏移负控；同6模板×3深度×3种子61001/61002/61003。8配置×486初态×shared/KEEP两方法=7776动作，其中新共享求解6×54=324图像配置、2916新共享动作；972历史共享动作与3888新KEEP动作另计。不能把配置重复当独立场景。所有动作必须落盘读回后再解析私有事实评价，继承共享材料与相机不确定界。

所有非KEEP保持global_gray、none/partial/full整场景分支、材料[0,1]、公开radius、511盒/分支、15/32容差与旧select_action。每新共享观测清冷缓存，不改旧solver。solver输入只包含图像、radius、公开epsilon/rounds/max_boxes/tolerance；不含unit_id/真值/模板/种子/偏差标签，不使用actual b反校正。RBdelta允许图像超出[0,1]，不误套上一轮裁剪图接口；它是受控原始信号诊断，不是真实相机可获取图。

## 配对终点

C0epsilon−R0epsilon：仅裁剪，窄界。
R0delta−R0epsilon与C0delta−C0epsilon：各固定图像只放宽界。
RBdelta−R0delta与CBdelta−C0delta：固定界加入偏差。
CBdelta−RBdelta与C0delta−R0delta：固定界裁剪效应及有限设计交互。
CBdelta4/16相对CBdelta：只改传播轮数。

主指标：偏移初态MAE/MSE、改善/同分/恶化、逐初态配对；独立报告真深度覆盖54、正确KEEP54、负控KEEP72、INCOMPATIBLE、凸包宽度/端点/保留集、节点与未决叶、冷缓存成本。分模板/噪声种子均值±SD，不做真实跨场景显著性检验。不要求合法方法间每例误差单调，只要求相对KEEP安全。

## 阶段与预算

M0：单测含先裁剪再加偏差的反例、raw越界、复用谱系、状态分母、全部决策前不读truth；非确认夹具测时。
M1：独立于运行作者的源码审查，若新agent不可用如实用既有代理，same-family/provisional。
M2：一个固定单CPU主批，0GPU/API/安装，主运行+独立核验总窗口5400秒。源码和PROTOCOL锁定后不得编辑；每观测checkpoint，若超时输出INCOMPLETE，不缩小样本追正数。
M3：另一作者核验生成、历史复用、树/证书/动作、评价和计数。使用已有核验核但不得导入新runner计算结果作为独立验证。
M4：报告C1的配对测量和C2的局部稳定/增量，所有正负结果保留。只该固定批次，无外部定时器或≥10独立job队列，无新算法扩展或发布。

## 范围与停止

所有旧目录只读、无部署/GitHub/GitBook修改。后续公共视图偏差及背景简单替代仍未执行，本轮结果用于决定其投入。测试/数值证据验收不等于论文原创性或真实迁移成立。
