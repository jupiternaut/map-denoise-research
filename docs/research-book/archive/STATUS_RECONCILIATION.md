# 旧待办对账：不是报告写过“下一步”，今天就还没做

这是本书最需要优先维护的一页。历史记录的“未做”保留当时状态；这里用后继证据更新今天的判断。已做过不等于做成功，也不禁止更有信息的复现。

| 旧消息或旧报告中的待办 | 后继证据 | 截至本书的正确状态 |
|---|---|---|
| 先比较单点与小邻域共同移动 | [LG025](experiments/LG025.md)、[LG028](experiments/LG028.md) | same-arm 已做，rigid 又作补充；不能原样再称“尚未执行” |
| 真实纹理证据接回学习器 | [LG030](experiments/LG030.md)、[LG031](experiments/LG031.md) | 已完成训练、输出与复核；新证据没有兑现原生安全修复 |
| 做新的真实场景确认 | [LG006](experiments/LG006.md)、[LG026](experiments/LG026.md) | 两批当时新同来源场景已确认；扰动恢复有通过，native未通过；以后重跑是已暴露回放 |
| 补外部几何基线 | [LG026](experiments/LG026.md) | RIMLS/PathNet已运行；官方COLMAP/等信息系统比较仍是另一缺口 |
| V25没有逐源代价体，需重算 | [CPR012](experiments/CPR012.md)、[LG002](experiments/LG002.md) | V25旧工件确实缺；V26已实现逐源证据，不应推断后续全工程仍缺 |
| 把带检验和连续投影的CPR跑一遍 | [CPR007](experiments/CPR007.md)–[CPR010](experiments/CPR010.md) | E0-E/F已运行；有条件增量但升级门槛未通过 |
| 记忆分支缺少初始历史 | [META013](experiments/META013.md)–[META015](experiments/META015.md) | 旧接口缺口成立；当前有限结构族已接入历史，T2/T4改善，T3漂移仍未解决 |
| 目录最优策略不迁移，停止调度路线 | [META007](experiments/META007.md)–[META011](experiments/META011.md) | 先纠正空目录停机/预算截面/条件化；H4增量保留，H6关闭的是该目录的精度竞赛 |
| TLA规格未检查 | [MC002](experiments/MC002.md)、[META005](experiments/META005.md)–[META007](experiments/META007.md) | 原翻译确实未检查；后续有限契约跑过TLC，不能反过来声称所有任务规格被证明 |
| V18的好解是否在新搜索中丢了 | [EG026](experiments/EG026.md)–[EG031](experiments/EG031.md) | 候选保留与同样本能力修复已经做过；crossfit剩余退化需用修复后的对照解释 |
| 新几何模型不能动得更远，扩大Oracle | [LG012](experiments/LG012.md)–[LG014](experiments/LG014.md) | 连续、宽位置池和场专属池均已测；它们的不同上界不能混成一个部署进度条 |
| 论文还没有LaTeX/PDF | [写作与迁移章节](chapters/05-software-and-thesis.md) | 后继本地XeLaTeX已生成90页PDF；不是Prism或学校模板验证，字数页数不增加实验依据 |

## 真正仍未完成的最近一步

最新报告提出的是 **单独校准的纹理见证，检验旧恢复器已选择的移动**，不是再次“添加纹理特征”。本书截止没有执行这个新验证器，也没有结果。是否做它仍由当前用户决定；本书只保存这个精确断点。

## 如何更新

找到旧待办已被后继实现时，增加对应记录与本页关系，不修改旧对话或旧报告。若只有“计划已经写好”，状态只能是“已规划”，不能填“已执行”。若报告原文互相冲突而尚未核清，保留冲突，不任选一个顺眼的版本。
