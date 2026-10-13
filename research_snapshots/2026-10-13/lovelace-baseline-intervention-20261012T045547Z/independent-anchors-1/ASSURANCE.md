# 核验范围

设计与源码审查代理：anchor_ablation_design，gpt-6-astra xhigh，由消融/实验桥接技能选择；源审查复用本轮设计上下文，明确不是fresh跨模型裁决。原始请求/响应均在 .aris/traces。

方法作者：anchor_solver_author；独立核验作者：polytope_source_review。核验作者保留既往A5源码审查上下文，但没有编写A8方法，不导入新solver、generator或runner；自己的几何、观测行、存在性扩展与前向检查另写。

源码审查PASS是同族暂定准备状态，不等于科学收益或真实迁移。正式VERIFY的职责是所有固定工件的机械一致性：source/input hashes、记录分母、证书、精确完整正见证、seed与144个真实生成世界覆盖。未知结果不解释为不可能，错深度排除不解释为本轮完成坐标修复。

原A7正式INCOMPLETE仍保留，其数字只以独立保存记录诊断标签使用；本轮明确新阶段，不替旧记录追认通过。
