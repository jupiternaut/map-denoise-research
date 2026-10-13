# A9 保证范围与审查分工
language: zh

## 可继承的核心能力
本阶段试图把“拒绝错深度”变成实际坐标修正：对整个连续域找一个保留真世界的外集，再夹住现任。选择规则不学习标签，也不读取GT来选点。

设闭合外集 O 的包络为 H=[L,U]，真深度 t∈O。HULL输出 p=min(U,max(L,s))：
- s∈H时p=s；
- s<L时s≤L≤t；
- s>U时t≤U≤s。
因此|p−t|≤|s−t|，平方误差同样不增。正确现任s=t不动。保证依赖外集覆盖，不保证边界坐标有完整前向解，不保证不连通UNION投影，也不等于点云整体拓扑/完整性保证。

## 整区间证据
深度与相机偏移构成的双线性投影由矩形内/外界包络。逆深度q变化的四条边扫过条带，像素内对称差有K|Δq|上界。固定e时后面投影不随前面深度变化，后面可见面积变化≤完整FRONT对称差。两个界相交仍 sound。
必要照片/漂移/校准/外观行允许的世界至少覆盖完整物理解释。只有在这些必要行和原始变量盒上严格成立的有理矛盾，才删除整个深度单元；未决全部保留。

## 角色与证据
- Solver作者：/root/anchor_solver_author；只编写continuous_solver.py及18测试，手写夹具，无正式或private读取。
- 独立checker作者：/root/continuous_verifier；只编写verify_continuous.py及14测试；独立重建区间几何/必要行，复用旧独立精确算术，不import新solver。
- Fresh消融规划：gpt-6-astra/xhigh，fork_none。
- Fresh源码审查：gpt-6-astra/xhigh，fork_none；11科学文件哈希及ACCEPT，另做108手写世界/46656面积包含检查。
- root：协议、输入编排、输出动作、事后评价与报告；8管线测试使总计40测试。A8生成事实为事后参照，不参与选择。

以上是same-family provisional assurance：角色独立、程序重建、源码审查，不声称其他模型家族或外部研究组独立复现。

## 修正链与预算
首次38项合并测试发现checker断言整个共享sys.modules中没有solver，这会把其他测试先导入solver误判为checker依赖。改为AST加干净子进程检查，仍要求checker不得导入solver及相关作者求解模块；40项再验通过。该修改不改变算法输出。
独立设计给出的300秒验证预留被root在正式输入前扩为1500秒；窗口3600/main2100，每分支3秒、127节点、15/32mm容差。阈值不是由正式几何误差选择。
3秒为合作式截止，精确分类和记录收尾可略超；全流程另有不可重置总截止。有限预算所得四臂集合可能不经验嵌套，必须公开原始集合和实际计算量。

## 正式结果后的独立算术复算
/root/continuous_result_recompute为fresh只读默认模型上下文；仅在COMPLETE/PASS后读取旧冻结事实。独立重建2592 ROWS（41472字段）、136 SUMMARY（2448字段）、2592投影选择，逐字段零差异；保存集合的覆盖成员关系及观察/预算统计匹配。没有import新solver、runner、auditor，没有重做照片几何证明。
完整response归档到.aris/traces/analyze-results/2026-10-13_run01/。原始spawn请求只保存范围摘要，最终follow-up为逐字回执；不把此算术审查当成正式前源码门禁。

root另建的事后描述分析首版将variation_index误当采集实现，实际它是eta，错误合并成2组。核对发现后改为按观察ID去除eta后缀，明确assert 36观察/18共享采集；新ANALYSIS_20261013_103705.json与独立复算的15改善/3不变一致。旧ANALYSIS_20261013_103412.json保留作修正链，但该文件paired_gain_counts不采用。此错误未进入冻结正式算法、公开选择、ROWS/SUMMARY或独立核验，不改主终点。REPORT中的18组统计采用独立正确复算。
