# A11 实验完整性审计

日期：2026-10-13。审计者：fresh GPT-6-Astra / ultra，独立上下文、只读；工具返回agent `/root/a11_integrity_review`。review_independence=same-family，acceptance_status=provisional。

## Overall：WARN；机械与算术PASS，无阻断FAIL

| 检查 | 状态 | 独立审计结论与依据 |
|---|---|---|
| A 真值来源与隔离 | PASS | 只转交测量和旧支持；支持保存后才读现任，所有公共证书与动作验收后才解析private。accumulation_solver.py:42、run_accumulation.py:30、audit_accumulation.py:99。真值是既有合成世界，不是外部物理GT。 |
| B 分母与精确算术 | PASS | 从tasks/truth/sets/predictions独立重建972行，全部SUMMARY/COMPARISONS/残余计数逐值一致；324/288/36/48/240/40和旧87分母准确。有理数区分平局。audit_accumulation.py:23、119。 |
| C 结果与声明 | PASS | 72支持、648动作、216新树齐全，收据/公共门/VERIFY哈希及时间先后一致，STAGE_STATUS为COMPLETE。修正报告总秒数、36前向与108覆盖的口径、COMMANDS路径，前版保留history。 |
| D 执行路径 | PASS | 两模式实接冻结内核；逐相机盒确实分类；先重验A9→A10-D，完整核对源/树/保留前沿/动作。accumulation_solver.py:65、audit_accumulation.py:71、verify_accumulation.py:238。 |
| E 范围 | WARN | 已暴露有限合成族；324初态不是324独立场景。一维HULL条件命题不推广至任意三维、真实照片或模型失配。THEORY.md:37、181；REPORT.md:89。 |
| F 评价类型 | PASS | simulation_only准确；实现独立不等于新科学盲测或跨机构确认。REGISTRATION.json:8、verify_accumulation.py:4。 |

## 主张影响

C1：在注册的固定新增上限下获有限描述性支持。继续D与DC的288偏移MSE分别1.113719311与1.094652913，DC低1.711957%。旧87信息性未改善输入新修复4与5。DC对继续D为74更好/210同/4较差，4者来自同一观测；不能声称逐输入支配。

C2：条件推导成立，保存记录一致。两新臂相对旧A10-D无回归，36正确、48负控原样保留。108是臂×观测覆盖检查；独立前向核验36套世界。

费用：新增上限相同，实际1059/3457分类、64.3930/162.6772分支搜索秒。合作式6秒允许轻微超时；搜索秒和不含wrapper、缓存、落盘及共享A10预计算，不证明等实际成本效率占优。

来源：199源码、12输入、14测试门哈希无漂移，历史106封存工件无漂移。审计者只读、哈希和有理算术复算，未跑测试或新实验。公共精确约束重建另外由正式独立检查器完成。

无需修改已封存科学结果。今后的主张必须保留上述范围和参照对象；本轮不部署、不发布。原始审计请求与响应保存在本地`.aris/traces/experiment-audit/20261013-a11/`，不纳入公开工件索引。
