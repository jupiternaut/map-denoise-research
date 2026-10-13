# 源冻结前实现更正

- 手工pipeline的四臂真世界前向测试发现：独立核验器把禁用外观时的tA要求为null，然而协议统一15维中tA始终是3个合法盒变量。修复核验器，不改变测量模型。
- 源码审查发现：共同相机K为空的候选增加了numeric_evaluations但未调用LP，旧A7等式numeric_lp_calls=numeric_evaluations+classifier_lp_calls不再适用。增加skipped_camera_candidates显式计数，核验等式扣除该数。修复记账，不改变科学阈值。
- 求解器作者初版用q=n=s/2实现有界残差分解，数学有效；为与登记的构造逐项一致，源冻结前改成q=clamp(s,±1/200)、n=s−q。

以上全部发生在新公开读数生成及正式批次之前；不以看到正式收益后的修正规則替换登记协议。旧A7源集失败不在本目录追认，通过明确源名单防止同类流程缺口。
