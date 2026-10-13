# A4 实验整合契约
- 输入：父data中36冻结观测/324task，按公开baseline80取18/162；保留432RGB、像素序及完整校准。
- 输出：outputs/appearance-v1（唯一新run），outers18×2、decisions324、TIMING36、results324、reference486、coverage36、summary及独立VERIFY。
- 源：旧ablation103文件完整图加本根py/AGENTS/PROTOCOL/REG；输入旧9加ablation3凭据。
- 归属：root runner/测试/文档；solver作者仅solver+tests；verifier作者仅verifier+tests；审稿只读。
- 验收：完整矩阵、同源校准、覆盖、精确树/证书/动作/指标匹配；收益正负不是机械验收条件。
- 停止：3600秒一次主+核验，无重置；完成有限A4即停；无GPU/网络/API/安装/部署。
- 状态：solver作者完成；verifier作者收尾；root集成；审稿进行。
