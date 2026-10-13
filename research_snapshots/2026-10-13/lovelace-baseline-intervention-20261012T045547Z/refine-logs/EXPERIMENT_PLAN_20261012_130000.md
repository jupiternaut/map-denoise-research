# Experiment Plan：新增深档边缘信息
目标：保留有效几何、修复不足。
C1几何成像干预能否突破深档0修复；C2共同依赖是否仍有独立增量。
三个baseline family以内：KEEP、calibrated、joint。非frontier方法，不引入学习器。

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 新几何参数化身份/包络及paired tape | 非确认测试 | exact B70身份，B80包含点见证 | CPU数秒 | cache跨baseline污染 |
| M1 | 源码审查 | fresh同族审查 | BLOCKING修复后PASS再生成 | 0GPU | 同族只暂定 |
| M2 | 锁定两基线 | 一个72冷solve批 | 完整972动作保存读回 | 估计CPU3–6分钟 | 成像改变可能伤首档；不得筛掉 |
| M3 | 独立机械核验 | 一次完整核验 | 几何/约束/树/动作/指标一致 | 估计CPU3–10分钟 | 历史primitive共享范围披露 |
| M4 | 主终点、全组、收益与代价 | 已保存数据结算 | 两主张分别判定 | 0新solve | 不把0harm当普遍保证 |

主图表：第三档40信息偏移B70/B80 × cal/joint MSE与改善数；附全六模板三深度、正确/negative控制、T_cal、外包宽与coldCPU。
采用：深档改善数增加+MSE下降且覆盖/正确/negative保持0harm。简化：cal达到joint则不追加复杂度。失败：B80不获益就停止这一baseline，不挑B90补救。
不做参数扫/新seed/LP/学习器/部署。0GPU，总main+verification1800秒。

