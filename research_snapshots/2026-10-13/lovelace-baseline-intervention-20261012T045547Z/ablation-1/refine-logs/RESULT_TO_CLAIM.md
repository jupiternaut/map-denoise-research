# A7/A1/A5 结果到主张结算

日期：2026-10-12 15:28:31，精确主机liekkas；主输出outputs/ablation-v1，核验PASS。

| 主张 | 判断 | 直接证据 | 边界 |
|---|---|---|---|
| C-A1：B80远缘F承担深档条件收益 | SUPPORTED WITH SCOPE | joint修复完整11、删F2、删邻列7/40；E80−E70=340741/401408 mm²；B70删F不变 | 已采集图像删推理约束；匹配数量不匹配信息；同源合成族 |
| C-A5：F与共享偏差推理协同 | SUPPORTED AS INTERACTION | B80深档cal−joint由0.239060382降到0.076523139；差中差0.162537244 mm² | 不作物理原因比例；calibration144/144一致 |
| C-A7：嵌套覆盖的一维投影不退步 | SUPPORTED WITH CONDITIONS | joint/cal36/36嵌套；projection0配对退步，hybrid5 | 投影平均MSE略差；不代表跨布局支配 |
| 保留正确初态与负控 | OBSERVED | 每格18正确、24负控KEEP；全部新动作相对初态零损伤 | 本批事实，不是真实照片保证 |
| 普适算法/真实场景收益 | NOT TESTED | 没有新实景或新生成器 | 不升级默认、不发布 |

详细精确账见analysis/CONTRASTS.json；完整目标驱动解释见REPORT.md。

