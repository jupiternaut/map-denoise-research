# A7 跟踪

时间：2026-10-13 07:12 Asia/Shanghai。主机liekkas；原目标保留有效几何、修复不足。
设计/注册：固定PROTOCOL和40/324/972分母。
手工：54/54 PASS；源码：same-family/provisional PASS，全部发现已修复后冻结。
正式：outputs/pose-v1，全部324/972保存；最终source-set guard失败，INCOMPLETE。未补跑，未改失败收据。
根因：事后analyze_pose.py临时放进根目录glob，已移至refine-logs；当前161原锁文件、9输入完全一致。
独立保存记录诊断：SAVED_RECORD_NUMERICAL_PASS，formal_acceptance=false；36真实完整生成参数覆盖PASS，旧A6的94工件未改。
主40：17F/22U/1旧I，零偏移6，固定seed0；288偏移103F/150U/35旧I；36正确24F/12U。
分析/REPORT/CHECKPOINT完成；没有几何修复增量。新锚定2×2仅提案。
GPU/API/安装/发布/部署均未执行。原3600秒期限未重置。
