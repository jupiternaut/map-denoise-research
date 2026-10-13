# A14 检查点：完成

日期2026-10-13。主机liekkas。唯一工作根：
`/srv/slam-research/grf/map-denoise/runs/lovelace-joint-boundary-20261013T123336Z`。
它解析到现有`/mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/`同名目录，未替换环境。

## 已完成

1. A12开发观测上定位停止精度与相机相关性松弛差异；祖先只读。
2. 单一候选：A13快法+固定四闭相机域覆盖，全部子域精确矛盾才排除。
3. 开发测试、fresh源码审查、冻结、原生成器新种子确认。
4. 五配置base/fine/joint@6、legacy@6/@24；18世界、90冷调用、630预测。
5. 54快状态完整核验后评分；fresh审计独立重建前向/全部误差，最大统计差异0。
6. 报告与日志保存。未改祖先数据、未上传GitHub、未改部署默认、未用GPU或外部API。

## 数值结论

扰动MSE：identity22.000893；base6.880610；fine6.845115；joint6.794984；legacy@6 7.016782；legacy@24 6.781458，单位mm²。
joint冷调用0.579480秒，旧长配置7.638390秒：快13.18倍、误差仍高0.1995%。base到旧长配置的同批误差差额收回86.3584%，不是总误差下降86%。
joint对base：41改善/67同/0差（108扰动），12/6/0（18世界）。对fine：37/33/38输入、8/3/7世界，均值改善但不逐例支配。
对identity：75改善/33同/0损伤；18正确输入保持。三个负控world是同一机制的三个种子；保留其失败修复结果。

## 审查状态与限制

源码PASS；结果WARN，同家族provisional，不能称跨家族验证。确定性前向/误差/证明检查PASS。
SOURCE_FREEZE内56项未变，但11项生成器独立核验动态依赖未事前登记。现有清单保持原样，不追溯补造；完整列表在refine-logs/EXPERIMENT_AUDIT.md/json。
仅同族模拟新参数；非真实点云、非跨机制、无统计显著性结论。legacy与新核的浮点观测有理化契约不同，未声称同一形式保证。
正式及附属CPU约246.37秒，fresh独立算术另1.058661秒；开发成本不主张精确完整总额。

## 下一入口（尚未执行）

先读REPORT.md与审计。若继续，固定四域，测试旧长配置与joint端点差距中的深度：它是否仍满足四域必要系统，还是仅被停止尺度留住？不按真值选诊断点，不在当前封存确认上事后调参；将模型/表示松弛与终止尺度分开后再定新锁定协议。
不要直接叠加8/16子域、增加照片、重写loss或回退为多路线竞赛。

## 重现入口

COMMANDS.md列顺序；所有输出排他创建，不能原地覆盖正式矩阵。现有Python：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B`，四线程变量为1。
正式实验进程已结束（90receipt终态完成，主执行会话退出0）；不需要继续等待后台求解。
入口工件：SOURCE_FREEZE.json；data/confirm/；outputs/formal-v1/；DIAGNOSIS.md；JOINT_DERIVATION.md；independent_audit.py/json；refine-logs/EXPERIMENT_AUDIT.md/json。
