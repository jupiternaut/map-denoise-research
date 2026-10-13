# A14 开发验证记录

2026-10-13，liekkas，既有open3d-019 Python，单线程CPU。

- 诊断Agent：六观测.025/.001，两观测旧24秒，三点整盒/分盒及精确见证复核。21.392384762 CPU秒；详见DIAGNOSIS与diagnostic_*.json。
- 构造Agent：7004必要行单调性检查、6真实开发单元保留、3四盒证书、18篡改拒绝、零截止和单盒未知保留，全部通过；初次测试把旧验证器的valid键误读为verified，修正测试后通过，含失败约0.88 CPU秒。未因结果更改数学构造。
- 主Agent：test_adapter四项通过（0.610秒test wall）；运行joint_tests再次通过（0.375294421 test CPU秒）；三臂开发冷启动分别base0.364、fine0.361、joint0.384秒，全部返回0，原始receipt在outputs/smoke-v1。
- 两次fresh源码审查启动受到agent thread limit而未执行，已有构造/诊断Agent完成后第三次成功启动；不将失败启动记为审查通过。
- experiment-bridge提及的research_contract模板未安装；本轮明确写出输入/输出/评价/权限契约，不安装替代模板。experiment-queue的SSH/GPU脚本不适用当前本机CPU任务，复用已验证本地冷进程队列与状态文件。

本文件不代表正式确认结果。
