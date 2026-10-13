# A11 检查点

原目标：保留有效几何、修复不足。状态：一次正式CPU批次完成，独立算术验收PASS；语义审查见EXPERIMENT_AUDIT。

主机liekkas；逻辑根 /srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z；物理根 /mnt/data4t-usb/experiments/srv/slam-research/grf/map-denoise/runs/lovelace-evidence-accumulation-20261013T051931Z。runs为既有符号链接，同一对象。旧A10和全部祖先文件未修改。

固定表示、观测、HULL。两臂都继承旧D支持；D→D继续深度细化，D→DC增加四相机盒，同额外128分类/6秒合作式上限、容差15/1024。没有读旧DC答案选择分支。

288偏移：旧D MSE1.146123、153改善；D→D 1.113719、157改善；D→DC 1.094653、158改善。36正确全保留，48负控全不动，原始损伤0。D→DC相对旧D146更准/142平/0差；相对D→D74更准/210平/4较差。5/87旧信息性未改善输入新获改善，82剩余。

主运行317.2396秒，核验332.6597秒，总650.0759秒。搜索秒和64.393/162.677，不能称完整端到端、严格等实际时间。原始数据与证书均在outputs/accumulation-v1。

本轮完成计划，不后台续跑；不改默认，不推GitHub。下一轮若继续，先区分残余外松弛和完整世界歧义，不以真值路由，不重写本次结果，不重置本次预算。

复核入口：REPORT.md、THEORY.md、PROTOCOL.md、TEST_GATE.json、outputs/accumulation-v1/verification/VERIFY.json、EXPERIMENT_AUDIT.md、ARTIFACTS.json/FINAL_SEAL.json（最终封存）。
