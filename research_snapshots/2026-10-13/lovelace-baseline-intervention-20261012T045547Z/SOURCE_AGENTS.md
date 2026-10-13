# 单次基线成像干预：职责与范围
language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z。
目标保留有效几何、修复不足；只比较公开相机位置(-70,0,70)与(-80,0,80)mm，不扫描。
旧runs全部只读，不修改或monkeypatch历史几何常量。新geometry显式接baseline，缓存键包含baseline。
root负责prepare_baseline.py/run_baseline.py/test_baseline.py、协议与报告。deep_support_analysis仅写baseline_geometry.py/baseline_solver.py/test_baseline_geometry.py；imaging_intervention_plan仅写verify_baseline.py/test_verify_baseline.py。fresh reviewer只读。
新seed61005，原prefix shared-appearance-v1，原6模板3深度9初态；两条件共用同一pose/noise tape，不加入baseline或unit到随机种子。
36公共图像324任务；KEEP/calibrated/joint共972动作；两拟合方法72冷求解。同条件同T_cal与511盒/branch、8轮、15/32mm，clip一次。不得额外材料采样、LP、学习器、参数扫、GPU/API/SSH/安装/发布/部署。
prepare单独进程生成公开图像/任务与私有评价真值；方法只用公开图像baseline/radius/clipped和初态动作。全部972动作保存读回、唯一计数后才parseprivate。
主批+完整核验共享1800秒绝对截止，超时保存INCOMPLETE不重置。正负科学结果不作为机械PASS前提。B70身份和B80点/区间守恒须在非确认夹具测试先过，源审查后首次正式生成。
源码启动后冻结，任何失败/修订不得覆盖已有结果。用apply_patch编辑。

