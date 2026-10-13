# 实际执行与只读检查

目标主机liekkas；路径由a11_io_contract.py验证逻辑/物理同一对象。既有Python，无安装、GPU或API。

正式前：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B check_tests.py`。失败日志保留；最终33项通过，TEST_GATE绑定当前14文件哈希。

唯一正式批次：`/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python -B execute_accumulation.py`。执行器自动运行run_accumulation.py及audit_accumulation.py。输出已经存在；禁止删输出或绕开existence guard重跑。

可重复的手写检查（不运行正式数据）：在本目录执行Python -B -m unittest -v test_accumulation_solver test_verify_accumulation test_accumulation_pipeline。

查看结果：REPORT.md、outputs/accumulation-v1/STAGE_STATUS.json、outputs/accumulation-v1/evaluation/SUMMARY.json、outputs/accumulation-v1/evaluation/COMPARISONS.json。精确有理数以字符串或numerator/denominator编码，不按浮点近似判断平局。

新一轮须另注册新目录、来源与预算；当前脚本是实验身份绑定程序，不是已经做好的跨平台SDK。
