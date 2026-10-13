# A5 手工夹具验收

主机 liekkas，runtime /srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python；CPU单线程、-B。

命令：python -B -m unittest discover -s . -p 'test_*.py' -v。

root完整复跑：44 tests，0.120秒，退出0、OK。只用手工夹具，不读取正式population/private。

覆盖：精确加权矛盾/带符号box最小值、八轮区间遗漏、边界见证、错误对偶/NaN/timeout/伪证书拒绝、UNKNOWN保留、原trace不改、同leaf见证、1296+324读回/唯一门禁、共享绝对预算、禁止旧目录写入。删budget叶后冻结trace元数据口径已修，正式矩阵budget叶为0。

这证明实现契约的手工案例通过，不代替正式成绩或真实成像soundness。
