# A4-V：共同材料身份与有界视图外观变化
language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-variation-1。本子阶段承接用户最新“继续”，覆盖父阶段已结束的具体任务限制；父data/recovery-1/ablation-1/appearance-1及所有历史结果只读。
目标保留有效几何并修复不足。本轮只做锁定B80六模板×三深度的受控外观干预，不新采集/新seed/新阈值搜索/学习器。
公开η=0和1/10；两档bounded均用设计界δ=1/10。实际d_vk=d_k+η*q_vk*min(d_k,1-d_k)，q=[[-1,0,1],[0,1,-1],[1,-1,0]]。方法只接照片、radius、baseline、clipped、method与arm规定的δ，不接真实材料/η/q/pose/noise。
strict_shared(δ=0)3/6变量、independent(δ=1)9/12变量、bounded(δ=1/10)9/12变量。δ端点调用冻结语义保证桥接；bounded只加材料差signed rows，保留8轮/511盒/15/32mm/全校准/旧hybrid。
η0 strict/independent72解与648动作复用封存；其余144冷求解/1296新动作。两档共1944方法动作+324KEEP，36观测、324任务。全部方法动作保存读回唯一计数后才解析新private评价。
root写prepare_variation.py/run_variation.py/execute_variation.py及对应tests、协议报告。appearance_solver_author仅写variation_solver.py/test_variation_solver.py。appearance_verifier_author仅写verify_variation.py/test_verify_variation.py。reviewer只读。
生成在独立进程：允许重建旧private作前向干预，禁止方法读取private；生成必须经源审查后，η0公开图像/任务/事实与旧身份。源启动后冻结，旧目录不得改写。
bounded与independent必须覆盖生成深度、生成分支、九材料及joint偏差；strict的失配/损伤必须记录但不是机械失败。negative模板等材在不同视图未必依然等材，不预设干预后负控身份仍成立；按原模板与实际mismatch分别报告。
CPU单线程，一次3600秒绝对预算包括正式生成+求解+独立核验，不重置。超时INCOMPLETE，旧产物不覆盖。先手工非确认夹具测试与计时，审查，再正式批次。
全部编辑apply_patch。无GPU/API/SSH/install/发布/部署/后台循环。本矩阵结束即停止。

