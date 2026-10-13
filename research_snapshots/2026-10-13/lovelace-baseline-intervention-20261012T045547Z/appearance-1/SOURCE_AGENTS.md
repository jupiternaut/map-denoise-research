# A4：跨视图材料共享机制
language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/appearance-1。父data、recovery-1、ablation-1与历史runs全部只读。此子阶段承接最新用户“继续”，覆盖父阶段已结束的具体任务限制；不改变旧协议或结果。

只做B80十八图像的逐视图材料对照。三全局材料参数复制为九个view-major材料参数，同视图内所有像素继续共享；joint仍三视图bias共享参数。新正式配置仅per_view，cal/joint各18次，共36冷求解324新动作。旧global_shared全图outer及324方法动作只引用，KEEP162作为参照；不新增种子或采集，不做逐像素材料。

root负责run_appearance.py/test_run_appearance.py/test_suite.py及协议报告。appearance_solver_author仅写appearance_solver.py/test_appearance_solver.py。appearance_verifier_author仅写verify_appearance.py/test_verify_appearance.py。审稿代理只读。所有编辑apply_patch，不修改他人文件。

方法与选择不能读取data/private。全部324新动作保存、完整读回计数，再评价私有真值。保持原432标量、像素身份、完整校准、公开baseline80/radius1/5、[540,660]、三遮挡分支、8轮、511盒、15/32mm、hybrid及误差界。扩展contractor只改变维度及材料连接，不改算术、行顺序或迭代语义。独立核验须重建行和传播，不能导入新solver/runner。

新正式主+核验一次3600秒绝对窗口，从正式入口开始，不重置；超时保留INCOMPLETE不覆盖。先过global旧维度身份、public编码边界、view-major稀疏行、true-shared-parameter嵌入、置信动作及篡改夹具。测试用手工非确认夹具，不重造正式种子。

主终点第三档40信息偏移/方法；另全144偏移、三档、18正确、24负控、全部18图的生成分支及重复九材料覆盖。有限outer嵌套实查，不预设有限算法必然嵌套。不以正收益决定机械PASS。不把九初态当九独立图像。

无GPU/API/SSH/install/部署/发布/后台循环。正式启动后新源冻结；当前有限对照完成即停止。
