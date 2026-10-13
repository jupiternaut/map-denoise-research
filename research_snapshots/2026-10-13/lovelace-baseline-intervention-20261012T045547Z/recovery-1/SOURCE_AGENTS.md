# Recovery-1：公开输入解码修复
language: zh
host: liekkas
唯一新写根：/srv/slam-research/grf/map-denoise/runs/lovelace-baseline-intervention-20261012T045547Z/recovery-1。原父根所有科学源码/协议/data/outputs只读。
本子目录不是新的科学实验，只恢复同一预注册72求解批次。复用父data，禁止新生成或新seed；禁止改baseline/求解参数/动作/评价。
原失败在第一个solve的map(rational, encoded_image)处，0solve/0action。只改JSON有理数解码与路径/原截止时钟。全部原源与失败输出保留。
主批和核验共享父outputs/baseline-v1/BUDGET.json的原1800秒deadline，不重置；不足则INCOMPLETE。
root负责子目录runner及集成测试/报告；imaging_intervention_plan仅子目录verify/test；fresh reviewer只读。
源码修复审查后冻结。全部972动作保存读回后才parseprivate；无GPU/API/SSH/安装/发布。

