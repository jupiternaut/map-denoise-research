# 给 Cursor 的启动说明

## 用户复制这一段到新会话

```text
你在 liekkas 的新工作区：
/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration

完整读取 AGENTS.md、TASK.md、research_plan.md、DATA_SOURCES.md、ACCEPTANCE.md、LONG_RUN.md、TASK_QUEUE.md、CHECKPOINT.md。
执行 python3 preflight.py，确认目标主机、路径和现有输入。
这次不是只给计划：请自主实现、运行并验证 V25 局部表面再生。
从首次执行起按约 8 小时工作窗口推进。能并发就分为观测/基线、V25 算子、独立评价三线。
旧工程和旧检查点只读，所有新写入都在本工作区。开发数据上允许大胆改变构造。
TASK.md 是起点，不是唯一算法；允许提出 IDEA_LAB.md 之外的局部再生算子。
使用“提出可检验预测→实现→真实局部实验→保留/修改/换机制”的循环；一次负结果不停止整项任务。
每次准备结束回复前检查 TASK_QUEUE.md；窗口内还有可执行任务就继续，不只写下一步建议。
不要停在计划、环境安装、单元测试或一张漂亮图；尽可能完成真实局部输出、同观测基线和封存评价。
所有阶段及失败留档；不存在的工具不要反复猜名。按 GPU_POLICY.md，GPU 空闲才启动 GPU 阶段，忙时走 CPU 分支。
我准备睡觉，常规技术选择自行决策。不能越权、不能编造收益、不能拿 GT 给方法选答案。
结束时交付 REPORT.md、实际点云与对照图、逐 ROI 结果、复现命令和可续跑 CHECKPOINT.md。
若客户端或上下文中断，下次先读检查点、任务队列并核对进程，从未完成步骤继续，不从头重跑。
```

## 打开方式

在 Cursor 用 Open Folder 打开**本文件所在目录**，新建会话，再粘贴上面的文字。
不要把旧 map-denoise 工程作为另一个根目录加入 multi-root workspace。
AGENTS.md 是通用入口；`.cursor/rules/v25.mdc` 是本项目常驻引导；AGENT.md 只是兼容指针。

## 已有 / 尚无

- 已有：任务书、数据索引、执行计划、验收规则、只读 preflight 脚本。
- 长程补充：LONG_RUN.md（运行/恢复）、TASK_QUEUE.md（连续工作）、IDEA_LAB.md（大胆构造方向）。
- 尚无：V25 算法、V25 运行结果、安装好的 COLMAP、可靠的空闲 GPU 承诺。
- 本目录不是已完成 V25 的报告；不要把规划阈值写成实验数字。

## 长跑的现实条件

Markdown 不能让宿主绕过执行时限、计费上限或自动恢复。
本次没有创建定时任务，也没有在后台启动 V25。
请保持 Cursor 会话/Agent 运行，并确保主机不会因睡眠而暂停工作。
不要让 Agent 为此擅自修改系统电源设置。

若会话提前结束但任务尚未完成，复制：

```text
继续 V25。先读取 CHECKPOINT.md，核对已记录进程和已封存输出，执行最早未完成的步骤。
读取 TASK_QUEUE.md、LONG_RUN.md；保留已有有效成果，在授权范围内继续真正的实现与实验。
不重复下载/运行完成的阶段，不把计划里的命令当成已存在的程序。
保持 AGENTS.md 权限边界；只追加本次运行记录。
```

## 只读起步命令（现在已可执行）

```bash
cd /home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration
python3 preflight.py
```

其余算法命令需要执行 Agent 实现后记录进 COMMANDS.md；本任务书不伪造可运行接口。

## ZIP 使用边界

压缩包是轻量任务交接，不含 98 张照片、大模型、旧缓存或已安装环境。
在本机直接打开原工作区即可。解压到别处只可审阅；不能把另一路径/机器的成功冒充本目标。
迁移到另一主机/路径需用户确认并更新数据映射与目标契约，不能绕过 preflight 的身份检查。
