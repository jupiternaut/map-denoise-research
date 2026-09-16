# V25 / Cursor 长程研究任务包

**这是已准备好的任务包，不是已运行的 V25 算法。**

目标：从多视图证据重新估计局部表面的形状和有效支持，尝试恢复旧点位移不能修复的几何错误。
用户授权开发上大胆改变构造；旧数据/旧检查点只读；GPU 空闲才用。

## 马上开始

1. 在 liekkas 上用 Cursor 打开本文件所在的原工作区。
2. 新会话粘贴 START_HERE.md 中的启动提示词。
3. 先执行 `python3 preflight.py`，然后按任务队列真正编程与实验。

工作区：`/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration`

## 文件各管什么

| 文件 | 职责 |
|---|---|
| AGENTS.md / .cursor/rules/v25.mdc | 行为、目标身份、权限和常驻入口 |
| START_HERE.md | 启动与恢复提示词 |
| TASK.md | 要验证的几何能力，而非唯一算法配方 |
| research_plan.md | 阶段、依赖、并行角色 |
| LONG_RUN.md / TASK_QUEUE.md | 不在 smoke 后停机；长任务恢复与下一项工作 |
| IDEA_LAB.md | 大胆构造示例与如何摆脱无效路线 |
| DATA_SOURCES.md | 已存在的真实数据、相机与只读来源 |
| ACCEPTANCE.md | 同观测基线、独立参考、变点数公平评价 |
| GPU_POLICY.md | GPU 空闲判断、CPU 备用路径 |
| CHECKPOINT.md | 执行状态；初始未运行 |
| preflight.py | 可执行的只读路径/环境检查，不运行重建 |
| PACKAGE_MANIFEST.json / HANDOFF_VALIDATION.json | 打包完整性和任务准备验收，不是科学实验结果 |

## 长程但不虚报自动化

首轮约 8 小时，允许多轮真正不同的构造和失败后继续；上下文切换按检查点恢复。
提示词不能绕过客户端执行上限、用量上限或主机睡眠。没有后台定时器、付费 API 或远程任务。
若 Cursor 提前结束，使用 START_HERE.md 的恢复提示词；不需要重做已完成的数据准备。

ZIP 只包含任务文档和预检脚本，不带大数据、不带密钥、不安装环境。实际执行仍依赖本机已列出的数据。
