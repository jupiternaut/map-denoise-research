# 开放模型构造挑战：从错误初始解释走向新干预预测

状态：**建设阶段已跑通 6 个系统；尚未做 12 系统开发锁定或 24 系统盲评。** 可信算法实验，不是闭卷 LLM 测验。旧 2340 世界调度竞赛未继续。

## 目标

检验一个比“完整目录内最优选点”更强的命题：初始科学模型失配时，Agent 能否构造初始候选库没有的可执行解释，通过主动干预检验，并在盲评新工况上取得超过强模型搜索基线的预测收益？

“新”首先指相对参与者初始模型与候选库新，不代表对人类科学发现了新规律。符号回归与主动系统辨识不是空白领域。

本阶段不继续 2340 世界的调度竞赛，不研究点云本体，不构造 GUI、MCP 平台或新的 Agent runtime。

## 任务文件

- `TASK.md`：给实验建设者的设计与实施顺序。
- `AGENTS.md`：实验建设者角色、交付与权限。
- `participant/AGENTS.md`：后续受限参与者的行为契约。参与者只能拿到此文件与公开 API，不得继承建设者上下文。

## 大胆之处

允许修改科学模型的结构、增设可解释内部状态、重写观测映射，并选择物理干预。最终外部任务保持不变。不是让 Agent 从一张完整答案表中挑选，也不是仅把一个模型族升级成另一个预先写好的候选。

## 保留的四条底线

1. 答案不进入参与者上下文或可读文件。
2. 新模型须在承诺预测后取得的新实验上接受检验。
3. 与有足够表达能力的强基线比较。
4. 报告失败、计算成本和全部盲评任务，不只展示成功案例。

无需先证明全局最优或建立完整形式验证体系。TLA+ 不是本阶段前置条件。

## 理论与基线来源（已核实，不是已安装）

- Brunton, Proctor & Kutz (2016), *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. https://doi.org/10.1073/pnas.1517384113
  - 已有稀疏方程发现；必须对比，而不能把“新增非线性项”本身当创新。
- Brunton, Proctor & Kutz (2016), *Sparse Identification of Nonlinear Dynamics with Control*. https://arxiv.org/abs/1605.06682
  - 显式处理输入与驱动；比纯无输入拟合更对口。
- Javdani et al. (2014), *Near Optimal Bayesian Active Learning for Decision Making*. https://proceedings.mlr.press/v33/javdani14.html
  - 信息服务于后续决策；不转移其保证到开放模型构造。
- Shojaee et al., *LLM-SR: Scientific Equation Discovery via Programming with Large Language Models*, arXiv v3 (2025). https://arxiv.org/abs/2404.18400
  - 已有 LLM 提出方程程序并拟合参数，不能把“写出新公式”本身作为我们的创新。
- Abhyankar et al., *LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search*, arXiv v1 (2026). https://arxiv.org/abs/2606.25039
  - 已明确联合假设构造与自适应轨迹获取，是必须进一步读全文/核实现成代码的直接近邻。当前只核验摘要与版本，未复现。
- Kabra et al., *LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs*, arXiv v1 (2026). https://arxiv.org/abs/2605.24043
  - 已研究假设生成、实验选择和机制更新。不能据此声称本提案率先实现自动科研。

因此候选研究缺口进一步收紧为：**失配位置未知时，如何判断应修改动力方程、增加状态，还是修正观测模型，并以前瞻干预预测验证这种修正？** 是否超出现有工作仍需全文比较与实验回答，不以检索遗漏证明原创。

候选软件可考虑 PySINDy / PySR，但执行者须先核对官方实现与当前环境；这里没有宣称其安装、运行或达到任何性能。

## 当前交付边界

本轮只形成可审阅任务包，未启动模拟、模型调用或后台任务。具体隐藏机制与锁定盲评集由建设者在独立区域准备，不应在交给参与者的文件里写出答案。
