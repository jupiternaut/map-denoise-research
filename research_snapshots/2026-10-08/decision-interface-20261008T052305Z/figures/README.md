# 决策接口实验：三组封存结果图

这些图由 `analysis/make_figures.py` 只读生成。绘图使用 `/usr/bin/python3`、Matplotlib 3.10.7、NumPy 2.3.5；没有安装软件，没有调用实验预测器、决策器或评价器，没有修改评分或重新调参。每图提供 PNG（300 dpi）与 SVG。图中文字为英文，使用可区分的颜色、线型和纹理。

`PROVENANCE.json` 记录全部输入路径、字节数、SHA-256、实际绘图数值、输出哈希。读取的 SUMMARY/ROWS/GATE 先与各自评价封存核对；旧 w002 曲线与旧 ordinary-stage 封存核对。绘图结束再次核对输入哈希。

## 1. 旧曲线接口消融

文件：`01_replay_ablation.png`、`01_replay_ablation.svg`。

同一批封存 ED 评分，依次显示 S0+P、S1+P、S1+M、S1+U、S1+R。四面板分别为 −60 mm 初态 MAE、正确初态 MAE、+60 mm 初态 MAE，以及正确初态的严格损伤数。MAE 单位为 mm；每个初态每臂分母均为 36 世界，其中只有 30 个不同图像张量、24 个基础组，不能当作 36 次独立确认。背景配对与三初态不是额外独立样本。拒绝仍以 KEEP 的误差计入。

该图区分尺度回退和动作选择的作用：回退改善偏移修复，但 P/U 仍保留正确输入损伤；M/R 在这批已暴露数据上通过接口检查。这是开发回放，不是独立迁移成绩。

## 2. w002：评分到动作的信息丢失

文件：`02_w002_score_to_action.png`、`02_w002_score_to_action.svg`。

仅展示一例已暴露的旧世界。左图为完整封存评分在 450–900 mm 可见域内的曲线，纵轴是灰度预测 MSE（灰度单位平方），空心圆标出五个动作候选。600 mm 是这五个候选中的最低评分者；不把它改称任意连续深度的严格全局最优。P 支持区间为 [486,782] mm，按长度均匀均值得 634 mm，选中已有候选 660 mm。右图显示原本正确的 600 mm 现任被 P 移坏 60 mm，M 则 KEEP。

这幅图只解释已知失败的决策接口，不增加独立样本，也不证明 M 在所有情形优于其他规则。

## 3. 确认的三初态与正确输入损伤

文件：`03_confirmation_endpoints.png`、`03_confirmation_endpoints.svg`。

比较 ED+S1+M、EF+S1+M、ED+S1+R、full9+M、full9+P、KEEP。每臂每初态为 48 个独立合成场景；三个初态来自同一场景，不能按 144 个独立样本计数。MAE 与损伤口径同图 1。各面板单独使用线性横轴，并直接标注数值；不跨面板比较柱长。这里显示描述性点估计，不显示或暗示置信区间。

主候选 ED+M 的偏移 MAE 与 full9+M 接近，但正确初态仍有 1/48 个超过 1e−9 mm 的严格损伤（正确初态 MAE 约 0.01524 mm），注册的严格保真门槛因此失败。该损伤未超过 7.5 mm 的实用阈值，不能用后者替代前者。封存数据中正确输入严格损伤数：ED+M=1、EF+M=10、ED+R=9、full9+M=0、full9+P=9、KEEP=0。

## 复现与软件致谢

在尚无同名图和 PROVENANCE 文件的输出目录中，运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B analysis/make_figures.py
```

脚本拒绝覆盖已有图；要重新制作不同版本，应另行设定新的输出位置，不删除或覆盖封存结果。

绘图采用 `kdense-matplotlib` 技能提供的面向对象绘图、可访问性和静态导出规范。软件资料引用：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [DOI: 10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。已于本次制作核对 arXiv 记录（最新修订 2026-09-02）；引用不固定旧版本后缀。
