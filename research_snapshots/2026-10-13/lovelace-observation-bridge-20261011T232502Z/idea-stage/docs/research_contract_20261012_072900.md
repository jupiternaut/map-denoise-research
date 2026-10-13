# Research Contract：观测桥接

Selected Idea：执行上一批已审阅A+B；不修改旧solver，只改变图像处理、公开误差界和传播轮数。
Core Claims：C1配对测量裁剪/宽界/偏差效应，C2测传播余量；两者允许零效应，不追逐预设改善值。
Method Summary：八配置shared/KEEP，同54基础观测、全初态、历史两个共享臂明确只读复用。
Experiment Design：六新共享配置2916动作，972复用共享动作，3888新KEEP动作，合计7776；先保存后私有评价。
Current Results：尚未运行。
Key Decisions：0GPU/API/安装，无发布/部署。总运行核验窗口5400秒；全部历史只读。
Status：协议已写，待两作者实现、测试、非作者审查、实际运行与独立核验。
