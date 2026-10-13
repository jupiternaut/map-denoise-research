# A4-V 实验结果记录

2026-10-12 19:22:11 Asia/Shanghai。目标：保留有效几何并修复不足。  
方法：同18来源前向外观干预，两档η；strict/independent/bounded三臂，cal/joint，其他约束、预算与决策冻结。  
结果：主40无bounded增量；全144非零joint多修复1例、另2例进一步改善，MSE降低0.9180%。bounded合法覆盖18/18且72外包50收窄，但主错误初态cal40/joint39仍在外包凸包内。  
解释：不能把已测收紧说成无操作；当前盒外包尚不足以认证更多移动。  
下一候选：联合可行性检查与有理不可行证书，未执行。  
报告：REPORT.md；独立核验：outputs/variation-v1/verification/VERIFY.json PASS。  
科学主终点不成立不导致删结果，机械PASS与科学增量分开。  
源和输入不改，正式进程已结束。

