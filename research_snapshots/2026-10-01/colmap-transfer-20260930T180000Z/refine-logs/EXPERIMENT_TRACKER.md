# 实验登记

| 阶段 | 状态 | 证据 |
|---|---|---|
| 场景、ROI、像素与方法冻结 | DONE | SCENE_SELECTION、PLAN、RUN_LOCK |
| 相机与CPU契约 | DONE | CAMERA_CHECKS、测试命令、CPU SUMMARY |
| 官方photo/geo推理 | DONE | 14 job_records、MVS_RUN.log |
| 输出先封存后读取GT | DONE | PREDICTIONS_SEALED、references/MANIFEST |
| 固定分母评价 | DONE | 2560点行、40ROI行、8尾部行 |
| 新代理独立复算 | DONE | review/INDEPENDENT_NUMERIC_CHECK.json |
| 限定结论与负结果 | DONE | REPORT、review/INTEGRITY_FINAL |
| GitBook/GitHub发布 | 单独核验 | 外部发布清单与远端commit；本登记不是上传成功凭证 |

主判定：两个新物体geo_fallback等权MSE均优于CPU，达到预定有限迁移条件。部署默认不变。
