# 本机 learned anchor 定位：恢复运行与完整结果

更新时间：2026-09-07 02:52（Asia/Shanghai）。**原定两臂、三种子训练已完成，独立复算通过；不是导航 SR，没有替换生产 CEC。**

协议：[ANCHOR_RELATION_GEOMETRY_PROTOCOL_20260906.md](ANCHOR_RELATION_GEOMETRY_PROTOCOL_20260906.md)。
历史进度：[ANCHOR_RELATION_GEOMETRY_PROGRESS_20260906.md](ANCHOR_RELATION_GEOMETRY_PROGRESS_20260906.md)。

## 1. 恢复了什么

用户要求重新启动此前暂停的本机 learned 实验。本次恢复固定支持 anchor 的关系定位对照，
不恢复旧候选排序、proof head 或 NavDP 训练。

- 数据、划分不变：原 train40 内，103 train pairs / 32 scenes，20 validation pairs / 8 scenes。
- 输入仍为冻结 DINO patch；一臂额外读取真实因果历史 LingBot XYZ。
- 两臂共用 mask、历史深度归一化和监督，模型宽度、loss、学习率及训练预算未变。
- 每臂 seed 11/23/37，各 1,200 steps；没有选择最好种子或按验证集提前停止。
- 复用已经完成的 74 段历史几何以及 8-pair overfit 结果，没有重复提取历史或重复小训练。

## 2. 暂停点与修复

旧 workflow 在 `verify_overfit8` 退出，不是仍有训练进程被挂起。
报错来自保存预测的数值编码：训练报告对原 float32 预测进行 float64 统计；旧 CSV
直接输出 NumPy float32 的最短往返小数，独立脚本却直接把这些小数当作 float64 预测。

在原始预测文件上逐组重算：直接解析的最大汇总差约 `3.96e-8 m`；先恢复原 float32
再独立计算后，各组差异不超过 `2.22e-16 m`。因此没有调整 `abs_tol=1e-8`，没有修改
旧实验数值或报告来通过检查。

- 旧格式读取：只为 learned float32 预测恢复原存储类型，reference 的 float64 保持不变。
- 新格式写出：先转换成 Python float，保留 float32 原值的完整往返表示；manifest 记录编码。
- 运行器增加 `--resume-after-overfit`，在新 workflow 目录接续，旧失败日志完整保留。
- 模型、损失、采样、几何提取和冻结协议的内容没有改变。

旧 overfit 预测 SHA256 保持：
`9855c6717c1f3747f2326dc4a25647acb5650f6d79b20c6ac132739231abf463`。

## 3. 全部内部验证结果

下表误差单位为米；每个 learned 条件均为相同 20 pairs。三种子不是三个独立验证集。

| 输入 | seed | train 平均位置误差 | validation 平均位置误差 | validation scene-macro | 位置误差 ≤0.5 m |
|---|---:|---:|---:|---:|---:|
| 视觉，共同归一化 | 11 | 0.1388 | 1.1171 | 1.2825 | 6/20 |
| 视觉 + 历史 XYZ | 11 | 0.1038 | 1.1401 | 1.2649 | 5/20 |
| 视觉，共同归一化 | 23 | 0.0953 | 1.0466 | 1.0547 | 5/20 |
| 视觉 + 历史 XYZ | 23 | 0.1434 | 0.9861 | 1.0374 | 6/20 |
| 视觉，共同归一化 | 37 | 0.0877 | 1.0475 | 1.1962 | 5/20 |
| 视觉 + 历史 XYZ | 37 | 0.1196 | 0.9926 | 1.1284 | 6/20 |

validation 平均位置误差的三种子平均：视觉 `1.0704 m`，视觉 + XYZ `1.0396 m`。
同数据参考输出：旧 frozen GCT 为 `0.4189 m`（20/20 有效，17/20 在 0.5 m 内）；
旧 finite-PnP 为 `0.2898 m`（仅 19/20 有效，16/20 在 0.5 m 内）。PnP 均值有缺失分母，
不把它当作 20/20 完整预测的均值，也不把这些旧参考输出称为新同机闭环。

结论：这版 XYZ 输入的平均改善较小，pair-macro 并非每个种子都改善；训练与未见场景
验证仍存在明显差距。没有达到协议中据此扩充训练或替换生产定位的证据要求。
这个结果不否定所有 learned relocalizer，但不支持仅延长当前小模型训练就能替代 CEC。

## 4. 完成与输出

- 原 overfit 独立复算：12 groups，`verified=true`。
- 本次完整对照独立复算：20 groups，`verified=true`。
- 恢复链总耗时约 25.0 秒；只包括复用特征后的训练与验证，不包括此前的几何提取。
- 没有提交 HPC 作业，没有重跑导航，没有改变真机服务或发布运动指令。
- 语法检查及 `git diff --check` 通过；本轮没有运行机器人测试，也没有 commit/push。

输出根：`.diagnostics/anchor_relation_geometry_20260906/`

- `workflow_resume_20260906T185058Z/launch_receipt.json`：本次源码与启动收据（目录时间为 UTC）。
- `workflow_resume_20260906T185058Z/completion.json`：完整完成标志。
- `scene32_8_geometry_v1/manifest.json`：不变的训练划分、种子和预算。
- `scene32_8_geometry_v1/report.json`、`predictions.csv`、`training_curve.csv`：全部结果。
- `scene32_8_geometry_v1/independent_verification.json`：独立重读预测与分组指标复算。
- `scene32_8_geometry_v1/*_seed*.pt`：六份 checkpoint，均未授权部署。
