# Table I 失败项补跑：提交记录（2026-09-11）

截至北京时间 2026-09-11 04:47，**两项精确补跑已提交，尚未运行**。
目前解决的是补跑提交与结果汇总流程；JPEG 摘要异常的根因尚未确定，不能写成已经修复。

## 1. 实际提交与实时状态

| 任务 | Job ID | 此次读取状态 | 内容 |
|---|---|---|---|
| 两项 GPU 补跑 | `17336842`，indices `110,158` | PENDING / QOSGrpGRES | MP3D/NavDP history26、MP3D/ViNT history32 |
| 新完整汇总 | `17336843` | PENDING / Dependency | 等待原数组终止与补跑成功后，汇总全部 210 个单元 |
| 原数组尾项 | `17306820_206` | RUNNING / ga034 | 保持原运行，不取消、不重复提交 |
| 原数组剩余 | `17306820_[207-209]` | PENDING / QOSGrpGRES | 保持原队列 |
| 原汇总 | `17306821` | PENDING / Dependency | 保留原始行为和输出，不用改写原失败记录 |

原批次共 210 个单元，此时 Slurm 已完成 204 个，1 个运行、3 个排队、2 个失败。
这是作业状态，不替代最终完整 verifier。两个失败单元的前三臂不作为完整配对结果纳入。

补跑每项使用 A100、1 GPU、10 CPU、96 GB、1 小时时限，最多并行 2 项。
`QOSGrpGRES` 是当前 GPU 配额等待状态；测试提交器给出的预计启动时间不是承诺。
CPU 汇总为 `cpu_short`、2 CPU、8 GB、30 分钟。

## 2. 故障与处理边界

- `110`：最后一个 Revisit-native 臂的 MemNav planning append JPEG 摘要不一致。
- `158`：最后一个 Revisit-native 臂的目标 JPEG 摘要不一致；服务端解码像素摘要却与之前一致。
- 两项都在 ga033，退出码均为 `1:0`；现有日志没有证明是硬件故障，也不能唯一定位编码、传输或摘要计算问题。
- 保留图像字节、帧、深度 transaction、目标消费检查，不通过删除断言或容忍不一致来放行。
- 完整重跑各自四臂，即共 8 条导航；不把原前三臂与另一进程的新第四臂拼接。
- 权重、目标、seed、臂顺序、GEM 阈值、2.5 m residual、600 ticks、horizon 8 和评分方式均不变。

首次尝试排除 ga033，但 `sbatch --test-only` 拒绝了 `--exclude=ga033`。
该次目录已创建，但未产生 GPU/汇总提交；保留为未提交尝试。正式提交取消节点排除，
走相同 A100 分区的正常调度，不绕过集群限制，也不保证实际分配到哪个节点。

若同样异常再次发生，仍须保存失败记录并调查，不能将本轮原样补跑描述为已完成根因修复。

## 3. 验证与结果保留

已经完成：

1. 按共享 SSH 手册复用 `alantorch` 的 `yz11502` 连接；未更改共享 master。
2. 核验原冻结源码清单、plan SHA、原失败状态和两份失败归档的完整 SHA。
3. `bash -n` 通过；本地原有 Table I 汇总/配对与 image-controller 接口测试 **8 passed**。
4. GPU、CPU 两种正式配置的 Slurm `--test-only` 均通过；提交后回读分区、时限、数组和依赖。
5. 验证合并视图的 **210 个符号链接**：仅 110/158 指向新输出，其余 208 项指向原目录。

这些检查保证本次提交范围及配对/汇总逻辑，没有验证尚未发生的 GPU 补跑结果。
没有修改导航 runtime、checkpoint、论文或已完成实验；没有取消原任务，没有删除原失败归档。

新汇总依赖：`afterany:17306820,afterok:17336842`。
最终仍由原汇总器检查全部单项 verifier、归档 SHA 和 210/210 完整性，再生成 SR/SPL。
原汇总可能因原失败项而报告 incomplete；本次应读取下面的合并汇总，不能混淆两个路径。

## 4. 精确路径与版本

原 runtime（原样使用）：
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table1_repaired_7b6bffa77891ef4f`

- runtime 清单 SHA：`7b6bffa77891ef4f041508dc7dc31b40fb56b7e1885334a9e180a1f06d76e099`
- plan SHA：`0964f52738e22f2d16b099196605ef2e9681f57afce20fbdb99b417d8b535622`

新增提交器封存目录：
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table1_exact_retry_20260911_7ed4b1d772b44429`

- 提交器 SHA：`7ed4b1d772b44429a0ac5fe651fee379fd4d6f2669140f7414b59267521825bf`
- 协议 SHA：`77b705eabac58ecff3efb23abdfbf4dc1df084041f14bb0d50ee06fadc7476f2`

补跑根目录：
`/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/exact_retry_20260911_7ed4b1d772b44429`

最终合并汇总（尚未生成）：
`/scratch/yz11502/Research/Nav-axis-uturn-results/table1_repaired_20260910/exact_retry_20260911_7ed4b1d772b44429/combined_evaluation/paired_summary.json`

该根目录的 `recovery_manifest.json`、`retry_submission.json`、`submission.json`
分别记录原失败证据、GPU 实际提交和汇总依赖。

首次未提交尝试的后缀为 `exact_retry_20260911_58cc3795ac66c935`，不能把其中的节点排除设置
误读为本次实际 Slurm 参数。它没有 `retry_submission.json` 或 `submission.json`。

详细诊断与实验不变量见 [补跑协议](TABLE1_EXACT_RETRY_PROTOCOL_20260911.md)。
提交器见 [submit_table1_exact_retry_20260911.sh](submit_table1_exact_retry_20260911.sh)。
