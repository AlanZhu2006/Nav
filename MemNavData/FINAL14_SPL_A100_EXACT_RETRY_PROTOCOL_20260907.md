# Final14 SPL：A100 原配置补跑

冻结日期：2026-09-07。用户授权：优先解决运行问题并补齐失败任务，不另开根因研究。

## 范围

本次选择的是运行规避方案，不声称已证明 H100 故障或 CUDA 同步根因。
截至准备时，本轮 A100 的 4 个完整 history 均已成功；失败的
`6,7,8,11,12,14` 均为 H100 上的 evaluator SIGABRT，尚无完整配对结果。

- 将这 6 个索引使用**相同 replay bundle**在 `a100_tandon` 补跑；每个 history
  仍在同一张 GPU、同一对持续存活的 server 上运行全部五臂、10 条 rollout。
- 不改环境、线程、CUDA 同步开关、权重、目标、历史、顺序、seed、600 步预算、
  到达阈值、SPL 测量方式或分析规则。唯一评测运行变更是限定到原允许池内的 A100。
- 1 GPU、10 CPU、72G、每元素 1 小时，最大并发 2。
- 已完成 history 保持原样；失败的 task/evaluation 目录整体移入
  `repairs/a100_exact_retry1_20260907/failed_attempts/`，保存 SHA，不删除、不覆盖。
- 只在原位置生成对应失败索引的新完整结果，不将失败尝试中的部分臂拼入新结果。
- 原 array 中尚未开始的 `18,19,20` 也转到 A100。集群已拒绝原地 partition
  修改（`Unspecified error`），因此仅在再次确认 PENDING、尚无输出后，取消这三个
  未启动项并与失败项一同提交；这不是重复评测。运行中的 index 16 不打断。
- 新 summary 等待原 array 结束及本次补跑成功，再使用原独立 verifier 核查全部
  21 histories / 210 rollouts。若还有未修复的失败，summary 必须报缺失，不能缩分母。

源代码仍为：

`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_spl_replay_50eb1204020a4d7c`

其 source receipt SHA-256：
`50eb1204020a4d7c38e34269cfb024989a1fee0fac6e9848792619c2472e36d5`。

结果根目录仍为：

`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_table3_exact_spl_20260907`

本地原有 SPL 测量测试 8 项通过，启动脚本语法检查通过。实际提交 job ID 另记于
提交收据；本协议不等同于已提交或修复完成。论文继续保留旧 SPL bounds，直到本轮
全部验证通过后一起替换本轮 SR/SPL。

## 11:23 执行补充

原 index 16 随后在 gh013 运行 17分53秒后以相同 `evaluator failed (-6)` 失败；
没有中止其运行。其 9 个文件已另存于 `repairs/a100_exact_retry_task16_20260907/`，
并以相同 frozen sbatch 单独补交 A100 job `17089989_16`。

当前队列覆盖全部未完成的 10 个 history：原 6 个失败加 3 个未启动项在
`17089916`（array 最大并发 2），新失败的 16 在单卡 job `17089989`。
已通过的 11 条不变。最新全量核验为 `17089991`，同时依赖两个 A100 job 成功；
此前尚未执行的 summary `17057433`、`17089917` 均已取消。

11:23 时两个 A100 job 均为 `PENDING (QOSGrpGRES)`；尚无补跑完成结果。
