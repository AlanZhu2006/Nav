# Table II 新双角色分支：HPC 编排与提交准备

日期：2026-09-11。此文件记录新 Table II 工作，不替换旧论文结果。

## 已完成的改动

1. 新增独立 HPC 编排 `table2_mixed_hpc.py`，读取冻结源清单，统一使用 scene/episode 身份。
   不直接扩展按 scene 索引的本机 pilot；同场景多个 episode 不会覆盖或混配。
2. A、B 配对、C 配对分任务运行；每个查询的 native/GEM 在同一 GPU、同一对 server 内完成。
   两臂先后顺序在每种角色内部按源序交替，而非只在交错数组上看似平衡。
3. B 数组完成后全局选择两种成功 native-B 来源，各取相同数量。
   测试明确覆盖「A0 只成功 Novel-B、A1 只成功 Revisit-B」仍能组成 50/50 的情形。
4. 新增 `table2_task_storage.py`：各 job 使用独立 node-local 目录，容器绑定同一实验虚拟根。
   上游归档按原路径恢复，原收据/图像/深度字节不改写；不将历史散帧堆到 scratch。
5. 新增新入口 CPU 预检、独立任务 verifier、逐阶段汇总、源码打包及四个 Slurm/启动脚本。
   打包显式包含此前遗漏的本机 Table II 协议及本次 HPC 协议。

## 控制器与方法没有变化

已核对 10 个核心控制/runtime 文件的 SHA 与完成的本机实际 A/B/C 记录一致：
bounded pure pursuit、标准 try_step 碰撞、RGB/深度栅格修复、后向目标真实转身再规划均保留。
没有换成 MPC，没有恢复旧 snap 落点执行，没有修改 GEM 阈值、canonical depth 或 NavDP 权重。
本轮只修改编排、存储、测试和文档。源构造继续使用已修复早期帧截断问题的本机版本。

## 小批范围

- HM3D 已有 `core_source_plan.json` 前 4 scenes，每 scene 前 2 个源载体，共 8 个新实际 A。
- 复用的是首位姿/相机参数和场景资产；不是旧 A 结果、expert history 或旧目标图。
- 新 A 使用独立初始朝向和共同的目标照片规则。
- A 最多 8 次；B 最多 16 个配对查询；C 最多 32 个配对查询。
- C 数量取决于全局 native-B 成功与双角色可构造性，供给不足如实记 0，不填入旧前缀。
- 明确为 integration/supply pilot；距离/朝向正式配额尚未冻结，不能作为正式新 Table II。

## 已验证

- 本机 91 项回归测试通过，包括新编排、历史隔离、全局混合、归档恢复及旧执行器契约。
- 四份 shell/SBATCH 语法检查通过。
- 源码包 3,652,479 bytes，内部含 1,756 个基础文件及新增协议/脚本。
- 包 SHA：`1284d6bf5709f07a2a54213dc7dcbc4232c735a6b0c043746da8b5de561b5d5c`。
- bundle receipt SHA：`eae5fd9d03e447009d455e3aabadd66b33c5207eafa916745d0600a0cf031fe9`。
- plan SHA：`e745b30588a47e2ac92cb8e51c3ba7459a49341690c2172d7b27de02b293bbc5`。
- 通过正确的 yz11502 共享连接上传，两端上述摘要一致。
- 登录节点 quota 快照：scratch 1.42 TB/5 TB，文件 3,535,486/5,000,000；不是无配额限制。

## 运行路径与当前状态

本机准备目录：`.diagnostics/table2_hpc_prepare_20260911_88TZOq`。

远端源码：
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table2_mixed_eae5fd9d03e44700`

远端结果：
`/scratch/yz11502/Research/Nav-axis-uturn-results/table2_mixed_pilot_20260911`

**2026-09-11 07:35（北京时间）：已提交，当前尚无新导航 SR。**

| 作业 | 内容 | 提交后回读状态 |
|---|---|---|
| `17347454_0` | 第一个实际 A 完整链路 | PENDING，`QOSGrpGRES` |
| `17347455_[1-7]` | 剩余 7 个 A | 等首个 A 任务成功退出 |
| `17347456_[0-15]` | B 双角色 native/GEM 配对 | 等全部 A 任务完成 |
| `17347457` | 全局等额 C 选择，并据此提交 C 配对与汇总 | 等全部 B 任务完成 |

`QOSGrpGRES` 表示项目/QOS 的 GPU 总额度暂时占满，不是脚本报错。所有 GPU 作业
已回读确认为 A100、1 GPU、10 CPU、96 GB、每任务 01:00:00，并发上限 2。
C 不提前申请空数组：全局选择写入后，按实际 C 查询数提交；零供给则仅汇总缺额。

远端 exact-container 预检已全部通过：新入口 5 种 CLI 组合、8 个源载体、
MemNav/NavDP 导入来源、Habitat/cv2/pyarrow 依赖、三个 checkpoint 及既有依赖清单。
三个 checkpoint 的 SHA 与已验证版本一致。两组依赖清单检查退出码均为 0。
本机额外使用真实已保存 C 查询文件执行 native/GEM `--contract_dry_run`，均通过且无导航输出。

远端收据：`submission.json`、`preflight/table2/verification.json`、
`preflight/environment_verification.json`、`preflight/checkpoint_hashes.txt`。
本机临时 HTTP server（PID 2936954）及 localhost-only 反向转发已关闭；共享 SSH master 未关闭。

接下来核验的是首个 GPU 实际 A 任务及跨节点归档恢复，不把 CPU 预检或排队状态写作闭环成功。
本轮没有训练、没有论文数字更新、没有操作机器人，也没有 commit/push。

## 不混入本轮的旧任务问题

只读查询发现旧 Table I exact retry `17336842_110`、`17336842_158` 均 FAILED，
对应 summary `17336843` 已取消。已读取 110 内部日志，仍是
`MemNav planning append received different JPEG bytes`，不能称为已修好或算作方法失败。
本轮未改该旧任务、未重交它，也没有为了启动新 Table II 删除同样的输入一致性检查。
新 pilot 的接口放行与实际结果仍须分别核验；旧问题可能再次触发时应保留现场处理。

## 08:05 更新：首条实际 A 完成，同步传输修复后继续

`17347454_0` 已 COMPLETED，完整任务耗时 **3 分 44 秒**。首条源
`rJhMRvNn4DS/episode_0000` 的实际 mono-native A 成功，执行 **157 步**。
已经从该实际历史构造 B-Novel 和 B-Revisit；单项 verifier 为 true，归档完整回读。
这只是首条采集成功，尚无 B/C 配对结果，不能报告总体 SR。

其 A0 归档不变：`A/task_000/artifacts.tar.gz`，40,900,237 bytes，SHA：
`e102ccfa316a2673445e0937d30fded094fc5657b5cfef11cd9aa591c9fa6686`。

Table I 两项 JPEG 故障已定位为 multipart 分块解析额外 CR，详见
`TABLE1_MULTIPART_REPAIR_STATUS_20260911.md`。同一修复同步到 Table II 私有服务，
防止尚未启动的 A/B/C 再遇到已知传输问题。

核实旧 `17347455/17347456/17347457` 均 PENDING、0 秒后，仅撤回这些未启动任务；
没有取消 A0，没有覆盖任何导航输出。替代链为：

| 作业 | 内容 | 提交后状态 |
|---|---|---|
| `17348182_[1-7]` | 原清单剩余 7 条实际 A | PENDING / QOSGrpGRES |
| `17348183_[0-15]` | B 双角色配对 | 等待 A 数组 |
| `17348184` | 全局 50/50 C 选择，再提交 C 与汇总 | 等待 B 数组 |

仍为 A100、每项 1 h、并发 2；没有新增源、换目标或放宽 GEM 条件。
scientific plan 的 SHA 仍为
`e745b30588a47e2ac92cb8e51c3ba7459a49341690c2172d7b27de02b293bbc5`。

新 runtime：
`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/table2_multipart_8b559e821e8c9df6`

新 runtime SHA：
`8b559e821e8c9df6b0a91c8ee80abeb22fcdd1748f40a37a4ce125a3cc5094de`

除了三个 server 的解析修复挂接，Table II 的两处记账入口分别记录
`planned_runtime_sha256` 和实际 `runtime_sha256`，并验证修复 manifest。
不能把新 runtime 伪装成旧 SHA；也不改写已完成 A0 的计划或历史文件。
控制器、GEM、构造器、臂顺序和成功判定均不变。

HPC 新 runtime 的 8 源、24 文件、五种 CLI 预检再次通过，收据位于：
`preflight/multipart_8b559e821e8c9df6/verification.json`。

续跑收据在原结果根下：
`multipart_continuation_8b559e821e8c9df6/submission.json`、`preserved_state.json`。
提交脚本：`MemNavData/submit_table2_multipart_continuation.sh`。
本轮本机完整相关测试 **233 passed**，没有修改论文或机器人，没有 commit/push。
