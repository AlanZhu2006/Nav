# Final14 精确 SPL 补跑：运行时崩溃诊断

核查日期：2026-09-07，北京时间约 10:22；10:35 补充实际 sbatch 提交记录对照。范围：已保存日志、冻结代码、Slurm 记账和历史诊断的只读复核，仅更新本文核查记录。本轮未更改正式作业、模型、参数或产物，未重提任务；真机常驻服务未被调用或重置。

## 当前结论

已确认发生的是 evaluator 子进程在查询执行期间的 `SIGABRT`，而不是 SPL 计算断言失败、导航未到达或作业达到时限。最值得验证的假设是模型 CUDA 工作与 Habitat/EGL 渲染交接的运行时问题；**尚未在本次失败 history 上完成同卡受控对照，不能认定根因已闭合，也不能保证 HTTP 同步开关单独有效。**

## 1. 本次现场证据

实验根目录：`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_table3_exact_spl_20260907`。

| History | Array task | 节点 | 退出时间 | 首个失败臂 |
|---|---|---|---|---|
| 6 | `17057432_6` | gh013 | 17分56秒 | zero native |
| 7 | `17057432_7` | gh005 | 19分28秒 | mono CEC |
| 8 | `17057432_8` | gh013 | 17分40秒 | metric native |
| 11 | `17057432_11` | gh013 | 17分56秒 | zero native |

- 6/7/8/11 的父进程均明确记录 `evaluator failed (-6)`；运行期间两个 server 仍有 HTTP 200 响应。
- 崩溃前连续 `memory_step` 完成日志相隔约 19 秒。这是调用之间的墙钟间隔，不等同于 LingBot forward 耗时。
- 正常 history 0 和原实验 history 6 的相同日志通常一秒内有多次更新。
- 截至本次扫描，`invalid value encountered in cast` 出现在 6/7/8/11 的首臂日志。对应冻结 evaluator 的 `depth_png_bytes` 将渲染深度转为 uint16；警告本身不能证明非有限深度的来源，更不能单独证明它导致 abort。
- 6/7/8 的 MaxRSS 分别为 `14921084K / 15077068K / 9943108K`，申请内存为 72G；运行时间低于 1 小时。Slurm 记录为 FAILED，而非 TIMEOUT 或 OUT_OF_MEMORY。
- 6/7/8/11 尚无完整 query CSV、plan JSON 或 terminal measurement，因而本次用于末步积分的回调没有留下成功收据，不能把它误称为 verifier 拒绝。
- 6/8 使用相同 GPU UUID；7 与同节点成功的 10 使用不同 GPU UUID。11 与 6/8 同节点、不同 GPU。因此不能仅凭节点名称得出“该节点必坏”或“所有 H100 不可用”。

截止快照：9 个 history 已独立核验通过，4 个失败，8 个仍待执行；没有完整新 SR/SPL。

## 2. 是否沿用原实验

远端 manifest 实算 SHA：`7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a`。
原冻结 source receipt：`5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216`。
重新对照 receipt 检查的 evaluator、NavDP server、MemNav server/policy agent、factorial runner/contract 均通过。

同一 population、历史、目标、600-step budget、1 m success radius、exec horizon 8、deterministic plan seed 和 role-hidden 合约保持一致。

但这不是原作业逐字重放：原 2026-08-19 五臂包含 `mono_raw_fixed`，2026-08-28 单独补了 zero-depth。本轮统一运行 Table III 的五行，用 `zero_native` 替换不在该表的 raw-memory 臂；增加末步测量包装，并把 buffer 改为节点临时盘、退出时归档。线程环境明确设置 OMP/OpenBLAS=4。以上差异必须保留在复现说明中。**时限不是实际差异：旧模板虽然写 2 小时，最终提交命令已覆盖为 1 小时，与本次相同。**

原实验和本次日志均显示 `cuda_http_handoff_sync=False`。本次并非新关闭这个开关；但“继承原配置”也没有覆盖后来手册提出的运行时同步检查。

### 2.1 实际 sbatch 与最终执行批次复核（10:35 补充）

不能只看最初 submission receipt 或模板默认值。通过最终 21 个 `server_receipt.json` 中的实际 job ID 回查 `sacct`：

- `16020578 / 16020580 / 16020581` 均在启动前取消，不是产出论文结果的作业。
- 最终 smoke 为 `16020635`，formal array 为 `16020636_[0-20%10]`，实际 SubmitLine 明确带 `--time=01:00:00`。
- 原 array 的 19 个 history 正常完成；indices 2、4 在创建实验输出前被系统取消，状态为 `CANCELLED by 0`，不是本次 evaluator SIGABRT。
- 两个缺失 history 由 `16026422_[2,4]` 使用同一 frozen bundle 补齐；最终 21 条均完成，独立 verifier 为 `verified=true`。
- 最终成功作业分布：15 个 A100、6 个 H100。单 history 耗时约 17分33秒至38分35秒。
- 扫描最终 105 个 `eval_*.log`，没有匹配到 `invalid value encountered`、`Aborted` 或 `SIGABRT`。21 个 MemNav server 日志全部为 `cuda_http_handoff_sync=False`。

| 项目 | 最终执行的旧 Final14 | 本次 SPL replay |
|---|---|---|
| GPU / CPU / RAM | 1 GPU / 10 CPU / 72G | 相同 |
| GPU 分区 | H100/A100 | 相同；具体分配的卡不同 |
| 实际时限 | 1 小时（提交参数覆盖模板） | 1 小时 |
| array 最大并发 | 10 | 4 |
| 容器和挂载 | 同一 CUDA 12.8.1 SIF、`--nv`、只读 PT1 overlay、两个 scratch bind | 相同路径与挂载方式 |
| Python 环境 | 同一 Habitat / MemNav 环境路径 | 相同路径；不能据此声称外部环境逐字节未变 |
| 进程布局 | 一个常驻 MemNav、一个常驻 NavDP；逐臂 Habitat evaluator | 相同布局 |
| HTTP CUDA 同步 | 未打开 | 未打开 |
| OMP / OpenBLAS | 脚本不显式指定；提交继承环境未完整保存 | 显式指定为 4 |
| RGB/depth buffer | scratch 下的 task buffer | 节点临时盘，退出时归档 |
| 启动与端口 | 原启动脚本、算术端口预检查 | 新包装、flock 端口对、额外退出归档 |
| evaluator 入口 | 原 role-pair evaluator | 末步测量包装后调用原 evaluator |

因此用户关于“旧 Final14 没有这种连续崩溃”的记忆有原始记录支持。项目其他批次曾出现 Habitat abort（见 `SEMANTIC_PROPOSAL_GATE_B_TASK18_INCIDENT_20260815.md` 和 HM3D ViNT exact-retry 记录），但不能将那些事故称为这批旧 Final14 的事故，更不能认定根因相同。

目前未发现某一个 sbatch 参数足以直接解释本次 SIGABRT。后续隔离诊断应把**原启动方式与新 SPL 包装的差异**也纳入区分；不能只比较同步开关，或把“同一源代码”直接等同于“同一完整运行环境”。

实际提交与修复的本地收据：

- `FINAL14_MONO_FACTORIAL_ATTEMPT6_SCHEDULER_AMENDMENT_20260819.json`
- `FINAL14_MONO_FACTORIAL_ATTEMPT6_MISSING_TASK_REPAIR_20260820.json`

远端 frozen sbatch、原启动脚本和 replay sbatch 的 SHA 与本机对应文件一致。本轮没有更改这些脚本或作业。

## 3. 重新核查手册引用的历史诊断

下列数值直接从原始 `runtime_timing` JSONL 重新汇总，而非抄写手册：

| 历史条件 | Job / GPU节点 | render 次数 | render 中位耗时 | memory HTTP 中位耗时 |
|---|---|---:|---:|---:|
| 无 Pi3X | 15891498 / gh128，H200 | 32 | 5.77 ms | 0.124 s |
| 有 Pi3X | 15891503 / gh134，H200 | 32 | 19.186 s | 0.125 s |
| 有 Pi3X + launch blocking | 15892310 / gh123，H200 | 4 | 5.88 ms | 0.186 s |

纯 Habitat 渲染 job 15890359 的 16 次渲染中位为 8.57 ms。

重要纠正：job 15892310 的真实 SubmitLine 设置的是 `MEMNAV_CUDA_LAUNCH_BLOCKING=1`；其 server launcher 的实际命令**没有** `--synchronize_cuda_http_handoff`。此外它把每个 query 的诊断步数从 16 改成 2，并换了 GPU。历史结果支持“异步 CUDA/渲染交接值得排查”，但不是 HTTP 同步开关单因素、同卡确认，也不是本次 H100 SIGABRT 的直接因果证明。

原始路径：

- `diagnostics/final14_geometry_lifecycle_20260817T1040Z_attempt2/{without_pi3x,with_pi3x}/logs/eval_stderr.log`
- `diagnostics/final14_geometry_lifecycle_sync_20260817T1055Z/logs/eval_stderr.log`
- `diagnostics/final14_geometry_lifecycle_sync_20260817T1055Z/logs/memnav_launcher.json`
- `diagnostics/final14_habitat_render_20260817T100816Z/h200_tandon.jsonl`

上述相对路径均以 `/scratch/yz11502/Research/Nav-axis-uturn-results/` 为根。

## 4. 此前提出但未执行的诊断计划

以下是 10:22 时的建议，未提交。用户随后明确要求优先补齐实验，不必另做完整根因分析；当前执行安排见第 5 节，不再等待本项诊断授权。原建议是在最多 20 分钟、单 GPU 的隔离作业中，使用已经失败的同一 history、原冻结权重与输入，在**同一分配的 GPU**上顺序检查：

1. 原异步执行方式；
2. 只打开 HTTP response 前的 CUDA 同步；
3. 如预算允许，只打开 CUDA launch blocking，作为交接问题的区分对照。

各条件使用相同诊断步数、相同 reset/seed、独立 server state、相同已冻结历史；不与正式 SR 混算。复用 `diagnose_role_pair_runtime.py` 记录 render、memory HTTP、planning HTTP、pursuit 和 geodesic 的分项耗时，启用 faulthandler。另记录渲染深度非有限数量及 GPU UUID/驱动；不自动替换坏值或回退策略。

必须先看到异步条件的异常被复现，才能用开关对照谈修复。如果原条件也正常，该次诊断只能记为“未复现”，不能宣布根因已解决。若同步只修复停顿但仍 abort，应继续区分两种故障。

正式补跑仍需另行确定修复版本和失败索引，保留已完成块与失败日志，不覆盖 frozen bundle，不把 partial rollout 作为 SR 失败或成功计入。

## 5. 已执行：A100 原配置补跑（11:18 更新）

用户授权后选择最小运行规避方案：保持本轮已经有完整成功记录的 replay bundle 原样，
只将补跑限定在 A100，不打开新同步开关、不改测量或导航方法。

- 11 个 history 已完成且逐项 verifier 为 true；未重跑或改写这些结果。
- 6 个失败索引 `6,7,8,11,12,14` 整条五臂补跑；其 54 个原始文件共
  90,938,217 bytes 已整体归档并复核 SHA，没有删除。
- 尚未执行的 `18,19,20` 因 Slurm 拒绝原地改分区，仅取消未启动项后转交同一 A100
  array；原 index 16 继续运行，未打断。
- 补跑 array `17089916`：`6,7,8,11,12,14,18,19,20%2`，A100，单卡每元素 1 小时。
- 新全量 verifier `17089917` 等待原 array 结束及 A100 array 成功；原尚未执行的
  summary `17057433` 已取消。核验仍要求 21 histories / 210 rollouts，不降低分母。
- 11:18 时补跑处于 `PENDING (QOSGrpGRES)`，尚无完整新 SR/SPL，也不宣称故障已根治。

详见 `FINAL14_SPL_A100_EXACT_RETRY_PROTOCOL_20260907.md` 与
`FINAL14_SPL_A100_EXACT_RETRY_SUBMISSION_20260907.json`。准备过程曾发生传输未完成的脚本
解析失败及 never-started task 的 sacct 记录延迟，均已检查并记录；未引入重复 GPU 提交。

11:23 补充：原 index 16 也以相同 SIGABRT 失败，现已归档并补交 A100 `17089989_16`。
原批次终态为 11 完成、7 失败、3 未启动迁移；全部 10 个未完成 history 均有对应
A100 作业，当前仍因 `QOSGrpGRES` 排队。最新全量核验改为 `17089991`，等待
`17089916` 和 `17089989`；`17089917` 已在执行前取消，不再作为当前 summary。
