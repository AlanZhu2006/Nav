# Revisit fresh-episode confirmation：运行记录

日期：2026-08-11（CST）
状态：HPC v2 在生成阶段 fail-closed；尚无效果结果。正在进行不读取 arm 标签的生成可行性探针。

## 冻结问题与规模

验证 `known_revisit_direct` 是否在新 episode 上稳定优于 `geometry_router`。20 个旧评测
scene cluster 每场景重新生成 8 条，共 160 条；不读取 development、final-reserved 或
blind。三臂共享独立生成的一条原生 Novel-A trace，且轮换 6 种 B-arm 顺序。

归因口径需精确：`geometry_router` 不只可能 veto activation，还会在 raw-DINO top-8 内用
SIFT/RANSAC 选择通过验证的候选；`known_revisit_direct` 使用 raw-DINO top-1。因此主比较是
**移除完整 geometry expert（候选复核 + activation gate）**，不是数学上只改变一个 veto
bit 的纯消融。旧 N=40 的 K=1/K=8 零结果是独立背景证据，不被本次重复作为新 arm。

协议：`REVISIT_FRESH_CONFIRMATION_PROTOCOL_20260811.md`；机器可读参数：
`revisit_fresh_confirmation_protocol_20260811.json`。

## 提交前验证

- Python compile、shell syntax、相关单元测试通过；
- 合成 160-episode manifest 验证通过；历史 goal 哈希碰撞会 fail closed；
- 本机真实闭环 smoke：`17DRP5sb8fy/episode_0003` 的 geometry/direct/native 三臂均复用
  trace SHA `be231258df805ed9515369477111f811ec60f14792389eb66c1349c03bb74f6f`；
  该 episode 的三臂 B 均失败，只用于 transport 审计，不是效果证据；
- 生成器确定性 smoke：同 scene/seed 两次生成，比较 847 个 metadata/parquet/goal/RGB/
  depth 文件，零哈希差异；`n_frames=422`，seed `2026081200`。

## 不可变源码

- source root：
  `/scratch/yz11502/Research/source_bundles/revisit_fresh_v2_pycache_20260811T1727`
- source receipt SHA256：
  `d4a686cef5427a340df8db6369481b9b05296dbb9e856fa1091806fd800519d1`
- protocol SHA256：
  `6c93788dcadc5eda66a3184f2c99d8a944335144ed8170cae6e005f515f04be0`

source root 已递归移除写权限；HPC 容器内再次运行关键测试并通过。所有 bytecode 均写入
Slurm 临时目录，不写回 source root。

## Slurm 链

结果根：
`/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v2_20260811T1730`

| stage | job | 依赖 | 作用 |
|---|---:|---|---|
| generation array | `15613183` | — | 20 scenes × 8 fresh episodes |
| manifest | `15613185` | `afterok:15613183` | 数据、资产、checkpoint、历史去重审计 |
| evaluation array | `15613187` | `afterok:15613185` | native-A trace + geometry/direct/native B |
| summary（已取消、未启动） | `15613189` | `afterok:15613187` | receipt 字段断言错误，未读取结果 |
| replacement summary v1（已取消、未启动） | `15613709` | `afterok:15613187` | 第二个 receipt 字段问题 |
| replacement summary final | `15613736` | `afterok:15613187` | paired McNemar + scene-cluster bootstrap |

提交时 generation 因项目 QoS GPU 并发额度为 `QOSGrpGRES` pending。其余 stage 正常等待
依赖；不是 preflight、源码或数据错误。不能在 report 生成前填写 SR 或作架构结论。

截至 2026-08-11 17:58 CST，generation 已完成 6/20 scenes（48/160 episodes），完成 task
均为 8/8、exit 0；其余 task 等待共享 `gpu48` QoS 配额。manifest、evaluation 和 summary
继续按 `afterok` 依赖等待。

后续已确认推进到 15/20 scenes（120/160 episodes），此前所有完成 task 均为 exit 0。SSH
认证失效前最后一次文件系统观测为 141/160，array task 18/19 正在运行；由于连接随后在读取
最终 `sacct` 前失效，不能把 141 当作完成数，也不能在恢复认证前断言最后几项成功或失败。
Slurm 依赖链本身不依赖登录会话，仍会自动运行或 fail closed。

### v2 最终生成状态（2026-08-11 19:30 CST）

SSH 恢复后从 `sacct` 和逐 task 日志确认：generation array `15613183` 最终为 `FAILED(2:0)`，
因此 manifest `15613185`、evaluation `15613187` 和 summary `15613736` 均由 `afterok`
依赖取消，未启动、未读取任何 arm outcome。磁盘上的 147 条只是未封存的部分生成物，禁止
进入评测或效果统计。

- scene task 0--16、18：各 8/8，完成；
- task 17，`i5noydFURQK`、seed `2026081217`：只生成 3/8，固定抽样预算耗尽；
- task 19，`gZ6f7yhEvPG`、seed `2026081219`：生成 0/8，固定抽样预算耗尽；
- 两个失败均发生在生成器正常退出并报告 `DONE` 后，由 sbatch 的 exact-count guard 以 exit 2
  拒绝；不是 CUDA crash、超时、磁盘错误或模型评测失败。

这暴露的是生成 preflight 不充分：统一的严格 episode 接受条件在两个小场景上的接受率很低，
而历史 `run_legs` 把外层 `make_episode` 调用数硬编码为 `6 × requested_count`。为区分“预算过小”
与“当前定义实际不可行”，新增了显式 `--episode_attempt_multiplier`；它只延长同一 seed 的 RNG
流，不改变 geodesic、co-visibility、heading、平滑或碰撞接受条件。临时可行性 job `15618301`
只跑 task 17/19，倍率从 6 提到 120，输出隔离在 `sampling_probe_20260811T1935`，不进入正式
manifest。

120 倍探针的最终结果为：`i5noydFURQK` 8/8、`gZ6f7yhEvPG` 5/8；HPC 与本机在后者
得到相同的前五条 episode 描述，本机把预算继续提高到 600 倍后得到 8/8。由此确认接受
集合非空，失败原因只是固定预算太小。v2 原生成的 `i5noydFURQK` 前三条与 120 倍 HPC
探针逐文件 SHA256 完全一致（分别比较 824、678、576 个文件，零差异）；正式新 run 仍将
从头生成并由 manifest 哈希封存，不能把任何 probe 数据直接
拼入。

v1 协议在没有启动 arm evaluation 的前提下版本化为 protocol v2：scene、seed 和所有
episode 接受条件不变，只把 `episode_attempt_multiplier` 冻结为 600，并把该值写入机器
可读 protocol 和最终 manifest receipt。

### v3 正式链（attempt budget 修复后）

新 source 从最终 analysis-fix v2 bundle 复制后，只合入上述生成预算参数、manifest receipt
字段、对应测试以及预结果冻结的独立审计脚本；全包 2,373 个文件通过 `sha256sum -c`，只读
条件下 9/9 相关测试通过。

- source root：
  `/scratch/yz11502/Research/source_bundles/revisit_fresh_v3_attempt600_20260811T2000`
- source receipt SHA256：
  `22a29884c5eba169c4a9dd12bd0398ba8162de04d97b75978903782797475062`
- protocol SHA256：
  `6264acfc955792acb09b5f21e658c277c7cbd2f712a1842fb8d8bc1fe679543c`
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811/fresh160_v3_attempt600_20260811T2000`

| stage | job | dependency |
|---|---:|---|
| generation array | `15622180` | — |
| immutable manifest | `15622182` | `afterok:15622180` |
| paired three-arm evaluation | `15622184` | `afterok:15622182` |
| frozen summary | `15622186` | `afterok:15622184` |
| independent raw audit | `15623100` | `afterok:15622186` |

提交后 generation task 0 已开始运行，其余 task 因 Priority 等待；所有下游仍 fail-closed。
在 `report.json` 产生且独立复算前不得填写效果结果。

v3 generation 最终 20/20 tasks 均 `COMPLETED(0:0)`，恰好生成 160/160 episodes。普通 scene
约 2--3 分钟；最长的 `gZ6f7yhEvPG` 在固定 seed 下于第 1,757/4,800 次候选获得第 8 条，
14:27 完成。该日志直接解释 v2 的 48 次固定预算为何得到 0/8，同时证明 v3 没有换 seed 或
放松接受条件。

manifest job `15622182` 于 34 秒内完成，receipt：

- `data_manifest.json` SHA256：
  `8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5`
- audit：20 scenes、160 episodes、training overlap `[]`、historical episode hash overlap
  `false`、development/blind read 均 `false`；
- manifest 内记录 protocol SHA
  `6264acfc955792acb09b5f21e658c277c7cbd2f712a1842fb8d8bc1fe679543c`、generator SHA
  `cb24379f65cefd4c6072f7f125b4ec04d0be97fd30f88602e3a84eb9ccbf5311` 和 attempt multiplier
  `600`。

manifest 成功后 evaluation array `15622184` 已自动开始；不读取中途 outcome。

### 跨机器生成加速门（拒绝）

为判断是否可用本机 4090 补齐待生成数据，使用冻结 v2 的相同生成器、参数、scene 0 和 seed
`2026081200` 做了预先规定的跨机器逐文件一致性探针。两侧均生成 8 episodes、5,242 files，
但首个 `gen_meta.json` 已有末位浮点差异（例如 `yaw_habitat` 相差约
`4e-16`），因此没有通过 byte-identical 门。该本机 probe 不进入 manifest 或效果统计；正式
数据继续只由正式 HPC v3 generation array `15622180` 产生，避免把跨机器数值路径混入同一
确认集。

### 评测实现复核

- trace source 的 A 段由 hybrid server 的 `phase` 路由调用独立 NavDP server，图子目标间距为
  0，因而是原生 ImageGoal NavDP 轨迹；
- 每个 arm、每个 episode 都以相同 seed 调用 server reset；shared-trace replay 同时重放
  Habitat pose/RGB、MemNav 长记忆和 NavDP 决策帧短记忆；
- `geometry_router` 与 `known_revisit_direct` 使用相同的 legacy metric adapter 和 mixed
  controller，主差异是完整 geometry expert（候选复核/重选及 activation gate）；native 是
  独立 ImageGoal 基线；
- 汇总器逐 episode 校验 trace SHA、三臂 A 结果、scene arm permutation、manifest SHA 和冻结
  参数后，才计算 joint/conditional-B 的配对检验与 scene-cluster bootstrap。

### Analysis-only receipt 修复

本机真实 smoke 暴露出原 summary validator 的一处字段断言错误：纯 NavDP native backend
仍会把 evaluator parser 的惰性默认值记录为 `revisit_adapter="legacy_metric"`，但 v2 summary
错误地要求 `None`。该字段在非 hybrid backend 下不可达，不改变任何动作；若不修只会在全部
rollout 完成后让汇总 fail closed。

进一步把三个 arm 的真实 smoke summary 逐一送入 validator 后，发现 native backend 未启动
MemNav，故两个 graph-subgoal receipt 字段也应为 `null`，而非 hybrid arm 的 `0.0/0.6`。
replacement v1 job `15613709` 同样在启动前取消。generation `15613183`、manifest
`15613185` 和 evaluation `15613187` 始终未修改。

最终分析代码按各 backend 核对实际 receipt，并新增回归测试；4/4 tests 在本机及 HPC
`/dev/shm` 临时目录通过，三个真实 smoke arm receipt 全部通过。最终只读 analysis bundle：

- `/scratch/yz11502/Research/source_bundles/revisit_fresh_v2_analysisfix2_20260811T1825`
- receipt SHA256：`4c77fb5d170d00ee4e5b958826546e34948a11cf59a26e72b28c8ea3bd6a204e`
- replacement summary job：`15613736`

机器可读替换记录位于 run root 的 `analysis_submission.json` 和
`analysis_submission_final.json`。这是预结果、analysis-only 修复，不查看标签、不改统计
规则、不改 rollout 源码或任何 arm。

正式 report 生成后，还将用
`independent_audit_revisit_fresh_confirmation.py` 从 raw CSV/plan receipts 独立复算；该脚本不
导入项目 summarizer，重新检查 160 个 key、逐 plan seed echo、A-trace SHA、配对计数、exact
McNemar 和 scene-cluster interval。其向量化 bootstrap 已用合成数据与冻结汇总器逐值对齐。
独立审计 job `15623100` 已预先以 `afterok:15622186` 接在冻结汇总之后；运行前再次校验
manifest/report SHA，且拒绝覆盖已有 audit 文件。

### v3 正式结果：fresh-episode 架构门通过

evaluation array `15622184` 的 20/20 scene tasks 全部 `COMPLETED(0:0)`，每个 scene 均有
8 条 shared-A trace 和 geometry/direct/native 三臂结果，共 160 个严格配对 episode。冻结
summary job `15622186` 随后正常完成，生成只读 report：

- `report.json` SHA256：
  `6a23ec06b2aa4e10801d1911a4f49f562b94542a7986b26f79626c4012156f49`；
- report audit：20 scenes、160 episode keys、shared native Goal-A trace、balanced arm order、
  training-scene overlap `[]`、development/blind read 均 `false`；
- scope 明确为“旧 20 个 scene clusters 上的 fresh-episode replication”，不是 fresh-scene、
  blind 或论文最终确认。

三臂闭环结果如下。Novel-A 完全共享同一 trace，因此三臂均为 118/160（73.75%）；B 条件
成功率只在这 118 条共同 A-success episode 上计算。

| arm | Novel-A | Revisit B \| A | joint A∧B | conditional-B SPL | conditional-B final distance |
|---|---:|---:|---:|---:|---:|
| `geometry_router` | 118/160 = 73.75% | 93/118 = 78.81% | 93/160 = 58.13% | 0.5565 | 1.843 m |
| `known_revisit_direct` | 118/160 = 73.75% | 109/118 = 92.37% | 109/160 = 68.13% | 0.7959 | 1.153 m |
| `native` | 118/160 = 73.75% | 31/118 = 26.27% | 31/160 = 19.38% | 0.1136 | 6.223 m |

预注册 primary contrast（direct − geometry）为：

- joint：`+20/−4`，risk difference `+10.0 pp`，exact McNemar
  `p=0.0015438795`，scene-cluster bootstrap 95% CI `[+5.0,+15.625] pp`；
- conditional B：同样 `+20/−4`，risk difference `+13.56 pp`，exact McNemar
  `p=0.0015438795`，scene-cluster 95% CI `[+6.42,+21.49] pp`；
- 20 个 gains 分布在 12 个 scenes，4 个 losses 分布在 2 个 scenes；12 个有 discordance 的
  scenes 中 11 个净正、1 个净负，不是单一 scene 驱动；
- direct 同时远高于 native：joint `+79/−1`、`+48.75 pp`、
  `p=1.34e-22`、scene-cluster 95% CI `[+38.125,+59.375] pp`。

冻结独立脚本已在本机只读复制的最小 raw-artifact mirror 上完成复算：160 个 raw keys、三臂
seed/Goal-A/trace SHA、逐 plan diffusion-seed echo、backend/adapter receipts、三组 paired counts、
exact McNemar 和全部 scene-cluster intervals 均与正式 report 逐值一致，输出 audit 为 `ok`。
本机 audit JSON SHA256 为
`154205c2aa52c7a0c23e32b5cc8e390f87ebc0a0cf415698e624da9291b52239`；冻结 audit script
SHA256 为 `399fc208d5d5bc21fe91534c8c2d3f0fa64049aeab195415f0f299dbe85029e8`，与本机工作树
逐字节一致。随后同一冻结脚本直接在原始 HPC run root 上生成只读
`independent_audit_login.json`，再次返回 `audit: ok`，其 SHA256 也恰为
`154205c2aa52c7a0c23e32b5cc8e390f87ebc0a0cf415698e624da9291b52239`。因此原始数据与
最小镜像的独立输出逐字节相同；排队中的永久 audit job `15623100` 保留为冗余复核，不再是
结论前置条件。最终 completion audit 另外确认 evaluation tasks `20 completed / 0 failed`、
160 份 shared-A trace、geometry/direct/native 各 160 份 plan，三个只读 receipt 均通过
`sha256sum -c`；本机 9/9 协议与汇总回归测试通过。

### 机制拆分与冻结架构决定

geometry 在 118 条 A-success episode 中激活 96 条（81.36%）。按其激活状态做预先结果之外的
描述性拆分：

- geometry **未激活**的 22 条：geometry 9/22，direct 20/22，配对 `+11/−0`；
- geometry **已激活**的 96 条：geometry 84/96，direct 89/96，配对 `+9/−4`。

这说明 primary 增益中超过一半来自 geometry hard gate 的假阴性；即使 gate 激活，绕过
RANSAC 候选复核/重选仍为净正。后一子组的普通 exact McNemar 为 `p=0.267`，只是机制描述，
不能作为独立显著性主张；正式结论由全体 160 条的预注册 clustered primary contrast 支撑。

所有冻结替换条件均满足：primary delta > 0、`p<0.05`、scene-cluster CI 下界 > 0，且 direct
joint 不低于 native。因此正式 decision branch 为：

`replace_geometry_hard_gate_then_seek_fresh_scene_confirmation`

架构含义严格限定为：在**已经进入 known-Revisit 分支**时，以 raw-DINO top-1 memory pose
经现有 legacy metric adapter 产生子目标，并继续复用相同 mixed NavDP controller；SIFT/RANSAC
不再拥有激活否决权或候选重选的主控制权，可保留为 telemetry、审计或安全告警。这个结果不
解决线上如何判定 Novel/Revisit，也不授权 blind eval 或 paper claim；这 160 条已经消费，禁止
继续调阈值。下一步是冻结上述简化分支，在与这 20 scenes 不相交的 fresh scenes 上做确认。

### v1 fail-closed 记录

v1 generation task 0（job `15613040_0`）在 14 秒内、生成任何 episode 之前失败：显式
`py_compile` 尝试在只读 source root 写 `__pycache__`。旧链 `15613040/42/44/46` 随即
取消。v2 将 `PYTHONPYCACHEPREFIX` 固定到 Slurm 临时目录，并在 source root 只读条件下
实测 compile 成功后才重新提交。该故障不产生数据或效果结果。

### 2026-08-13：actual online-A Revisit 标签补充审计

原生成器按 expert-A 轨迹选择 Revisit B，而正式 memory 来自 shared online-A trace。为排除
两者错位，补充审计在不重跑策略的前提下，把每个 Goal-B 与实际 online-A 的所有帧重新做
遮挡感知 3D 共视。结果：A-success 分母中的 `118/118` 均有 `max covis>=0.20`，其中
`113/118` 有强支持 `>=0.50`；全部 11 条 `<0.20` episode 都是 A failure，不进入 conditional-B
统计。因此 `109/118` 与 direct-minus-geometry `+20/-4` 没有被 expert/online 标签错位抬高。
强支持子集仍为 direct `104/113`、geometry `89/113`、配对 `+19/-4`、
`p=0.00259948`。

HPC job `15655698` 在 L40S 上 `COMPLETED(0:0)`，160 个 Goal-B 和 34,798 个 online trace
JPEG 均逐哈希复现；独立本机 RTX 4090 复算的 160 条曲线、rows 和分层统计与 HPC 逐字段
一致。完整口径、receipt 与输出 SHA 见
`REVISIT_FRESH_ONLINE_OBSERVABILITY_RESULT_20260813.md`。这个结论确认 actual-online Revisit
标签，但仍不解决部署时的 Novel/Revisit 自主选择。
