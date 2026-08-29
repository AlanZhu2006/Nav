# 2026-08-29 会议实验活跃总账

更新时间：2026-08-29 15:35（Asia/Shanghai）。本文件是当前调度与证据边界的
唯一简表。任何 active rollout 都不读取 partial SR、SPL、final distance 或逐臂
outcome；只有 independent raw-file verifier 通过并写入最终 seal 后，才允许打开结果。

## 1. 当前结论先行

- **HM3D 的 ViNT 两行已经成为正式正结果。** 在冻结的 28-history、21-scene、
  56-query mixed Novel/Revisit population 上，CEC 对 Novel 做到 0 takeover 和 exact
  fallback；Revisit 从 `3/28` 提升到 `19/28`，paired `+16/-0`，风险差
  `+57.14 pp`，exact McNemar `p=3.05e-5`。Overall 从 `6/56` 到 `22/56`。
- **上一轮 NavDP 的 `14/56 vs 14/56` 不是方法零结果。** 它有 2,657 个
  runtime-failure plans，全部由 authority endpoint 契约不匹配触发，随后按设计
  exact fallback。它只能作为 fail-closed 基础设施审计，不能进入性能表。
- **正确的 NavDP 完整双臂修复已经第三次提交。** 前两轮 smoke 分别暴露并阻断了
  authority import 闭包缺失和两个同名 `policy_agent.py` 的跨进程 namespace 碰撞；
  第三轮 process-local import precedence smoke 已通过，54-rank formal array 正在
  运行。任何 endpoint/runtime failure 仍会直接判为任务失败，不能被统计为正常
  reject。
- **MP3D 第二数据集的 outcome-blind query 构造已完成，但未过 prospective power
  gate。** 独立 verifier 得到 14 histories / 10 scenes，front/side/rear 为
  5/0/9，低于冻结的 20 / 12 / 每方向至少 4；因此 controller 四行没有提交，也没有
  新 MP3D SR。下一步只审计预先冻结的 phase-2 source episodes 能否做新的 full-mono
  source expansion，不降低任何 query 阈值。该扩源 16-scene full-mono Goal-A
  collection 已全部完成；104 条 source traces 的 immutable ledger 已冻结，新的
  36-scene construction/verifier 链已提交，但仍未运行 query controller。
- **HM3D lifelong B2 仍在等待 GPU 优先级。** 它是会议 Table 2 的补充机制实验，
  不是 powered confirmation。
- **会议 Table 2 的主缺口已经冻结为一项新的 construction-only 实验。** 它不复用
  已经跑过的旧 Goal-C，而是在 22 条 sealed actual-mono `Novel A -> Novel B` 成功
  前缀上重新构造一条 Novel-C 与一条 Revisit-C；两者都对完整 A+B history 定义，
  runtime role 隐藏。首轮 smoke 在 policy 前拦截了 task-bundle dependency closure
  缺失；修复只补齐精确依赖和 resolved-signature preflight。replacement
  smoke / 22-cell construction / finalizer / independent verifier 为
  `16545221 / 16545222 / 16545223 / 16545224`；replacement smoke 已通过，formal
  construction 已启动；verifier 过门前禁止闭环。

## 2. HM3D Table 1：跨 controller 正式结果与修复

### 2.1 冻结比较对象

- 28 条 actual-mono causal histories；
- 21 个 fresh HM3D scene clusters；
- 每条一条 Natural Novel 与一条 Revisit query，共 56 queries；
- runtime 不读取 Novel/Revisit role；
- 每个 controller 内 native 与 CEC 同 query、同 seed、同 checkpoint、同输入模态、
  同步数预算并严格 paired；
- NavDP 与 ViNT 的绝对 SR 不作为 controller superiority 的 paired estimand。

### 2.2 ViNT：已封存、可报告

运行链：原数组 `16526731`，唯一非科学崩溃 cell 18 精确重跑 `16540176`，
replacement aggregate/verifier `16540207/16540208`。独立 verifier SHA-256：
`89a5e85f7899e283ce8924fd1777e98b061ceb6577641cb6811b4c5f6b02bf32`。

| Role | ViNT native | ViNT + CEC | paired gain/loss |
|---|---:|---:|---:|
| Novel | 3/28 | 3/28 | 0/0；0 takeover，exact fallback |
| Revisit | 3/28 | 19/28 | +16/-0 |
| Overall | 6/56 | 22/56 | +16/-0 |

Revisit 风险差为 `+57.14 pp`，exact McNemar `p=3.0517578125e-5`；overall 风险差
为 `+28.57 pp`，scene-cluster CI 为 `[+20.0,+36.54] pp`。CEC 在 27/28 Revisit
queries 获得接管权，在 0/28 Novel queries 接管。它首次给出强闭环证据：CEC 的
proposal--witness--authority 接口不只适用于 NavDP。

### 2.3 NavDP 旧链：为什么无效

旧链先后修复了 byte-identical JPEG 的 causal transaction cache 和 aggregator
读取错误字段两个基础设施缺口。analysis-only replacement `16542535/16542536` 与
joint seal `16542548` 均已完成；只有在 seal 后才打开结果。

打开后发现 `mono_native = mono_cec = 14/56`、CEC takeover 为 0，并且所有 CEC
请求累计 2,657 个 runtime-failure plans。原始 receipt 的确定性错误是：

```text
RuntimeError: certificate endpoint used wrong authority policy
```

原因是 transaction-aware NavDP server 与旧 MemNav authority endpoint 被拼在一起；
后者不接受/回显 evaluator 冻结的 `strict_certificate` policy。因此 evaluator 对每次
异常都安全地退回 native。ViNT 在同一 population 上能接受 27/28 Revisit，进一步
排除了“历史没有几何 witness”这一解释。

### 2.4 NavDP 正确修复与 smoke 闭包门

新 server overlay 只组合两个已经 receipt-bound 的父组件：

- Final14 strict-authority MemNav server/policy agent；
- full-mono transaction-aware NavDP server。

checkpoint、DINO/LightGlue/PnP、certificate 阈值、2.5 m residual、query、history、
seed、arm order、600-step budget 与 1 m success radius 均未改变。完整 native/CEC
双臂必须同进程重跑，不能复用旧 native。

新增三层硬门：history runner、aggregator、independent verifier 都要求
`runtime_failure_plans == 0`。正常 certificate reject 仍合法并 exact fallback；
endpoint/proof runtime failure 会终止 cell 并阻断全部统计。

第一轮 smoke `16543736` 在正式 rollout 前以 `2:0` 失败，并自动取消全部下游。
错误不是 outcome，而是组合 overlay 漏复制 authority 父包中的
`router_candidates.py`；`policy_agent.py` 因而从旧 base 解析出不兼容实现，缺少
`causal_goal_support_indices`。这证明 smoke 门有必要，也证明此前仅做 symbol grep
不足以验证 Python import resolution。

closure-2 修复只加入同一个 Final14 authority 父 bundle 的 byte-identical
`router_candidates.py`，并新增 overlay-first `PYTHONPATH` 下的真实
`import policy_agent, router_candidates` 断言。没有改变方法、阈值或 population。

- replacement task bundle：
  `hm3d_table1_navdp_authority_transaction_repair_82e71f19ee7f4e52`；
- task receipt：
  `82e71f19ee7f4e5233fae499633ce5a233c9c036bb41b9e2bf7d4f0f18effd7d`；
- closure-complete server overlay receipt：
  `718661db1733d5de16cd86687eec880a8d02fc5ae5ca982e1ab7d5bde5e96f7d`；
- 远端隔离预检：8/8 contract tests、完整 bundle SHA、authority dependency import、
  endpoint/transaction symbols、bash syntax 与 Slurm test-only 均通过；
- replacement smoke：`16544226`；
- 54-rank formal array（并发 2）：`16544227`；
- aggregate / independent verifier / replacement joint seal：
  `16544228 / 16544229 / 16544230`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/navdp_authority_transaction_closure_repair_20260829T053627Z_82e71f19`。

closure-2 smoke `16544226` 同样在 formal 前以 `2:0` 停止。MemNav authority import
已经正确，但通用 `PYTHONPATH` 把 MemNav 的脚本级 `policy_agent.py` 放在 NavDP
同名模块之前，使 `navdp_server.py` 无法导入 `NavDP_Agent`。这仍是零 rollout 的
基础设施错误。

第三轮只改变进程局部的 sibling precedence：MemNav server 优先解析 `memnav/`，
NavDP server 优先解析 `navdp/`；二者仍只访问同一 receipt-bound source set。提交前
远端以各自真实 `PYTHONPATH` 导入模块并核验 class/source path，9/9 contract tests、
bundle hash、bash syntax 和 Slurm test-only 全部通过。

- task bundle：
  `hm3d_table1_navdp_authority_transaction_repair_51ee9a4ca063c7f1`；
- task receipt：
  `51ee9a4ca063c7f1125a69dd226918ea9a8bd6404c379df58b28e2b296e68eec`；
- smoke / formal：`16544384 / 16544385`；
- aggregate / verifier / seal：`16544386 / 16544387 / 16544388`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/navdp_server_namespace_repair_20260829T054615Z_51ee9a4c`。

smoke `16544384` 于 4 分 25 秒完成并返回 `0:0`。它验证了 authority endpoint、
transaction 和两个 server 的 process-local Python namespace 能在真实 GPU 进程中
贯通。随后 54-rank formal array `16544385` 自动启动，并发严格为 2；15:35 时 27 个
rank 已结构完成、2 个正在运行，`16544386 / 16544387 / 16544388`
仍按依赖等待。尚未读取任何 partial policy outcome。

## 3. MP3D Table 1：第二数据集构造

MP3D 官方 90 scenes 已被 train/development/blind/Final14 使用，因此不能声称
fresh-scene。冻结的合法 claim 是：

> reused scene/history，new outcome-blind query 的跨数据集、跨 controller replication。

协议从 Fresh160 的 20 scenes / 40 actual-mono histories 出发，不读取旧 Goal-B
outcome；每条 history 尝试构造一个 `covis<0.10` 的 unsupported Novel 和一个
`0.55<=covis<=0.90` 的 Revisit，并用 JPEG SHA 与 pose+yaw 同时排除旧 consumed
queries。prospective power gate 要求至少 20 histories、12 scenes，且
front/side/rear 各至少 4。只有 independent construction verifier 授权后才允许提交
NavDP/ViNT 四行 controller rollout。

- construction task bundle：
  `mp3d_table1_new_query_6813d50153b355e5`；
- smoke：`16542589`，`COMPLETED 0:0`；
- 20-cell construction array：`16542590`，并发 2；
- finalize / independent verifier：`16542591 / 16542592`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_new_query_20260829/construction_20260829T050401Z_6813d501`。

全部 construction cells、finalize 和 independent verifier 均已正常完成。独立结构
复算得到：14 histories、10 scene clusters、28 queries；方向分层为 front 5、side 0、
rear 9；旧 consumed Goal-B identity overlap 为 0，`policy_outcomes_read=false`。
verifier SHA-256 为
`0270d7fd2fd7e7e6c05181cbd34b7c2eef01b843b98e83e4efb0fcb671264fb4`。

因此 `formal_policy_evaluation_authorized=false`：这是一项有效的 constructibility
null，而不是方法 SR 零结果。没有提交任何 controller rollout，也没有读取新 SR。
当前允许的下一步仅是利用在旧 query outcome 之前已冻结的 phase-2 16 scenes × 4
episodes source identities，重新收集 actual full-mono Goal-A histories并构造全新 query；
必须排除所有已消费 query identity，并再次独立通过同一个 prospective power gate。

### 3.1 Full-mono source expansion：已冻结并提交

旧 phase-2 的 64 条 metric-depth Goal-A trace 不复用；只复用其在旧 query outcome
之前已冻结的 16 scenes × `episode_0002..0005` identities。新链使用 frozen NavDP +
causal LingBot monocular depth 重新收集 Goal A，500-step / horizon 8 / 1 m success
contract 与第一批 MP3D actual-mono source 对齐。

为了不把旧 query 重新包装成“新 query”，构造器把 Attempt-7 与 phase-2 的 natural、
support-controlled 四份 sealed manifest 全部读成 forbidden identity ledger：共核验
112 条原始 query identity，去重后覆盖 15 scenes / 92 identities；新 Novel/Revisit
任一 JPEG SHA 或 pose+yaw 命中都会拒绝。旧 policy summary/outcome 不被读取。

- task bundle：
  `mp3d_table1_fullmono_source_expansion_f3e7c3e5ab65ba4f`；
- receipt SHA-256：
  `f3e7c3e5ab65ba4f49a6c47b1d6b29eb04fa6d6d39307c6deddcbdcf414c3f3d`；
- source smoke：`16544770`，`COMPLETED 0:0`，2 分 42 秒；
- 16-scene / 64-episode source collection：`16544773`，并发 2；
- ledger freeze + construction deferred launcher：`16544776`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5`。

16 个 source cells 已全部完成，deferred launcher `16544776` 也以 `0:0` 完成。它生成
了 36 scenes / 104 source traces 的 immutable ledger，且回执明确
`policy_outcomes_read=false`：

- expanded ledger SHA-256：
  `666d4e86f9ca7f7b52b1324044a73d096c9435bbeeaa0c231deb647076c2de17`；
- 36-scene construction / finalize / independent verifier：
  `16545793 / 16545794 / 16545795`；
- controller rollout 仍未提交。

只有 verifier 再次给出 `formal_policy_evaluation_authorized=true`，才允许另行提交
NavDP/ViNT 四行。

## 4. HM3D lifelong / Table 2

上游 factual-C integrity barrier 覆盖冻结 22 histories / 15 scenes；最终 17 条
accepted histories 可进入 B2。两者属于不同阶段的分母，不能混写。

- node-affinity repair bundle：
  `hm3d_lifelong_node_affinity_repair_ddd01842308dfa37`；
- resume launcher：`16540396`，已完成；
- true-stack smoke：`16540468_[0]`，等待 `Priority`；
- node-affine formal launcher：`16540469`，等待 smoke dependency。

这条线要补的是会议 Table 2 中 retained history 随 leg 积累的机制证据。它仍明确
标为 underpowered；即使结果方向漂亮，也不能用 17 条样本冒充 powered confirmation。
它也不能替代 Table 2 的 mixed-role 主比较。

### 4.1 Table 2 mixed-role 主比较：已冻结并提交 construction-only

旧 Revisit-C 已经有运行结果，因此不能直接重跑并称为 result-blind formal query。
新协议只复用在任何 `C/B2/C2` outcome 之前封存的 22 条 successful actual-mono A/B
前缀，并重新构造两个从未执行过的 Leg-3 query：

- Novel-C：对完整 A+B causal history 的 max covis `<0.10`；
- Revisit-C：来自真实 A/B observation 周围的受控位姿扰动，combined-A/B covis
  `[0.55,0.90]`；
- 旧 Goal-B/Goal-C 的 JPEG SHA 与 pose+yaw identity 全部禁止复用；
- front/side/rear 采用均衡首选、确定性 fallback；
- prospective gate：至少 16 histories、10 scenes、每方向至少 3；未过门不得跑 policy。

首轮 smoke `16545154` 在 90 秒、任何 query policy rollout 之前以 `1:0` 停止，并自动
取消 `16545155 / 16545156 / 16545157`。原因是首个 immutable task bundle 没有携带
当前 `build_final14_role_pair_scene.py`，Python 因而从旧 base bundle 解析出不含
`maximum_paired_distance_m` 和 `NaturalNovelConstructionError` 的历史实现。这是确定性
dependency-closure 错误，没有方法 outcome。

replacement bundle 加入 role-builder 的完整直接闭包，并在 Habitat 启动前用
`inspect.signature` 核验五个冻结参数和结构化 rejection type；科学协议、query seed、
阈值、population 和 controller 均未改变。本地与远端真实环境已经通过 5/5 合约回归、
Habitat cold import、resolved source-path/signature、bundle SHA、JSON、Python 与 bash
预检。replacement 冻结提交为：

- task bundle：`hm3d_table2_leg3_repair_8e909a5ba81c146d`；
- task receipt：
  `8e909a5ba81c146dd2fe7c9e179fee553672608ee78c08ab4cbfc5b8be23baf1`；
- smoke / 22-prefix construction / finalizer / independent verifier：
  `16545221 / 16545222 / 16545223 / 16545224`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table2_leg3_mixed_role_20260829/construction_repair_20260829T064841Z_8e909a5b`；
- 提交时 `future_policy_evaluation_submitted=false`，未读取任何新 query outcome。

replacement smoke `16545221` 已于 1 分 29 秒完成并返回 `0:0`；它在真实 H100/Habitat
进程中通过了 dependency source-path/signature、输入哈希和单个 prefix 构造。随后
22-cell formal construction `16545222` 自动启动，并发严格为 2。15:35 时 12 个 cells
已结构完成、2 个正在运行，finalizer/verifier 按依赖等待；尚未读取任何构造 population
统计。

闭环阶段也已 result-blind 准备完成，但尚未提交：现有 query runner 新增显式
`actual_ab` history contract，运行前要求 A/B 两段长度、prefix receipt 与逐帧 trace
语义一致；Table-2 专用 pair/analysis DAG 只运行 `mono_native / mono_cec`，并在独立
统计中复算原始 final distance、exact fallback、Novel/Revisit takeover、A/B factual
waterfall 与 Revisit anchor 来自 A 还是 B。输出被硬标为 `C | A,B`，并显式禁止报告
成无条件三段 joint SR。本地 16 项相关回归全部通过；提交脚本会在 construction
verifier 未授权时 fail closed。

该设计把会议表里的分母纪律写死：Leg 1/2 报来源 factual waterfall；Leg 3 报
`C | A,B` 的 paired Novel/Revisit effect，绝不把条件于 A/B 成功的分母包装成无条件
三段 joint SR。协议见
`HM3D_TABLE2_LEG3_MIXED_ROLE_PROTOCOL_20260829.md`。

## 5. 与会议清单的精确对账

| 会议交付物 | 当前状态 | 下一道门 |
|---|---|---|
| Table 1 HM3D 四行 | ViNT 两行成立；NavDP 第三轮 smoke 已通过，formal 正在运行 | `16544385 -> 16544386 -> 16544387 -> 16544388` 全链 seal |
| Table 1 MP3D 四行 | 首轮 14 histories / 10 scenes / side 0 未过 power gate；full-mono source expansion 已完成并冻结 36 scenes / 104 source traces；新 construction 已提交 | `16545793 -> 16545794 -> 16545795`；仍禁止 controller rollout |
| Table 2 HM3D by leg | 22 条 factual A/B prefix 已封；replacement smoke 已通过、formal construction 正在运行；专用 actual-AB paired runtime 与独立分析已预备但未提交；B2 仍等待 | `16545222 -> 16545223 -> 16545224`；过 power gate 后才运行 `submit_hm3d_table2_leg3_navdp_hpc.sh` |
| Depth ablation | 已有 Gate C/D、Final14 factorial 与 full-mono 证据 | 先审计能否同 population 重组，禁止拼不同分母 |
| CEC mechanism ablation | Raw/CEC/known-role 底层证据大部分已有 | 统一导出 retrieval accuracy、FA/FR；不急着重跑 |
| Real robot | 软件合约与 no-motion transport 已有 | 仍缺冻结 protocol 下的 paired autonomous trials |
| Trajectory-length bins | 当前 2--9 m 为主，长距离桶不可构造 | 低优先级，主表完成前不启动 |

## 6. 当前最优执行顺序

1. 让已通过 smoke 的 HM3D NavDP 54-rank formal 完成并封存；期间不读 partial SR。
2. 让 MP3D 36-scene construction/verifier 自然完成；不降低阈值、不复用已消费
   query，也不在 verifier 授权前提交四行。
3. 让 Table 2 新 Leg-3 construction-only DAG 先过独立 power gate；只有授权后才提交
   `mono_native / mono_cec`，不复用旧 Goal-C outcome。
4. 利用空闲 GPU 让 lifelong B2 自然启动，但不为抢队列取消 Table 1。

统一方法名、输入模态、hidden-role、分母、统计与结果开放规则见
`CONFERENCE_EXPERIMENT_CONTRACT_20260829.md`。该契约只索引冻结 protocol，不替代
episode-level receipts 或 independent verifier。
5. HM3D 与 MP3D Table 1 seal 后生成会议主表；Table 2 始终把 factual A/B waterfall
   与条件 Leg-3 treatment effect 分开报告。
6. 不再启动新 learned module、GOAT 适配、length-bin 或额外 controller 支线；它们
   现在都不在论文关键路径上。

## 7. 工程与审计状态

- 共享 `alantorch` ControlMaster 正常，远端身份严格为 `yz11502`；本轮所有 HPC
  操作均通过共享 SSH 的交互 PTY 完成。不能再把 no-PTY/SCP channel 的挂起误报为
  Slurm controller down。
- 主提交器已永久改为固定 authority+transaction 组合 overlay，避免未来重新使用
  transaction-only endpoint。
- NavDP canonical direction 字段已从 Novel query 的
  `assigned_direction_stratum` 读取；不再访问虚构的 history-level 字段。
- MP3D/HM3D aggregators、verifiers 与 sbatches 已参数化，但 dataset claim scope
  必须显式给出。
- 当前扩源实现已通过 129 项相关回归；提交 receipt 与本状态更新完成 commit/push 后，
  工作区才可称为本轮 release。

权威提交收据：

- `HM3D_TABLE1_NAVDP_AUTHORITY_TRANSACTION_REPAIR_SUBMISSION_20260829.json`；
- `HM3D_TABLE1_NAVDP_AUTHORITY_TRANSACTION_CLOSURE_REPAIR_SUBMISSION_20260829.json`；
- `HM3D_TABLE1_NAVDP_SERVER_NAMESPACE_REPAIR_SUBMISSION_20260829.json`；
- `MP3D_TABLE1_NEW_QUERY_SUBMISSION_20260829.json`；
- `MP3D_TABLE1_FULLMONO_SOURCE_EXPANSION_SUBMISSION_20260829.json`；
- `HM3D_TABLE2_LEG3_CONSTRUCTION_SUBMISSION_20260829.json`；
- `HM3D_TABLE2_LEG3_CONSTRUCTION_CLOSURE_REPAIR_SUBMISSION_20260829.json`；
- `HM3D_TABLE1_NAVDP_ANALYSIS_PATH_REPAIR_SUBMISSION_20260829.json`。
