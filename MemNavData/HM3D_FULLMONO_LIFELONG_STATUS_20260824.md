# HM3D actual-full-mono lifelong multileg：提交与运行总账

更新时间：2026-08-24（Asia/Shanghai）。本文件记录冻结协议之后的基础设施状态；科学
定义仍以 `hm3d_fullmono_lifelong_protocol_20260824.json` 和内容寻址 task bundle 为准。

## 1. 冻结目标与正式比较

事实前缀为 actual-mono `A -> Novel B`；随后执行 `C(Revisit A) -> B2(Revisit B) ->
C2(Revisit A)`。主比较是共同成功且逐动作一致的 C prefix 之后，B2 上：

- `all_prior`：允许检索实际亲历的 A+B；
- `initial_leg_only`：检索上限永久锁在 A；
- `forced_reject_native`：保留同一 proof stream，但 CEC 永不获得执行权。

all-prior 与 initial-only 在同一 allocation、同一 GPU、同一组已加载 MemNav/NavDP/CEC
进程内顺序执行，并按 population index 平衡先后顺序；forced-native 随后在同一 GPU
上用新的 fail-closed hub 执行。正式指标为逐段 SR、条件 SR、prefix survival、joint
SR、B2 paired gain/loss、exact McNemar 和 scene-cluster bootstrap CI。

## 2. 提交前证据

- 核心 pytest：`69 passed`；
- Habitat/构造 unittest：本机 `17 passed`；
- 隔离 staging bundle：619 files，脱离工作区运行通过；
- 本机真栈：all-prior/initial-only 的 host、GPU UUID、MemNav PID/start ticks、NavDP
  PID/start ticks、hub PID/start ticks 完全相同；forced-native 使用同 GPU 的新进程；
- 远端标准 Singularity + Habitat Python：`17 tests OK (skipped=1)`；
- evaluator 完整参数：`contract_dry_run OK`；
- 远端依赖：Habitat pip-vendored requests `2.32.4`，`__init__.py` SHA-256
  `1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f`，
  `__version__.py` SHA-256
  `143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577`。

第一次远端 preflight 的 bundle `e4a6c1ca3807bd45` 在创建任何 Slurm job 之前被
`ModuleNotFoundError: requests` 拦截。修复只显式接入并哈希固定 Habitat 已有的 vendored
dependency；没有安装包、改变方法、读取 outcome 或覆盖失败 attempt。

## 3. 正式不可变身份

- task bundle：
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_cbef63fd46d88451`
- task receipt SHA-256：
  `cbef63fd46d88451296fbfcb88ee605861497795c916c28deffbac2f1fdee909`
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_20260824/formal_20260824T041000Z_cbef63fd`
- smoke root：同上并追加 `_smoke`；
- 提交时未读取 query outcome；这是 consumed-scene mechanism confirmation，不主张新的
  fresh-scene generalization。

## 4. Slurm DAG

| 阶段 | Job ID | 提交时状态 |
|---|---:|---|
| A/B result-blind construction | `16265026` | pending scheduler |
| seal A/B population | `16266719` | replacement dependency |
| factual actual-mono B collection | `16266720` | replacement dependency |
| causal A+B prefix construction | `16266756` | replacement dependency |
| seal query population | `16266758` | replacement dependency |
| remote true-stack smoke | `16266761` | replacement dependency |
| formal same-process three-arm eval | `16266768` | replacement dependency |
| aggregate | `16266769` | replacement dependency |
| independent verifier | `16266771` | replacement dependency |

正式 array 上界是 130 个 paired evaluation tasks / 390 个 logical arms；实际分母只能由
sealed population receipt 决定。任何中间 completion 或 partial SR 均不改变方法、阈值、
population 或停止规则。

提交后第一阶段已于远端 `2026-08-24 00:43 EDT` 启动首批两个 array task：

- task 0：`2:29` 完成，completion hash
  `2dae6278...3099`；
- task 1：`4:42` 完成，completion hash
  `eb829786...70d8`；
- 两者均为正常完成、无程序失败；各自能构造 Revisit-C，但在冻结的跨 history、
  natural-direction Novel-B 约束下没有合格 donor，因此 `constructible=0`。

这只是 result-blind population constructibility attrition，不是导航 SR，也不会据此放宽
阈值。其余 52 个 task 仍在等待，Slurm 原因在 `QOSGrpGRES`、
`QOSMaxGRESPerUser` 与 `Resources` 间刷新；截至远端 `01:16 EDT`，本账户无 GPU job
实际运行，说明没有代码或依赖失败，只有共享分区调度等待。当前保留原 DAG，不为抢
队列改变分区、population 或科学参数。

尝试把尚未运行的 80-step remote true-stack smoke（job `16265066`）时限从 4 小时
缩到 1 小时，Slurm 以 `Unspecified error` 拒绝；其时限仍为 4 小时。该值只是上限，
不会强制 smoke 跑满，也不影响正式协议，因此不重建依赖链。

## 5. 零历史 scene 的基础设施 repair

原 build wrapper 对父实验中合法的两种空 scene 处理不一致：若存在空 `online_a`
manifest，则能写出零行 completion；若父实验因固定次数 source generation 未完成而明确
封存为 0 history、因此没有 `online_a/`，旧 wrapper 会在任何导航或 query outcome 被
读取前报 `online-A root missing`。父 population receipt 的 54 scenes 全审计确认恰有五个
同类索引：`11,15,34,40,44`；它们全部同时满足：

- 父 per-scene completion 为 `status=complete`；
- `query_policy_outcomes_read=false`；
- Goal-A success、materialized history、retained history 都为 0；
- completion SHA 与冻结 parent population receipt 一致；
- 没有 `online_a` manifest。

修复没有补生成 trajectory，也没有改变 donor、covis、距离或 population 阈值；只为这些
父证据已证明为零行的 scene 写入 hash-bound zero-history attrition receipt。本机 Habitat
门为 `19 tests OK`，远端同容器为 `19 tests OK (skipped=1)`。不可变 repair bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_fullmono_abfix_171ccb30ff2c17f8
SOURCE_BUNDLE.sha256 SHA-256:
  171ccb30ff2c17f8523b7d533ded2705a80ffcf3b8a9e5990223e36995f6ad64
repair array: 16266646 (5/5 COMPLETED, 0:39--0:40 each, CPU only)
```

五个新 completion 已逐文件重算 SHA，并再次确认 0 histories、0 constructible、
`query_policy_outcomes_read=false`。Slurm 拒绝原地修改旧 seal dependency，因此旧的
8 个、从未运行的 downstream jobs 已精确取消；原 build 与全部 fragment 均保留。
replacement seal 用 `afterany:16265026` 等待原 array 全部终止，并把已经完成且原始文件
复核通过的 repair 作为提交前硬门。replacement DAG 的机器可读 receipt 为
`HM3D_FULLMONO_LIFELONG_DOWNSTREAM_REPAIR_RECEIPT_20260824.json`。

截至远端 `2026-08-24 01:59 EDT`，原 build 已处理 index 0--25：24 个正常完成，
index 11/15 按已知旧 wrapper 问题失败且均已有有效 repair completion；index 26 正在
运行，27--53 等待。尚无 query SR 被读取。

## 6. 完成门

只有远端 `independent_verification.json` 明确给出 `verified=true`，并且 verifier 已直接
重读原始 A/B mono receipts、三臂 metrics/plans、completion hashes 与 compute identity，
才能把逐段 SR、prefix survival、joint SR 和配对统计写入本文件并对外报告。

## 7. 2026-08-25 result-blind power expansion 与 shared-C 修复

旧冻结构造只取每条 donor factual A 的最后一帧，最终仅得到 2 条有效 history。v2 不读取
任何旧/新 query outcome，在每条 donor 的完整 factual trace 上固定取 8 个 linspace 时刻，
每个 recipient 最多冻结 2 个、且最多每 donor 1 个候选，并优先不同初始方向 strata。
synthetic candidate identity 与 source online-A identity 分开保存；scene 仍是 bootstrap cluster，
共享 recipient-A 依赖必须报告。目标为至少 24 histories / 15 scenes，未达到仍标为
underpowered，禁止放宽阈值或 replacement。

同时删除旧三臂独立执行 C 的混杂：actual-mono A/B 后只执行一次 factual C，在任何 B2
outcome 前 seal；仅 C-success population 进入 B2，并在同一精确 A/B/C prefix 后比较
all-prior、initial-leg-only 与 forced-native。主 endpoint 是 B2，不再把 downstream C2
混入 treatment effect。Goal-session replay 明确保证不 append memory、不运行 diffusion。

本机、隔离 bundle 与远端生产门均通过：110 pytest、20 本机 Habitat tests、远端 Habitat
22 tests、远端 MemNav/PyTorch 38 tests、完整 contract dry-run 和 11 个 Slurm test-only。

正式 root 与前五段任务：

```text
run root:
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_20260824/
  formal_20260824T171704Z_2ce2ae67
bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_fullmono_lifelong_2ce2ae67e7a6d65b

construct AB       16289393
seal AB            16289398
factual B          16289422
construct prefix   16289433
seal population    16289438
```

Torch `gpu48` 会把 dependency-held array elements 计入 `QOSMaxSubmitJobPerUserLimit`；同时
预提交 collection 与 evaluation 两个 260-array 会被拒绝。因此新增 hash-pinned deferred
launcher `16289882`：population seal 后仅提交 collection；collection seal 与 80-step
true-stack smoke 成功后才提交 evaluation。运行时 job IDs 分别原子写入
`deferred_submission/collect.json` 和 `deferred_submission/evaluate.json`。恢复总收据为
`HM3D_FULLMONO_LIFELONG_POWER_EXPANSION_RESUME_20260825.json`。截至提交时没有 query SR。

## 8. 2026-08-25 17:07 运行更新与 array-bound 归因

Power-expansion 的 result-blind seal 最终只保留 8 histories / 6 scene clusters；冻结目标为
24 / 15，因此 population 合法但 underpowered。8 条 factual-C collection 已全部完成，
其中 C success 为 5/8，覆盖 4 scenes。该数字只决定共同 B2 起点 population，不是任何
CEC-vs-baseline SR。

旧 immutable deferred launcher 错把 maximum 260 source candidates 当成 sealed population
长度，提交了 `16318975_[0-259%4]`。截至本节快照，0--136 已完成，其中索引 0--7 是实际
collection，其余均为 `outside_sealed_population` 空任务；137--259 因 `QOSGrpGRES`
等待。依赖整个 array 的 `16318976` seal、`16318977_0` smoke 与 `16318978` evaluation
launcher 尚未运行，所以没有新的 B2 或 multileg SR。

工作区源码已经把 collection/evaluation array 改为：验证 `SEALED` 与
`population.json.sha256` 后，读取 `accepted` 的精确非零长度，超过冻结上限 260 则
fail closed，并在 receipt 中记录实际 array。该修复只影响未来 content-addressed bundle，
不会回写或伪装改变已经提交的 `16318975`。
