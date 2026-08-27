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

## 9. 2026-08-26 v2 最终 shared-C pilot

旧 immutable array 最终自然结束，aggregate 与独立 verifier 均完成。最终共同 C prefix
只有 5 histories / 4 scenes，因此以下结果严格是机制 pilot：

| B2 arm | success |
|---|---:|
| `all_prior` | `4/5` |
| `initial_leg_only` | `2/5` |
| `forced_reject_native` | `2/5` |

- `all_prior` 对另外两臂均为 paired `+2/-0`，exact McNemar `p=.5`；
- `all_prior` 在 3 条 episode 中实际接管，3/3 都使用 factual-B anchor；
- 两条基线失败/`all_prior` 成功的 gain 均来自该 B 段新增历史，未见 paired loss；
- 三臂共享精确 A/B/C prefix、same-process pairing 为真；
- full-mono audit 覆盖 5/5 histories，A/B/B2 metric-depth reads 为 0；
- `shared_c_independent_verification.json` 给出 `verified=true`。

结果方向支持“持续积累的后续历史具有边际价值”，但 N=5 不能升级为正式 lifelong
confirmation。

## 10. 2026-08-26 prospective v3 construction power gate

v2 不能通过原样重跑扩样：130 条 materialized A histories 只产生 33 个冻结 Novel-B
候选；21/33 factual B 失败、另 4/33 在实际 B 终点后不满足冻结 C 距离带，最终只剩
8 个 B-supported histories，再经 factual C 后只剩 5 个。该 attrition 已从原始 completion
收据复算。

因此 v3 在读取任何新 B/B2 outcome 前冻结为单独的 construction-only 阶段：

- 每个 donor factual trace 固定取 24 个 linspace 时刻；
- 每 recipient 最多 8 个候选、每 donor 最多 4 个；
- 任意两个冻结 B 候选平面距离至少 2.0 m，使两个 1.0 m success disk 不重叠；
- 仍保持跨 history、Novel covis `<.10`、A-to-B/B-to-C `2--9 m`、相同楼层及固定排序；
- 只有封存候选达到 `>=96 histories / >=15 scenes`，才授权后续 factual-B；否则在任何
  v3 导航 rollout 前停止。

冻结协议：
`hm3d_fullmono_lifelong_power_expansion_protocol_20260826.json`。

本机、隔离 bundle 与远端生产门：

- 本机/staged：`113 passed`；
- 本机 Habitat：23 tests OK；
- 远端 Habitat：25 tests OK（1 skip）；
- 远端 MemNav：38 tests OK；
- evaluator contract dry-run：OK；
- 所有 Slurm 脚本均通过 `--test-only`。

不可变提交：

```text
run root:
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_fullmono_lifelong_power_v3_20260826/
  formal_20260826T141733Z_375f0b68

bundle:
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_fullmono_lifelong_375f0b6879b2ff87

construct-only array: 16401203_[0-53%4]
seal/gate:            16401233
```

提交时 array 因 `QOSGrpGRES` 等待；seal 等待 dependency。没有 factual-B、factual-C 或
B2 job 被创建，`query_outcomes_read_at_submission=false`。本轮下一次科学决策只读取 sealed
construction count 与 scene count，不读取导航 SR。

## 11. 2026-08-27 v3 gate 结果与构造性归因

v3 construction 54/54 scenes 与 seal 均已完成，但只得到 `52 histories / 19 scenes`；
冻结门为 `96 / 15`，因此 `construction_target_met=false`、
`factual_B_authorized=false`。没有任何 v3 factual-B、C 或 B2 rollout，也没有新 SR。

独立 result-blind audit 在 54 scenes 上逐候选复算，并精确复现全部 52 个 sealed donor
identity：

- 130 条 materialized actual-A 中只有 80 条能在 runtime anchor `frame >=39` 后构造
  距 endpoint 至少 2 m 的 Revisit-C；其余 50 条最大范围均为 `0.638--1.995 m`；
- 80 条 C-constructible histories 中，73 条有 same-floor donor，68 条满足 A-to-B
  距离带，59 条同时满足 B-to-C 距离带，最终只有 33 条满足 Novel covis `<.10`；
- 3,840 个 temporal proposals 中有 330 个 eligible，但只覆盖这 33 个 recipients；2 m
  去重后为 52 candidates，未损失任何 eligible recipient coverage。

这证明 v3 失败是“短 actual-A 历史 + 跨 successful-online-history donor 稀缺”的结构性
构造问题，不是 24-frame 采样不足。继续加密 temporal frames 或放松安全/独立性阈值均
不会形成可信扩样。

当前先做不读结果的 direct-natural-B audit，检验 Novel-B 是否可以像 Final14 一样由同场景
navmesh 确定性渲染，而 C 继续严格来自 actual-online A。单场景 GPU smoke `16441089_1`
已通过。原全量 array `16441206/16441207` 在未启动时取消，因为“一条 A 一个 B”的
理论上限 80 无法检验原 96-candidate gate；修正版每条 A 最多 4 个且相隔至少 2 m，GPU
smoke `16441408_1` 正在等待配额。CPU smoke `16441094_1` 已确认无 CUDA/EGL 时不能
渲染。完整数字、不可变 bundle 和允许结论见
`HM3D_FULLMONO_LIFELONG_CONSTRUCTIBILITY_AUDIT_RESULT_20260827.md`。

## 12. 2026-08-27 direct-natural-B 正式构造审计

`16441408_1` 已在 H100 上用 53 秒正常完成；该单场景有 3 条 controlled Revisit
history，但仍没有 Natural-B candidate。它只通过 renderer/runtime smoke，不能用于判断
54-scene population。

按照现行 HPC hardening，正式模板已从 `h200_public/l40s_public` 收窄为
`h100_tandon,a100_tandon`；科学 Python 与成功 smoke 逐字节相同。新 immutable bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_lifelong_natural_b_audit_e2832e17231534e3
SOURCE_BUNDLE.sha256 SHA-256:
  e2832e17231534e38db1b3b507ddf68881ee8bd56548c3151235e52a397f3121

run root:
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_fullmono_lifelong_power_v3_20260826/
  natural_b_audit_formal_20260827T125748Z_e2832e17

scene array: 16464404_[0-53%4]
seal:        16464407 (afterany + explicit 54-fragment completeness gate)
```

本轮仍是 result-blind construction-only：没有创建 factual-B、C、B2 或任何导航评测
job。只有完整 `summary.json` 达到原有 `96 candidates / 15 scenes` reference gate，才允许
另行冻结 v4；不读取 partial candidate count。

## 13. 2026-08-27 direct Natural-B gate 最终通过

54/54 array、seal 和独立 verifier 已全部完成且 exit 0：

```text
scene array: 16464404_[0-53%4]
seal:        16464407
verifier:    16464666
```

最终 sealed summary 与独立 raw-fragment recount 完全一致：

- 130 条 materialized actual-online A；
- 80 条 controlled-Revisit-C constructible histories；
- 61 条 recipients 至少有一个 Natural-B；
- 99 个 Natural-B candidates；
- 35 个 scene clusters；
- 99/35 超过冻结的 96/15 reference gate；
- 最大 online-A covis `0.0897222 < 0.10`；
- 没有读取 query/navigation outcomes，也没有自动取得 evaluation authority。

因此 v4 5-leg population 可以进入“先冻结、再 materialize”的阶段，但尚未运行任何 factual
B/C/B2/C2。完整数字、哈希、独立 verifier 的能力边界和下一道门见
`HM3D_FULLMONO_LIFELONG_NATURAL_B_AUDIT_RESULT_20260827.md`。

## 14. 2026-08-27/28 v4 materialization seal 与独立复核

v4 对全部 54 scenes 完成了 query-asset materialization；原 seal `16465110` 仅因
renderer-free CPU finalizer 间接导入 Habitat `quaternion` 而失败。54 个 GPU fragment
均成功且没有重跑。finalizer 随后改为直接验证并复用 fragment 中已冻结的 contract，
不再导入 renderer-side builder。第一次 replacement `16469893` 又暴露出旧 source-task
parser 不认识 v4 schema；它同样在读取导航结果前失败。

R2 replacement 已完成：

```text
seal:       16470326  COMPLETED
verifier:   16470334  COMPLETED
```

独立 verifier 给出 `verified=true`，并复核：

- 130 条 materialized A histories；
- 99 个 candidate histories、61 个 recipient histories、35 scenes；
- direction strata：front 20 / side 22 / rear 57；
- 396 个 query assets 与 499 个 population-ledger entries 哈希一致；
- runtime role visibility 为 none；
- materialization 阶段未读取 query/navigation outcomes，也未执行 factual B。

最终 factual-B gate 为 `99/35 >= 96/15`，正式授权下一阶段。修复没有改变 protocol、
候选 identity、阈值或任何 GPU materialization 输出。

## 15. 2026-08-28 factual-B 分片提交与 parser-path 修复

99 个冻结候选按 scene 划分为 59 个 shards，每 shard 最多 2 条、并发 4；这样保留同场景
server 复用，又把单 task 时限控制为一小时。schedule SHA-256：
`5b89096c613893a3963d34079b382140d1a8cd4e1fb648968da65f93f6eafbef`。

初始 array `16471189` 的首批 task 在启动模型后确定性失败：collector 由旧 server bundle
的绝对脚本路径执行，Python 将该目录放到 `sys.path[0]`，压过 v4 overlay，因而加载了
只认识 v3 的 parser。状态审计为 4 个同因失败、4 个启动中取消、51 个未启动；
`factual-B completion=0`，没有导航结果污染。旧 runtime 日志全部保留。

修复将 collector、prefix constructor 与 population finalizer 的**字节相同副本**放入 v4
task bundle，仅改变脚本目录与模块解析来源；真实 evaluator、MemNav/NavDP servers 仍来自
原冻结 server bundle。HPC 生产容器分别用 MemNav 与 Habitat Python 验证三个模块均从
v4 task root 解析，5 项测试通过。新不可变 bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  hm3d_lifelong_natural_v4_parserfix_14316838b2bec0c9
SOURCE_BUNDLE.sha256 SHA-256:
  14316838b2bec0c9e2c4714ffc8aae247650aa3c796a75f1ace288e86b1b9d60

replacement factual-B array: 16472222_[0-58%4]
deferred prefix launcher:     16472263 (afterany:16472222)
```

`16472263` 不预占第二个 GPU array。它只在 99 个 factual-B completion 及 sidecar 全部
存在后，顺序提交 99-task prefix construction；随后一个 CPU job 原子完成 population seal
与独立 raw-file verifier。该 verifier 会从原始 B trace/plan、mono-depth receipts、prefix
attrition、复制资产及完整 file ledger 重算 population，不读取 C/B2/C2 outcome。当前没有
formal multileg SR；replacement array 提交时因 `QOSGrpGRES` 等待。
