# CEC proof-locked executor portability protocol

日期：2026-08-27（Asia/Shanghai）
状态：**设计与本机真栈接口门已冻结；尚未产生 Fresh-HM3D controller utility 结果。**

## 1. 为什么需要重做，而不是继续旧 ViNT multileg

旧 controller-portability 矩阵的直觉是对的：CEC 应该是一个可移植的 memory
authorization interface，而不应只绑定 NavDP。问题出在实验单位。

旧 multileg 设计让每个 controller 独立执行 C，再在 B2 比较。Controller 一旦改变 C
轨迹，B2 起点和进入 B2 的历史也随之改变，因此不能再把 B2 差异归因于“同一个 CEC
proof 由谁执行”。严格 finalizer 正是因此拒绝旧 270-cell 表。修正后的 shared-C 只留下
ViNT `4 histories / 1 scene`，三臂均 `4/4`；它证明接口运行，但没有统计 utility。

新实验不再依赖 multileg，也不要求分叉后的 proof stream 相同。完整 rollout 中，不同
controller 观察到不同后续图像，本来就应该产生不同的后续 proof；强制它们相同反而违反
因果性。

## 2. 新的科学问题

正式问题分成两个互补层次：

1. **proof identity / execution isolation**：在完全相同的历史、目标和查询起点上，同一个
   首次 CEC proof 能否无歧义地投影到不同 controller 的原生接口？
2. **conditional utility portability**：当相同的首次 proof 被授权后，不同 frozen
   controller 能否相对“看见相同 proof 但不给控制权”的 shared-native baseline 获得闭环
   utility？

第一层证明接口与归因，第二层才证明这个接口不是只会连线。

## 3. 为什么使用 Fresh HM3D，而不是 Final14 或 B2

冻结来源：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_fresh_fullmono_mixed_role_20260820/
  formal_20260820T143609Z_e6dd44c6
```

该 population 有 `28 histories / 21 scenes / 56 queries`，历史来自 actual mono Goal-A，
不是 expert trace。原 independent verifier 已通过。

更关键的是，只读取 CEC plan receipt、完全不读取 navigation outcome 后可复算：

- 28 个 query 的首次 certificate 为 accept；
- 28 个 query 的首次 certificate 为 reject；
- accept 全部发生在 query 第一个 decision，且首次均为 uncached；
- 每条 query 的后续 cached decision 均未翻转；
- 事后分析显示 accepted set 恰好是 28 个 Revisit，rejected set 恰好是 28 个 Novel。

因此它天然提供一个 deployment-visible accepted set。运行时 manifest 只列
`scene/episode/query_id` 和 proof/hash binding，**不含 Novel/Revisit role**。Role 只存在于
独立 audit receipt。

这是已消费外部场景上的 controller ablation，不冒充新的 fresh-scene confirmation。它的
价值是归因干净、分母足够、全单目且不需要重新生成 A。

## 4. Certified Handoff Packet

一次 accept 被封装为单次使用的 `Certified Handoff Packet`：

```text
ordered causal-history digest
current RGB digest
target ImageGoal digest
selected anchor index + anchor JPEG digest
atomic certificate + proof digest
scale-free [forward, left]
fixed-radius contract = 2.5 m
```

Packet 不含 role、Habitat pose、GT distance 或 oracle scale。它绑定当前 observation，因此只
能授权**一个高层 controller decision**；controller 移动以后若还要接管，必须基于新的当前
图像产生新的 CEC proof。这样避免把旧 robot-centric bearing 当成世界坐标重复使用。

运行时 causal-history digest 不是固定复用 A 段哈希：首个 decision 以 sealed online-A trace
SHA 为起点，之后按 `prior history + current RGB + active goal + decision index` 递推。两臂在
首个动作前必须得到同一个 packet；轨迹分叉后 history chain 和后续 packet 应随真实观测分叉。

三种主投影：

| executor | packet projection | 含义 |
|---|---|---|
| NavDP | original ImageGoal + 2.5 m bearing mixed goal | 当前论文执行器 reference |
| ViNT | SHA-bound certified history anchor ImageGoal | 视觉 goal-controller family |
| iPlanner | 2.5 m PointGoal + read-only LingBot mono depth | point-goal local-planner family |

这三臂不能被描述成“完全相同输入”：它们消费同一个 proof，但通过预先声明的原生投影消费
不同 payload。ViNT 不能被伪装成 PointGoal controller，iPlanner 也不能被伪装成原生
ImageGoal controller。

GNM、NoMaD、ViPlanner 保留为 secondary expansion。Primary triad 没有建立 utility 前，
不靠增加相近 controller 数量制造一张大表。

## 5. 同进程 paired treatment

每个 `(history, controller)` 在一张 GPU、同一组已加载进程里顺序运行两臂：

```text
grant:
  CEC accept -> controller-native projection
  CEC reject -> shared mono NavDP

forced_reject_native:
  仍运行完全相同的 retrieval/certificate/proof receipt
  即使 proof accept 也不给动作权
  每个 decision -> shared mono NavDP
```

两臂共享：

- 同一 actual-online A RGB replay；
- 同一 query start、goal、seed、600-step budget 和到达判据；
- 同一 MemNav、mono NavDP、alternate controller 与 proxy 进程；
- 同一 GPU UUID；
- 同一 first proof SHA、anchor 和 goal boundary。

两臂只使用两个不可变 hub：一个永远 grant，一个永远 force reject。顺序按 history/controller
哈希平衡。该设计既避免运行中切 flag，也避免跨机器 CUDA 差异。

首次 grant 后允许轨迹分叉。Verifier **只要求分叉前的 first proof 完全相同**；不再错误地
要求后续 proof stream 相同。

## 6. 执行阶段

### Stage 0：已完成的本机真栈门

Consumed MP3D `gxdoqLR6rwA/episode_0000`，每臂 8 steps：

| controller | first proof | anchor | same-process pair | 8-step SR | 结论 |
|---|---|---:|---|---|---|
| ViNT | `c287e710f320...` | 121 | pass | `0 vs 0` | visual-anchor contract pass；无 utility 结论 |
| iPlanner | `c287e710f320...` | 121 | pass | `0 vs 0` | bearing-pointgoal contract pass；无 utility 结论 |

两种 controller 的 first proof 与 anchor 也完全相同。两条 grant 在 8 steps 后都仍未成功，
且该单 episode 距离没有改善，所以不能把 smoke 写成性能结果。

### Stage 1：Fresh HM3D 四场景基础设施 pilot

固定 indices `0, 7, 14, 21`，对应四个不同 scene；三种 primary controller；每个 controller
同进程运行 grant/forced 两臂。Pilot 只决定基础设施能否进入 full population，不按 SR 选择
controller，也不调 radius、budget 或 checkpoint。

必须全部通过：

- first proof/anchor 在 paired arms 中完全一致；
- cross-controller first proof identity 一致；
- cross-controller first packet SHA 一致；
- no role / no metric-depth / no runtime failure；
- alternate controller、proxy、MemNav 与 NavDP process receipt 完整；
- 600-step job 在冻结时限内结束。

### Stage 2：完整 accepted-set utility

若 Stage 1 合约通过，固定运行全部 `28 histories / 21 scenes`。不按 pilot SR 删 controller。

每个 controller 报告：

- grant 与 forced SR、SPL、final distance、path length、collision；
- paired gain/loss、exact McNemar；
- scene-cluster bootstrap 95% CI；
- first-handoff 与全 rollout latency；
- takeover/fallback coverage；
- 对现有 CEC->NavDP reference 的描述性差异。

Primary promotion rule：至少一个非 NavDP executor 必须相对自己的 same-process forced
baseline 得到正向、统计可辨认的闭环 utility，且无合约失败，才允许写“utility transfers
across controller families”。若只通过接口而无 SR 增益，论文只能写“proof projection is
implementable”，不能写 controller-agnostic effectiveness。

## 7. Novel safety 的处理

不为每个 controller 重跑 28 条长 Novel rollout。原因不是隐藏失败，而是 source receipt 已
证明 28/28 first reject 且决策不翻转；reject 分支根本不会调用 alternate controller。

Formal bundle 仍会对每种 controller 运行至少两个 Novel exact-fallback smoke，核验：

- zero alternate takeover；
- shared mono NavDP action/seed receipt；
- short-context shadow 对 ViNT 正常；
- trajectory hash 与 forced baseline 一致。

Novel 的统计安全/效用主证据仍来自原 56-query mixed-role Fresh-HM3D experiment，而不是
本 conditional accepted-set ablation。

## 8. 允许与禁止的论文结论

若完整结果通过，允许：

> CEC exposes a proof-carrying episodic handoff whose accepted evidence can be
> projected to heterogeneous frozen executors, while rejection retains the
> same monocular ImageGoal fallback.

仍禁止：

- 所有 controller 接收完全相同 tensor/token；
- CEC 对任何任意 controller 都有效；
- iPlanner/ViNT 在原生 ImageGoal 上优于 NavDP；
- 8-step smoke 是 SR 结果；
- consumed accepted-set ablation 是新的 cross-dataset confirmation。

## 9. 当前实现

- packet contract：`MemNavData/cec_handoff_contract.py`；
- packet tests：`MemNavData/test_cec_handoff_contract.py`；
- accepted-set freezer：`MemNavData/build_cec_accepted_query_manifest.py`；
- same-process authority-pair audit：`MemNavData/audit_cec_authority_pair.py`；
- paired runner：`MemNavData/run_cec_controller_portability_smoke_local.sh`；
- current regression：相关 contract/hub/proxy tests `98 passed`；新增 manifest tests另有
  `2 passed`。

下一步不是直接扩成六种 controller：先在远端从 56 条 source proof receipt 生成并独立验证
role-free accepted manifest，再提交四场景、三 controller、same-process paired pilot。
