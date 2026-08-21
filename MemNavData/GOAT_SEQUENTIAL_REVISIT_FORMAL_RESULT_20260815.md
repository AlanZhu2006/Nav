# GOAT Sequential-Revisit Formal Result — 2026-08-15

## 1. 一句话结论

34-scene frozen GOAT external evaluation 完整通过运行与独立复算，但 CEC 与 official GOAT
完全持平：`4/34` vs `4/34`、paired `+0/-0`、McNemar `p=1.0`。

更重要的是，post-hoc actionability audit 证明这不是一个可解释为“CEC 干预无效”的普通
null：CEC 在 34 条中没有执行过一次 motion override。5 次 certificate accept 全部发生在
official policy 已输出 `SUBTASK_STOP` 的那一步；冻结协议要求原样保留 STOP，因此
NavDP plan、非 STOP actionable accept 和实际 override 均为 0。两臂 34/34 的动作和位姿
逐步完全相同。

最准确的分类是：

> `degenerate_noop_no_executed_intervention`：当前 GOAT population 与 exact-STOP contract
> 下的部署可行动性失败，而不是 bearing utility 的有效反证。

## 2. 冻结评测与完整性

- 34 个 HM3D `val_unseen` scenes，各一条 outcome-blind 冻结的 exact repeated-ImageGoal
  episode；
- official released stochastic CUDA policy，seed 100；
- native / CEC 严格配对，arm order 17/17 平衡；
- 每臂最多 5000 actions；
- CEC 只可在 certificate accept 后覆盖非 STOP motion；
- official `SUBTASK_STOP` 必须原样执行；
- controller 不读取 evaluator 的 Revisit role 标签；
- 34 个 GPU tasks 全部 `COMPLETED 0:0`；
- 34 raw JSON + 34 SHA sidecars 齐全；
- summary 与 independent verifier 均 `COMPLETED 0:0`；
- independent verifier：`verified=true`、`raw_result_count=34`。

审计字段：

- complete results：34/34；distinct scenes：34/34；
- native-first：17；CEC-first：17；
- both arms entered target：34/34；
- all prefixes paired：true；
- all official subtask stops preserved：true；
- no-accept exact fallback：true；
- GOAT commit：`74c41d19d4a4c3608d1575b512087b5a529aee0e`；
- checkpoint SHA-256：
  `55e89c3d083198d4add4e9e70164b54ff892900963a2925471362e2d4761b3eb`；
- formal manifest SHA-256：
  `aaedc6fb0c6d3787b5c8c61eed2c2d943320f595f9b1783f881febc544121397`。

## 3. Confirmatory outcome

| arm | target success | rate |
|---|---:|---:|
| official GOAT | 4/34 | 11.76% |
| role-free CEC | 4/34 | 11.76% |

配对统计：

- gains：0；
- losses：0；
- risk difference：0.0 pp；
- scene-cluster bootstrap 95% CI：`[0.0, 0.0] pp`；
- exact two-sided McNemar：`p=1.0`。

以上是这个 targeted exact-recurrence stratum 的结果，不是 full GOAT benchmark score，不能
与 GOAT 论文公布的 overall SR/SPL 横向相减。

描述性分层同样无差异：

- description→image：native `1/18`，CEC `1/18`；
- image→image：native `3/18`，CEC `3/18`。

两个分层有 2 条 episode 重叠，因为目标 instance 之前同时以 description 和 image 出现；
因此分层 n 不能相加作为总分母。

## 4. 覆盖与安全诊断

- target candidate-supported episodes：20/34；
- target certificate-accepted episodes/events：4/34、4 events；
- pre-target nonrecurrent accepts：1 episode、1 event；
- 总 certificate accept：5 events。

target rejection event counts：

| reason | count |
|---|---:|
| no causal candidate | 550 |
| precheck fundamental inliers | 201 |
| precheck query hull coverage | 6 |
| precheck reference hull coverage | 4 |
| minimum inliers | 4 |
| minimum reference coverage | 1 |
| status ok but not accepted | 1 |

1 次 pre-target accept 没有造成行为损失，因为它同样发生在 official `SUBTASK_STOP`，执行
动作未被替换。但它说明 certificate 不能被描述为完美的 semantic Novel/Revisit classifier。

## 5. Actionability audit：为什么主结果必然完全相同

该审计是读完 formal 后新增的 post-hoc 机制审计，明确标记为
`posthoc_intervention_audit_not_preregistered=true`，不能用于调方法或阈值。它逐条重读 34
个 raw JSON，并验证每个 SHA sidecar。

结果：

| actionability quantity | result |
|---|---:|
| certificate accept events | 5 |
| accepts on official `SUBTASK_STOP` | 5 |
| actionable non-STOP accepts | 0 |
| NavDP plans | 0 |
| executed override events | 0 |
| episodes with first override | 0 |
| native/CEC action+pose exact episodes | 34/34 |

四次 target accept 分别发生在 formal index `2/10/21/30`，一次 pre-target accept 在 index
`18`；五次的 `action_source` 均为 `official_goat_subtask_stop`，`navdp_plan_present=false`。

因此 confirmatory summary 中的 `mechanistic_coverage_gate_passed=true` 暴露了一个协议设计
缺口：冻结 gate 只要求 candidate/certificate coverage，没有要求“非 STOP accept + 实际
motion override”。它证明 observation-level constructibility，却没有证明 control-level
actionability。这个缺口不能在已消费的 34 scenes 上事后改 gate 来制造新的确认结果。

## 6. 科学解释

### 已建立

- 官方 GOAT CUDA policy、真实 causal history、CEC sidecar、NavDP service、严格 arm pairing
  和完整写盘链已经工程上打通；
- 34 条都真正进入 repeated ImageGoal target，不是 population 构造为空；
- reject 时 exact fallback 在外部 HM3D runtime 中成立；
- strict geometry certificate 在这个分布上的可执行覆盖严重不足；
- 当前 certificate 更像一个局部、高共视 relocalizer，而不是能提前提供长程方向的通用
  episodic compass。

### 未建立

- CEC 能提升 official GOAT sequential-Revisit success；
- bearing 在 GOAT 上无效；本实验没有执行 bearing intervention，不能作该反推；
- certificate 能稳定地在需要导航时早于 STOP 接管；
- GOAT external result 能作为论文正向主结果。

### 结构性原因

严格几何 certificate 需要足够视角共视，证据往往在 agent 已到达相似历史视图、official
policy 已准备 STOP 时才最强；而冻结协议又禁止 CEC 延迟或替换 STOP。这形成了一个时间上的
矛盾：证据可靠时已经失去 motion-control 窗口。20/34 有候选但只有 4 条 target accept，且
全部在 STOP，支持“覆盖/时机错配”而非“已执行方向无效”。

## 7. 对论文和下一步的约束

1. 不在这 34 条上降低 certificate 阈值、改 STOP 契约、筛 episode 或重跑新方法；它们已
   完整消费。
2. GOAT 结果应作为诚实的 external stress-test / limitation，而不是正向 benchmark gain。
3. 论文正向主证据仍是 MP3D 上的 actual-online Revisit utility；certificate 的已证价值是
   role-free abstention / exact fallback，而不是已经证明的 SR 上限提升。
4. 若未来继续 GOAT，新的预注册 constructibility gate 必须包含：
   - 至少若干 target accept 发生在 official non-STOP motion；
   - 至少若干 episode 实际生成 NavDP plan；
   - 至少若干 episode 出现 executed motion override；
   - population 具有非平凡时间/空间 Revisit 间隔。
5. “让 certificate veto STOP”是一个新的 stop-correction 方法，不是本方法的小修；它会改变
   exact official-stop safety contract。必须先在训练/开发 population 验证，再去新的未见
   population，不能用本次两个 accepted-failure 案例作为调参依据。

## 8. 不可变结果路径与哈希

Formal root：

`/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815/formal_20260815T134809Z`

- summary：`goat_sequential_revisit_summary.json`
  - SHA-256：`18eb758bff6dfe59eee5cb43239ca01c607e790cd34ce464e50c6491d321268a`
- independent verifier：`goat_sequential_revisit_independent_verification.json`
  - SHA-256：`3cd8d7f3d57cdb912ad39198aa40b7d38a26c5db9ef9461252cab7f481d42945`
- post-hoc actionability audit：`goat_sequential_revisit_actionability_audit.json`
  - SHA-256：`1b312befac206e2fd517fb826f9738d28503bf488cc20c770599c709184baddf`
- actionability audit source：
  `posthoc/actionability_audit_ac06b86876900903.py`
  - SHA-256：`ac06b86876900903a9f3fe336dba99db6eaad4328f5ba69b70c2100ac77ee0c9`

Local reproducible audit source：

`MemNavData/audit_goat_sequential_revisit_actionability.py`

