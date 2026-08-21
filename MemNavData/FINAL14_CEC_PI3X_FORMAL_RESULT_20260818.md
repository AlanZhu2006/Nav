# Final14 CEC 与 Pi3X Learned Relocalizer 正式结果

更新时间：2026-08-18（Asia/Shanghai）  
状态：**42/42 完成，正式 summary 完成，independent verifier 通过**。

## 1. 一句话结论

Final14 支持把 **Certified Episodic Compass（CEC）**保留为论文主方法：它在不读取
Novel/Revisit role 的条件下，将 standard Revisit SR 从 native 的 `4/21` 提高到
`20/21`，并在 role-balanced natural protocol 上显著超过 always-on raw-DINO fixed
bearing（`28/42` 对 `21/42`，配对 `+8/-1`，McNemar `p=0.0391`）。

CEC 的可识别价值不是在已知 Revisit 上继续大幅提高方向上限，而是
**proof-before-control**：保留 Revisit utility，同时减少 unsupported Novel 上的错误
历史接管。

Pi3X learned arm 达到 `19/21` Revisit，显著超过 native，但没有通过预注册的
non-inferiority 与 Novel proof-safety 门，因此不能替代 CEC，也不能升为论文 primary。

## 2. 完整性与可复现性

- natural-direction：`21` histories、`10` scene clusters、每条包含独立 Novel 与
  standard Revisit query；
- hard-support：`21` histories、`10` scene clusters；正式 analysis role 仅为
  Revisit，配套 Novel 只用于 instrumentation；
- 五臂严格配对：`native / raw_fixed_bearing / geometry_fixed / certified /
  learned_pi3x_spatial`；
- runtime role visibility：`none`；
- 每条 episode 的所有 arm 共享 Goal-A replay、RGB、ImageGoal、NavDP checkpoint、
  deterministic diffusion seeds、预算和 success criterion；
- `42/42` completion receipts 均存在并通过各自 SHA-256；
- summary job `15903547`：`COMPLETED 0:0`；
- independent verifier job `15903548`：`COMPLETED 0:0`；
- independent verifier：`verified=true`；
- summary SHA-256：
  `ab704752abcf624aebd9a598c80659995a8b443d6e7fb0e7944554b8ae320f07`；
- verification SHA-256：
  `268a104a64a0d9a040010646abc046f1d29323b17705ca4781ea9f10073d5318`。

本地封存目录：

```text
.diagnostics/learned_relocalizer_20260817/
  final14_attempt7_formal_result_20260818/
```

远端权威目录：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  final14_cec_learned_20260817/
  final14_learned_20260817T115533Z_attempt7_handoff/
```

## 3. Population 与统计功效边界

冻结协议从 `14` 个 untouched Final14 MP3D scenes 开始：

- Goal-A successes：`47`；
- materialized histories：`47`；
- 同时满足 standard Revisit 与 natural Novel 合约的 histories：`21`；
- 有效 scene clusters：`10`；
- standard/natural 预注册目标：`28 histories / 10 scenes`，实际
  `21 / 10`，history target 未满足；
- hard-support 目标：`16 histories / 8 scenes`，实际 `21 / 10`，目标满足。

因此：完整运行和统计复算都有效，但 natural primary population 必须明确标为
**underpowered relative to the frozen 28-history target**。协议禁止在打开结果后追加
episode、放宽构造条件或补入其他 MP3D scenes。

这不抹去已经观察到的配对效应，但限制了措辞：不能把本次写成达到全部预注册功效的
最终确认，也不能用后验补样修复该标签。

## 4. Natural-direction 主协议

### 4.1 SR 与 SPL

| arm | Novel SR | Revisit SR | 两类 query 合计 SR | 合计 SPL |
|---|---:|---:|---:|---:|
| native | `7/21 = 33.33%` | `4/21 = 19.05%` | `11/42 = 26.19%` | `0.1217` |
| raw fixed | `2/21 = 9.52%` | `19/21 = 90.48%` | `21/42 = 50.00%` | `0.3509` |
| geometry fixed | `9/21 = 42.86%` | `18/21 = 85.71%` | `27/42 = 64.29%` | `0.4145` |
| learned Pi3X spatial | `8/21 = 38.10%` | `19/21 = 90.48%` | `27/42 = 64.29%` | `0.4249` |
| **CEC** | **`8/21 = 38.10%`** | **`20/21 = 95.24%`** | **`28/42 = 66.67%`** | **`0.4400`** |

“合计”只是同一 frozen population 内的 role-balanced query 汇总，不是 2-leg joint
success，也不能替代 Novel/Revisit 分表。

### 4.2 CEC 对 native

| estimand | paired gain/loss | 风险差 | scene-cluster 95% CI | exact McNemar |
|---|---:|---:|---:|---:|
| Revisit | `+16/-0` | `+76.19 pp` | `[+55.56,+94.74] pp` | `p=3.05e-5` |
| Novel | `+1/-0` | `+4.76 pp` | `[0,+14.29] pp` | `p=1.0` |
| 两类 query 合计 | `+17/-0` | `+40.48 pp` | `[+28.13,+52.17] pp` | `p=1.53e-5` |

CEC 的 17 条合计增益没有任何配对损失，其中 16 条来自 Revisit。这是因果在线历史对
冻结 NavDP 有控制价值的 fresh closed-loop 证据。

### 4.3 CEC 对 raw fixed

| estimand | paired gain/loss | 风险差 | scene-cluster 95% CI | exact McNemar |
|---|---:|---:|---:|---:|
| Revisit | `+1/-0` | `+4.76 pp` | `[0,+15.79] pp` | `p=1.0` |
| Novel | `+7/-1` | `+28.57 pp` | `[+4.35,+53.33] pp` | `p=0.0703` |
| 两类 query 合计 | `+8/-1` | `+16.67 pp` | `[+2.78,+31.25] pp` | `p=0.0391` |

这第一次在 fresh mixed-role population 上建立了 CEC 相对简单 raw-DINO bearing 的
显著整体优势。分角色结果说明增益来自风险控制，而不是 Revisit ceiling：raw 已有
`19/21` Revisit，但在 Novel 上只有 `2/21`；CEC 保留 `20/21` Revisit，同时将 Novel
恢复到 `8/21`，与 native 的 `7/21` 同量级。

### 4.4 CEC 对旧 geometry

- 两类 query 合计：`28/42` 对 `27/42`，配对 `+2/-1`，`p=1.0`；
- Revisit：`20/21` 对 `18/21`，`+2/-0`，`p=0.5`；
- Novel：`8/21` 对 `9/21`，`+0/-1`，`p=1.0`。

CEC 没有证明显著超过旧 geometry。论文不能声称每个新几何部件都带来独立 SR 增益；
应该强调更清楚的 atomic proof、scale-free output、exact fallback 和 open-set
risk--coverage contract。

## 5. CEC 的开放集授权行为

Natural Novel 共 `21` 条：

- 完全拒绝：`19/21`；
- 这 19 条全部与 native requested/returned seeds、selected trajectory 和 executed
  trace 精确一致；
- certificate accept/takeover：`2/21`；
- Novel 相对 native：`+1/-0`；
- runtime failure plans：`0`。

因此不能写“CEC 在 Novel 上零接管”或“certificate 是形式化安全保证”。能写的是：

> 在 fresh mixed-role MP3D population 上，CEC 对 19/21 unsupported Novel queries
> 完全拒绝并精确回退；两次接管没有造成配对净损失。

## 6. Hard-support 诊断协议

Hard-support 的正式 analysis role 只有 Revisit；Novel 结果是重复 instrumentation，不能
与 Natural Novel 合并或作为第二份 Novel 证据。

| arm | hard-support Revisit SR | SPL |
|---|---:|---:|
| native | `3/21 = 14.29%` | `0.0396` |
| raw fixed | `19/21 = 90.48%` | `0.6462` |
| geometry fixed | `19/21 = 90.48%` | `0.5292` |
| **CEC** | **`19/21 = 90.48%`** | `0.6091` |
| learned Pi3X spatial | `18/21 = 85.71%` | `0.6051` |

- CEC 对 native：`+16/-0`，风险差 `+76.19 pp`，scene-cluster CI
  `[+61.11,+90.00] pp`，`p=3.05e-5`；
- CEC 对 raw：`+0/-0`；
- CEC 对 geometry：`+1/-1`；
- learned 对 CEC：`+1/-2`，净少一条。

这进一步说明：当历史支持被构造保证时，raw direction 已接近上限；CEC 的独特价值在
unsupported/open-set 情况下决定是否允许历史改变控制。

## 7. Learned Pi3X 正式判定

### 7.1 闭环结果

Natural standard Revisit：

- learned：`19/21`；
- native：`4/21`；
- paired `+15/-0`；
- 风险差 `+71.43 pp`；
- scene-cluster CI `[+47.82,+90.48] pp`；
- `p=6.10e-5`。

Learned arm 有真实 utility，不是失败到不可用。但相对 CEC：

- Revisit：`19/21` 对 `20/21`，`+0/-1`；
- Novel：两者完全相同，均为 `8/21`；
- 合计：`27/42` 对 `28/42`，`+0/-1`。

### 7.2 预注册 qualification

| gate | 结果 | 原因 |
|---|---|---|
| L1：对 native 有用 | **通过** | 正净增益且 scene-cluster CI 下界大于零 |
| L2：对 CEC non-inferior | **未通过** | 点估计 `-4.76 pp`，但 cluster CI 下界低于冻结的 `-10 pp` margin |
| L3：Novel safety/exact fallback | **未通过** | fallback/runtime 条件通过，但 accepted bearings 中存在 `>90°` catastrophic errors |
| primary promotion | **未通过** | L1--L3 未全部通过，且 natural population target 未满足 |

Natural protocol 的 learned accepted bearing audit：

- accepted plan bearings：`479`；
- median error：`8.05°`；
- p95：`149.52°`；
- `>90°`：`69`；
- runtime failures：`0`；
- 完全 abstain 且 exact-native queries：`19`。

这说明 learned proof 可以产生很强闭环效用，但尚不能稳定区分方向正确与灾难方向。
不能因为 controller 最终成功而把错误的 proof 解释为可靠重定位。

正式决策：**CEC 继续作为 primary；Pi3X learned relocalizer 降为 promising but
unqualified learned simplification / negative result。**不得在 Final14 failures 上调
checkpoint、threshold、consensus 或 bridge density 后再次把同一 population 当测试集。

## 8. 延迟与部署边界

对冻结 raw plans 做只读 cache-lifecycle 复算后，Natural protocol 的正确口径为：

- CEC 每个 query 的首次 uncached certificate：`n=42`，median 约 `3.40 s`，p95
  约 `26.35 s`；
- CEC 真正的后续 cached bearing update：`n=1346`，median 约 `0.152 ms`，p95
  约 `0.428 ms`；
- learned first query median：约 `2.69 s`，p95 约 `3.39 s`；
- learned one-anchor update median：约 `349 ms`，p95 约 `500 ms`；
- learned peak allocated GPU memory：约 `28 GB`；
- stored online history：median `166` frames，最大 `364` frames。

原正式 summarizer 按字段非空收集 CEC latency；由于 cache hit 会保留首次
`uncached_relocalization_ms` receipt，它把 42 个首次值随 replans 重复为 1388 个值，
得到的 `1.46 s` median 是 replan-weighted，不是 query-level latency。该问题只影响延迟
口径，不影响 SR、SPL、配对统计、certificate 决策或 independent verification；正式
summary 文件不被事后重写。完整审计见
`MemNavData/FINAL14_CEC_CACHE_LATENCY_AUDIT_20260818.md`。

selected anchor index 与首次 latency 的相关为 `r=0.817`；Revisit 首次 median 为
`10.64 s`，Novel 为 `0.817 s`。代码路径审计表明，主要长尾来自为较晚 anchor 顺序
重放 dense causal LingBot depth；后续 pose-to-bearing update 已经近乎免费。

这些数字证明当前实现可运行，但还不是最终实时系统。下一项合理的工程研究不是改变
certificate，而是把历史帧的 DINO、局部特征和单目 depth 因果地预计算/缓存，并用
decision-equivalence test 证明 proposal、certificate、bearing 和动作不变。

## 9. 对论文主张的影响

### 已支持

1. causal actual-online visual history 可显著恢复 frozen NavDP 的 Revisit failures；
2. 只输出 scale-free bearing 的 residual interface 足以把记忆转为闭环控制收益；
3. always-on high-coverage memory 会干扰 unsupported Novel；
4. proof-before-control 在 mixed-role population 上形成更好的 utility--interference
   权衡，并显著超过 raw fixed 的 role-balanced aggregate；
5. runtime 不需要显式 Novel/Revisit classifier 或 role label。

### 未支持

1. CEC 形式化保证 Novel safety；
2. CEC 在 Revisit SR 上显著超过 raw 或旧 geometry；
3. learned proof 已替代显式 certificate；
4. Pi3X accepted bearing 已达到可靠部署标准；
5. Final14 达到了 natural protocol 的全部预注册 population power；
6. Novel direction 本身已经解决。

## 10. 冻结后的下一步

1. 不再打开 Final14 做模型选择；
2. 将本结果、HM3D external Revisit、actual-online 3-leg 和既有负结果整理为论文主表；
3. 生成 Revisit utility 对 Novel interference 的 risk--coverage 图；
4. 先做 CEC 在线 feature/depth cache 的 decision-equivalence microbenchmark；只有
   proposal、certificate、bearing 和动作 `0` mismatch 后才报告延迟与 memory scaling；
5. 只有论文需要“open-set safety 跨域”主张时，才冻结新的 HM3D mixed-role 协议；
   不能把已经完成的 HM3D Revisit-only 评测说成尚未运行；
6. 若继续 learned route，必须扩展独立训练 scenes、重新定义 catastrophic-bearing
   proof target，并在全新 untouched test 上 prospective evaluation。
