# 2026-08-20：Full-Mono 完成与 fresh HM3D confirmation 启动

> 23:00 后的 mono-primary 实机与论文重构进展由
> `STATUS_20260820_MONO_PRIMARY_REALWORLD_PAPER.md` 继续记录；本文保留 Full-Mono
> 实验完成和 fresh DAG 提交时的原始总账。

本文是截至 2026-08-20 23:00（Asia/Shanghai）的最新执行总账。冻结 protocol、远端
summary 和 independent verifier 优先级高于本文。

## 1. 今天建立的新结论

HM3D actual-online Full-Mono mixed-role 已完整结束并通过独立复算：

| arm | Novel | Revisit | overall |
|---|---:|---:|---:|
| mono native | 2/8 | 0/8 | 2/16 |
| mono raw fixed | 3/8 | 8/8 | 11/16 |
| mono CEC | 2/8 | 7/8 | 9/16 |

- CEC vs mono native：`+7/-0`，`p=.015625`；
- Revisit：`0/8 -> 7/8`，`+7/-0`，`p=.015625`；
- CEC accept：Novel `0/8`、Revisit `8/8`；
- Novel exact fallback：`8/8`；
- Goal-A/query simulator metric-depth reads：均为 0；
- independent verifier：1,231 个 Goal-A mono receipts、1,809 个 query mono receipts、
  48 条 raw final distances 全部复算一致。

这证明完整系统可以按以下链路闭环运行：

```text
actual causal RGB Goal-A
  -> frozen LingBot first-40 scale + mono NavDP rollout
  -> actual-online causal RGB memory
  -> role-free CEC proof / abstain
  -> mono NavDP query control
```

但它使用了此前 HM3D metric-controller 实验已经出现过的 scene identities，且 CEC 没有
超过 raw fixed（`9/16` vs `11/16`，`+1/-3, p=.625`）。准确口径是“完整 RGB-only
integration 成立”，不是 fresh-scene generalization、mono-metric equivalence 或
CEC-vs-raw superiority。

正式结果：`MemNavData/HM3D_FULLMONO_MIXED_ROLE_RESULT_20260820.md`。

## 2. 为什么下一项必须是 fresh mixed-role

当前证据缺口已经缩成一个明确问题：CEC 在全新外部 scene 上，是否仍能同时保持
Revisit utility 与 Novel abstention。继续 MP3D、重新调 certificate、换 planner 或重训
learned proof 都不能补这个缺口。

新协议从 HM3D val 的 100 scenes 中排除历史已消费的 46 scenes，冻结剩余全部 54 scenes
及 archive 顺序。初始前缀为 30 scenes；若在任何 query arm 运行前，shared Goal-A 与
role-pair 构造未达到 24 histories / 15 scene clusters，则按固定 6-scene block 扩至
36/42/48/54。扩展只能看共享 pre-query constructibility，不能看 native/raw/CEC SR。

协议：

- `MemNavData/HM3D_FRESH_FULLMONO_MIXED_ROLE_PROTOCOL_20260820.md`
- `MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json`

Scene selection 已由
`MemNavData/audit_hm3d_fresh_fullmono_selection.py` 从 archive member list、36-scene
consumed audit 和旧 heldout10 protocol 独立重算通过：`100 - 46 = 54`，无交集。

## 3. 新正式 DAG

Run root：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
```

Immutable bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fresh_fullmono_mixed_role_e6dd44c66eb72d90
```

Bundle receipt SHA-256：
`e6dd44c66eb72d905f4db96db9f604c042cc0d89c77a16cd4dc92eabb99c8f01`。

| stage | job ID | status at handoff |
|---|---:|---|
| asset prepare | 16080295 | completed 0:0，32 s |
| 54-scene source generation | 16080301 | running，max 2 GPUs by current QOS |
| sealed parent manifest | 16080319 | dependency |
| actual mono Goal-A | 16080320 | dependency |
| role-pair construction | 16080332 | dependency |
| frozen prefix finalizer | 16080338 | dependency |
| 80-step smoke | 16080339 | dependency |
| formal paired query | 16080342 | dependency |
| summary | 16080348 | dependency |
| independent verifier | 16080360 | dependency |

Task `16080301_1` 已在 L40S 上 4/4 生成成功；task 0 正常运行。其余 pending reason 是
`QOSMaxGRESPerUser`，表示当前账户只获得两张并发 GPU，不是代码或 controller 故障。

## 4. 代码变化与验证

新增：

- fresh-scene selection auditor 与测试；
- 54-scene asset extraction；
- fixed-attempt source generation 与显式 scene-level attrition；
- pre-navigation parent manifest builder；
- fixed-prefix population selector；
- fresh Full-Mono submit DAG。

泛化后的旧 Full-Mono 代码仍通过旧 9-scene regression：

- MemNav Python：22 tests passed；
- Habitat Python：15 tests passed；
- exact remote container：17 tests passed，1 environment-dependent test skipped；
- JSON、`py_compile`、全部 shell `bash -n` 通过。

CEC threshold、proposal order、2.5 m residual、NavDP checkpoint、LingBot checkpoint、
成功半径和 step budget 均未改变。

## 5. 基础设施事件

科学 DAG 提交完成后，第一次 receipt upload 使用了错误的 `scp -S SOCKET` 语义；scp
把 `-S` 当作连接程序而不是 control socket。没有重提任何 job。随后用
`-o ControlPath=SOCKET` 补传同一份 receipt，远端 SHA-256 为
`16b0c68d1fcb4349b519efa258e2848dbf8ffe66f83888c36234265e87cbba68`。

事件：
`MemNavData/HM3D_FRESH_FULLMONO_SUBMISSION_RECEIPT_UPLOAD_INCIDENT_20260820.json`；
操作手册已补充该陷阱。

## 6. 当前论文口径

现在允许写：

- CEC 是 proof-before-control / open-set action authorization；
- 完整 causal RGB-only integration 已在 reused-scene HM3D 上成立；
- fresh-scene confirmation 已 prospective freeze 并提交，但结果尚未产生。

仍禁止写：

- fresh Full-Mono 已确认；
- CEC 显著优于 raw memory；
- mono 与 metric RGB-D non-inferior；
- certificate formal safety / zero false accepts；
- Novel ImageGoal 已解决。

## 7. 下一动作

不读取 partial SR、不调方法。只监控 stage completion、exact pending reason、生成 attrition
和 runtime failure。最终必须等 summary 与 independent verifier 都成功后，才把 fresh
结果写入论文。若某个 task 是 infrastructure failure，保留 partial output 并 exact-index
repair；fixed-attempt constructibility attrition 则按冻结协议保留，不重试。
