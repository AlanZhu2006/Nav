# CEC 持续记忆与五段导航协议（2026-08-21）

## 1. 当前系统到底记什么

CEC 的长期状态不是“只保存 Novel A”。一次 episode 内，每个实际观察到的 RGB 都按时间写入同一条因果历史：

```text
goal A rollout ─┐
goal B rollout ─┼─> persistent causal RGB / DINO / LingBot history
goal C rollout ─┤
later goals  ───┘
```

一个新 goal 第一次查询时冻结 `goal_start_frame`，该 goal 的候选上限为 `goal_start_frame - 1`。因此：

- 当前 goal 之后产生的帧不能被它自己检索，避免“边走边把自己当 Revisit”的泄漏；
- 下一个 goal 可以读取此前所有已完成 goal 的实际在线轨迹；
- full episode reset 才清空长期历史；普通 goal switch 不清空长期历史；
- NavDP/ViNT 的短时 controller context 与 CEC 长期历史是两个不同状态。

旧 double-Revisit 的 `initial_leg_only` 是因果消融：它故意把 C 的候选上限压到 online-A 末尾，用于隔离“保留更早 A 记忆”的作用。这不能代表部署时默认只记录 A。部署默认是 `all_prior`。

## 2. 本次审计发现并修复的 4+ leg 问题

原实现的 goal-conditioned cache 只用目标 JPEG 哈希索引。`A -> B -> A` 中第二次 A 会复用第一次 A 的 `goal_start_frame`，因而无法读取第一次 A 之后新增的在线历史。2/3-leg 的目标 JPEG 各不相同，不会触发这个问题，所以已有结果不受影响；重复目标的多-leg 评测会受影响。

修复后的生命周期是：

- 因果 RGB、DINO、LingBot pose/KV 与 anchor-depth cache 跨 goal 保留；
- 每次 goal hash 发生切换，开启新的 `goal_session_index`；
- goal-specific shortlist、candidate ceiling、certificate、sticky anchor、graph route 和 learned proposal cache 在 session 边界失效；
- 同一个 goal session 内仍保持 frozen shortlist/certificate 语义；
- 同一目标离开后再次出现，会以当时的完整历史建立新的因果候选集合。

新增逐动作 receipt：

- `cec_goal_session_expected_start`；
- `cec_goal_session_started`；
- `cec_goal_session_index`；
- `cec_goal_start_frame`；
- `cec_candidate_ceiling`；
- `cec_long_term_memory_preserved`。

相关契约测试目前为 **96 passed**。

## 3. strict-v4 五段序列：只作 lifecycle smoke

直接复用严格 `multileg_v4_role_paired_20260812` 三段资产，执行：

```text
A -> B -> C -> B -> C
```

后验解释为：

1. A：初始 ImageGoal；
2. B：相对 online-A 的 Natural Novel；
3. C：对 A 历史的长程 Revisit；
4. B2：回到机器人实际执行第一段 B 时新获得的地点；
5. C2：经过另一个 goal switch 后再次调用 C。

B2/C2 故意使用与 B/C 完全相同的目标 JPEG。这不是简化，而是对 goal-session 边界最严格的测试。运行时只向 policy 传 RGB、ImageGoal 和可选 candidate ceiling；上述 role 字段只在 rollout 后分析，绝不进入 controller。

但旧 strict-v4 的 C 来自生成时 expert-A，而不保证任意重新采样的 actual-online A
都观察过该视图；并且 Novel-B 一旦失败，后续全部因果删失。因此它适合测试重复 goal
hash、session reset 和连续写入代码，不适合作为正式 multi-leg SR population。

两个本机 smoke 诚实结果：

- seed `0`：A/B 成功，C `600` 步失败；session `1→2→3`、memory `0…1157`
  连续，但 B2/C2 未被评估；
- seed `20260830`：A 成功、B 失败，后腿正确删失。

它们没有提供 multi-leg 正向 SR，也没有推翻 CEC；它们复现的是此前已经确认的
expert/online support 与 Novel survival 问题。

## 4. 正式因果设计：factual A/B prefix 后的 `C → B2 → C2`

正式 evaluator 改为复用 2026-08-14 已冻结、逐帧哈希的 actual-online NNR benchmark：

```text
factual online A frames   0 ... a
factual online B frames a+1 ... b
             ↓
       C(A-memory) → B2(B-memory) → C2(reopened C session)
```

- A/B 都是 frozen native NavDP 实际执行轨迹，不是 expert replay；
- C 在两臂中都固定只能检索 A，因此 treatment 前物理轨迹完全相同；
- `all_prior` 的 B2 ceiling 是 factual B 末尾；
- `initial_leg_only` 的 B2 ceiling 仍是 factual A 末尾；
- all-prior 的 C2 使用其新 session 的自然因果 ceiling，initial-only 仍固定为 A；
- 所有目标 role 只用于事后解释，不输入 controller。

这将主问题从“能否侥幸活过两个 Novel goal”改成一个可配对的因果问题：

> 在 A 后实际完成的 Novel-B rollout，是否会成为以后可认证、可执行的新长期记忆？

对应代码为 `eval_shared_online_lifelong_nnr.py`；独立 verifier 为
`independent_verify_shared_online_lifelong_nnr.py`。

## 5. 两个冻结 arm

### `all_prior`（部署主臂）

每个 goal session 可以读取冻结合同允许的全部实际在线历史。正式 NNR 扩展中，C
仍锁在 A 以保持共享 treatment prefix，B2 首次开放 factual B，C2 再恢复自然 ceiling。
它回答：系统是否会随着完成更多目标而获得新的可调用能力。

### `initial_leg_only`（记忆累积消融）

所有 post-A goal 的 candidate ceiling 均固定为实际 online-A 的最后一个 memory frame。controller、目标、seed、预算均不变。它回答：B2/C2 的能力是否真的来自 A 以后新增的经验，而非只是在重复利用初始 A。

后续还需增加 shared native 或 forced-reject arm，作为“没有 CEC 接管”的系统基线；它不应与 `initial_leg_only` 混为一谈。

## 6. 主指标

多-leg 不能只报最后 joint SR，因为一次早期失败会使后续全部被因果删失。正式结果同时报告：

- prefix survival：完成至少 1/2/3/4/5 个目标的 episode 比例；
- 每个 episode 在首次失败前完成的目标数；
- 5-leg joint SR 与 joint SPL；
- B2/C2 中 CEC 是否接管；
- B2/C2 被选 anchor 是否晚于 online-A boundary（`used_post_A_memory_*`）；
- full-history 相对 initial-only 的 episode-paired survival/count 差值；
- scene-cluster bootstrap CI；
- 首次 proof 与缓存命中 latency。

B2 若两臂都成功，还必须报告 paired steps、path length 和 SPL；不能因为 SR 饱和就把
显著的路径/动作代价差异丢掉。正式扩样前还必须用 GT co-visibility 在**读取任何新
multi-leg outcome 之前**冻结 factual-B support population，避免把“到达 B 的位置”
未经审计地等同于“B goal image 在 online-B RGB 中可重定位”。

条件 SR 只作诊断。不能把“只在前面成功的 episode 中评后腿”当成无偏主结论。

## 7. 当前 N=1 factual-prefix lifecycle pilot

使用 sealed NNR `dhjEzFoUFzH/episode_0003`；本机只做路径重映射，benchmark、A/B
trace、metadata、parquet、Goal A/B/C 资产哈希逐项与 HPC 原件一致。

| query | all-prior | initial-A-only |
|---|---:|---:|
| C | success，anchor 83（A） | 完全相同 |
| B2 | success，anchor 254（factual B） | success，0 takeover / native fallback |
| C2 | success，anchor 83（A） | success，anchor 83（A） |
| B2 steps | 109 | 198 |
| B2 path | 3.746 m | 6.895 m |

差值：B2 `−89` steps（`−44.9%`）、`−3.149 m`（`−45.7%`）。两臂 joint 都为
`1/1`，所以没有 SR 增益；这是 N=1 效率/生命周期机制结果，不是总体性能结论。

all-prior session receipts 为 `index 1/2/3`、ceiling `102/255/478`；initial-only 为
`1/2/3`、ceiling `102/102/102`。长期 memory 分别连续到 `566` 与 `657`。独立原始
文件 verifier 为 `verified=true`，并确认两臂在 B2 treatment 前的 A/B/C 物理前缀一致。

首次 proof 最大延迟：C `13.18 s`、新 B goal `46.22 s`；重复 C 的最大 decision
latency 降至 `0.647 s`。这提示长期系统还需要显式报告首次定位延迟与缓存命中延迟，
不能只报告 steady-state FPS。

## 8. 代码与当前状态

- goal-session 生命周期：`NavDP/baselines/memnav/policy_agent.py`；
- server receipt：`NavDP/baselines/memnav/memnav_server.py`；
- CEC hub receipt：`MemNavData/cec_controller_portability_hub.py`；
- 五段 evaluator：`MemNavData/eval_lifelong_5leg_habitat.py`；
- factual-prefix multi-leg evaluator：
  `MemNavData/eval_shared_online_lifelong_nnr.py`；
- 独立 verifier：
  `MemNavData/independent_verify_shared_online_lifelong_nnr.py`；
- 本机启动器：`MemNavData/run_cec_controller_portability_smoke_local.sh`，`EVAL_KIND=lifelong_5leg`。

下一步不是直接把 N=1 写入论文，而是先对 sealed NNR population 做 factual-B
co-visibility pre-audit，再冻结 `all_prior / initial_leg_only` 配对 manifest 和 source
bundle。现有 NNR 场景已被架构开发消费，因此扩样仍是内部机制确认；论文外部结论还需
未参与设计的 scene split 或真实机器人连续任务。

## 9. 正式内部扩样提交（2026-08-21）

已提交严格依赖链：

```text
factual-B result-blind support audit (16121493)
  -> paired all-prior / initial-leg-only array (16121506_[0-18%4])
  -> lossless aggregate (16121515)
  -> independent raw-file verification (16121524)
```

源 population 为 sealed NNR 的 `19 episodes / 8 scenes`。审计在读取任何新
`C/B2/C2` 导航结果前，以 `max factual-B co-visibility >= 0.20` 冻结 supported
分母，并单列 `>=0.50` strong support。每个 supported episode 的两臂在同一 Slurm
job/GPU 上顺序执行，奇偶 task 反转 arm 顺序；每对 time limit 为一小时。完整 receipt
为 `LIFELONG_NNR_EXPANSION_SUBMISSION_RECEIPT_20260821.json`。

这仍是被开发消费过的 NNR scenes，因此只能升级多次 Revisit / memory accumulation
的内部机制证据，不能取代 HM3D 或真机的外部确认。
