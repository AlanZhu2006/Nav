# REVISIT：已知阶段 RANSAC 消融结果

日期：2026-08-11（CST）  
状态：consumed 20-scene 架构消融完成；**不授权 blind、论文声明或直接替换 baseline**。

## 一句话结论

在 benchmark 已明确声明 Goal-B 是 Revisit 时，RANSAC 硬门确实漏掉了大量有用的 memory
proposal；去掉硬门把 conditional Revisit SR 从 `21/29` 提到 `26/29`。但配对只有
`+6/-1`、exact McNemar `p=0.125`，且出现一条原生本可成功却被 memory 干预破坏的真实
回退。因此：**RANSAC 不能再被解释成 Novel/Revisit 判别器，也不应继续作为唯一硬 veto；
但 raw-DINO direct 还不能安全替换它。**

## 1. 实验到底比较了什么

本实验只改变 Goal-B 的 memory activation：

| arm | Goal-A（Novel） | Goal-B（已知 Revisit） |
|---|---|---|
| `geometry_router` | 原生 NavDP；本次 false activation 为 0 | raw DINO top-8，经 SIFT/RANSAC 和两次确认后才接管 |
| `known_revisit_direct` | 逐像素复用 geometry arm 的完整 Goal-A trace | raw DINO top-1 直接调用相同 LingBot pose + mixed NavDP；pose 不可用时 fail closed |

所以这里没有让 RANSAC 参与 Novel 决策。旧 `memory_geometry` 通用实现会在两段都检查 router，
但本次 Novel 段实际 `0/40` 激活；在最终架构中应直接按已知 goal kind 隔离：Novel 只走
native，Revisit 才打开 memory residual。

## 2. 主结果

| 指标 | geometry router | known-Revisit direct | 差值 |
|---|---:|---:|---:|
| Novel A | 29/40 = 72.5% | 29/40 = 72.5% | 完全共享 trace |
| Revisit B \| A success | 21/29 = 72.4% | **26/29 = 89.7%** | **+17.2 pp** |
| Joint A∧B | 21/40 = 52.5% | **26/40 = 65.0%** | **+12.5 pp** |
| conditional mean SPL | 0.582 | **0.806** | +0.225 |
| conditional mean final distance | 2.203 m | **1.162 m** | −1.041 m |

严格配对结果：

- `both success = 20`；
- `direct only = 6`；
- `geometry only = 1`；
- `neither = 13`（其中 11 条是共享 Novel A 失败，真正 B 段 both-fail 为 2 条）；
- exact McNemar two-sided `p=0.125`；
- scene-cluster bootstrap 的 joint risk-difference 95% CI 为 `[0,+25] pp`；
- 6 个 gain 分布在 6 个不同 scene，1 个 loss 位于另一个 scene。

正式冻结分支为：

```text
inconclusive_build_cluster_geometry_ablation
```

即方向一致但证据未达到“零损失 direct replacement”的预注册安全门。

## 3. 七条 discordant pair 的独立复算

| scene / episode | 配对结果 | geometry 接管 plans | direct 接管 plans | final distance：geo → direct |
|---|---|---:|---:|---:|
| `ac26ZMwG7aT / 0000` | direct gain | 36 | 36 | 3.583 → 0.965 m |
| `gZ6f7yhEvPG / 0000` | direct gain | 0 | 11 | 3.358 → 0.979 m |
| `i5noydFURQK / 0000` | direct gain | 0 | 12 | 6.655 → 0.964 m |
| `qoiz87JEwZ2 / 0000` | direct gain | 0 | 6 | 3.815 → 0.975 m |
| `uNb9QFRL6hY / 0000` | direct gain | 0 | 6 | 11.377 → 0.963 m |
| `yqstnuAEVhm / 0001` | direct gain | 0 | 9 | 9.657 → 0.968 m |
| `pLe4wQe7qrG / 0001` | direct loss | 0 | 33 | 0.985 → 3.407 m |

五个 gain 是清楚的 RANSAC recall gap；`ac26` 则说明两次确认造成的接管时机差异也可能
改变闭环轨迹。不能把全部 `+6` 都写成“RANSAC 选错 anchor”。

## 4. 最重要的新归因：place 正确不等于 controller 输入可执行

把 direct 所选 anchor 映射回共享 Goal-A 轨迹后，六个 gain 的 anchor 到
`path_nearest_anchor` 的平面距离为 `0.11–1.52 m`。这说明 RANSAC reject/insufficient
确实漏掉了可提供有用方向的 proposal。

但唯一 loss 更关键：`pLe/0001` 的 direct anchor 距 `path_nearest_anchor` 只有
`0.07–0.37 m`，并不是明显的错误地点；尽管如此，它仍把 native 的成功变成失败。前四次
replan 中：

| 诊断 | native/geometry-inactive rollout | memory-conditioned rollout |
|---|---:|---:|
| candidate endpoint mean | 1.953 m | **0.185 m** |
| 全零 trajectory 次数 | 0/4 | **2/4** |

后续同 FIFO、同 seed 的只读 shadow 已在全部 7 条 discordant pair 上完成。`pLe` 的前四次
memory endpoint / PointGoal 为 `0,.127,0,.092`，但成功的 `qoiz/i5` 也有短暂平移抑制，随后
恢复。因此单次短轨迹不是 action failure，也不能直接形成 gate。

更关键的代码审计显示：NavDP 会把 PointGoal forward 分量截到 `[0,10]`；`pLe` 前四个
memory bearing 约为 `+175°, +160°, +176°, -166°`，后向信息在进入网络前被删除。NavDP
又把短候选的 `x,y` 清零而保留第 3 维，当前 pure-pursuit 却只消费 `x,y`。当前最强归因是
**后向 memory pose 超出 PointGoal controller 的支持域并触发执行接口失配**，而不是已经证明
需要一个 learned action expert。

作为探索性旁证，在 29 条 direct-eligible episode 中，3 条失败的前四次 rollout endpoint
均值为 `0.388 m`，26 条成功为 `1.505 m`；全零比例为 `50.0%` vs `19.2%`。这是同一
consumed pool 上的 post-hoc 诊断，只能用于提出下一假设，**不能据此选择阈值或声称预测
能力**。

## 5. 修正后的优雅架构

当前 benchmark 不需要再学习一个 Novel/Revisit existence expert；goal kind 已知。真正需要
分解的是两个不同可靠性：

```text
                    known Novel ───────────────► frozen native NavDP
goal-kind switch
                    known Revisit
                         │
                         ▼
              temporal DINO memory proposal
                         │
                         │
                         ▼
               controller support check
              forward >= 0 / forward < 0
                 │                 │
                 ▼                 ▼
          mixed Image+Point     native ImageGoal
                 │                 │
                 └──── next replan ┘
```

角色边界是：

1. DINO/temporal cluster 负责提出“历史上哪个位置”；
2. RANSAC 只提供高精度 place-support likelihood，`insufficient` 必须是 unknown，不能是
   task negative；
3. PointGoal 是否处于冻结 NavDP 的输入支持域是确定的接口条件，不需要学习；
4. residual 逐计划 fail closed：后向 PointGoal 回到原生 ImageGoal NavDP，下一次重新判断，
   不永久锁存；
5. 该模块只存在于已知 Revisit branch，Novel branch 完全不调用 RANSAC。

这比“语义 expert + 几何 expert”更贴合当前证据：本任务无需猜 goal 是否被访问过，而
`pLe/0001` 证明 place 正确仍可能在 source/controller 接口上失效，但尚未证明必须引入第二个
learned latent。

## 6. 后续：简单支持域修复已被否定

1. shadow-actionability 已完成：113 个同 FIFO、同 seed 配对，事实轨迹 7/7 与旧 direct 完全
   一致；完整结果见 `REVISIT_ACTIONABILITY_SHADOW_RESULT_20260811.md`。
2. 零参数 `navdp_front_support_v1` 的 `pLe` T0 得到 `+1/-0`，但 T1 前 4 个完整场景立刻变成
   `+0/-4`；按预注册 `loss=0` 门提前停止。
3. behind mixed 通常正是完成 U-turn 的方向来源；逐计划回退 native 会删除有效 memory
   guidance。不得恢复剩余 16 scenes，也不得在本 pool 上缩窄角度阈值。
4. endpoint、critic、behind 与前四步 bearing slope 均未成为可靠 safety signal。若继续，只能
   另立独立协议验证保留 memory bearing 的显式 rotate-first 执行契约，而不是再叠 gate。

完整负结果见 `REVISIT_FRONT_SUPPORT_RESULT_20260811.md`。

现在不应启动长训练、bearing head、X-NavDP controller replacement 或新的 semantic expert。
眼下缺的是对一次真实回退的可证伪接口归因，不是参数量。

## 7. 审计边界

- 20 scenes、40 episodes；共享 Goal-A trace、episode seed、逐 plan diffusion seed；
- scene overlap with training：空；输入与报告 SHA256 均复核通过；评测服务已退出；
- 本次当前代码下 Novel A 为 `29/40`，与旧 R0 的 `26/40` 不同。代码/执行器在两次实验间
  已变化，因此只能使用本次同进程配对效应，禁止与旧 R0 跨运行拼 joint 数字；
- 自动报告：`.diagnostics/revisit_known_phase_ablation_20260811/report.json`；
- 冻结协议：`REVISIT_KNOWN_PHASE_ABLATION_PROTOCOL_20260811.md`。
