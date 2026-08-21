# Unknown-goal Memory Support：Nested Scene-OOF 结果

日期：2026-08-11（CST）  
状态：train-only 离线门已完成；**未通过部署门，不授权闭环或 action expert 长训**。

对应协议：`MemNavData/UNKNOWN_GOAL_SUPPORT_OOF_PROTOCOL_20260811.md`  
实现：`MemNavData/analyze_unknown_goal_support_oof.py`  
测试：`MemNavData/test_unknown_goal_support_oof.py`

## 0. 一句话结论

新版 learned router 在方法上修正了旧 Phase-B 的核心缺陷，并稳定优于公平 DINO；但它
没有在匹配风险下超过 hard-geometry expert。当前失败不再是“误激活过多”，而是
**真正有 memory support 时过度 abstain**。因此冻结分支为：先改善自然时序/多视角的
support observability，不进入 action expert 或长时闭环。

## 1. 相比旧 learned router 改了什么

| 旧 Phase-B/GLP | 本轮 F：factorized support | 改进目的 |
|---|---|---|
| 一个分数同时承担“是否有记忆”和“选哪个 anchor” | existence head 与 conditional pairwise ranker 分开 | 避免相对排序能力被迫承担绝对校准 |
| candidate-level 绝对 score，跨场景 threshold 漂移 | top-2 set distribution + 连续 RANSAC + pose-hypothesis agreement | 显式表达歧义和证据一致性 |
| RANSAC 容易作为 hard gate/teacher reject | pass 是正证据；reject/insufficient 保留连续量 | 不把“未证实”误写成 Novel |
| development 被反复用于阈值判断 | 40 train scenes 上 nested scene-OOF | held-out outer scene 不参与拟合或阈值 |
| 曾出现固定 0.5 的不公平 DINO 基线 | DINO 与 F 使用同一 geometry strict-risk budget | 比较同一 operating risk |
| candidate AUC 被误当部署成功 | correct support decision、correct/wrong anchor、strict FP | 直接评价 router 的部署决策单元 |
| 单次 split 容易偶然 | 三个 scene-fold seeds，全部通过才 Go | 抑制 split luck |
| 可能隐含 A/B/C phase | 推理特征完全不含 phase/goal role/state name | 符合真实 unknown-goal 部署 |

注意：这是设计和评测上的实质改进，不等于性能已经成功。

## 2. 数据与审计

- repaired Phase-B train：40 scenes、480 sessions、1,098 candidate rows；
- deployment candidate：每 session 恰好 top-2，共 960 rows；
- session teacher：155 positive、281 strict no-match、44 ambiguous；
- 155 个 positive 中 128 个的 deployment top-2 含 positive；
- teacher-forced candidate 只进入 outer-train conditional ranker，不进入 held-out deployment；
- development、已耗尽的 20-scene pool、blind 均未读取；
- outer 5-fold、inner 4-fold，scene-disjoint；seeds 为 20260811/12/13。

输入 SHA256：

- Phase-B rows：`193c29da7e2904061691361d5285d2211ff61b997619156f8b74262fde18237b`
- geometry evidence：`ef9374b31a63e4012525e0e68e7749b94569b6b263162143ee516bd4b8260463`

## 3. 主结果

### 3.1 Support decision

分母固定为 155 positive + 281 strict no-match = 436 extreme sessions。

| Seed | 方法 | Correct support | Correct anchor | Wrong anchor | Strict FP |
|---:|---|---:|---:|---:|---:|
| all | Hard geometry H | 365/436 = **83.72%** | **93/155** | 14/155 | 9/281 = 3.20% |
| 20260811 | DINO D | 354/436 = 81.19% | 82 | 13 | 9 |
|  | Factor F | 359/436 = 82.34% | 85 | **12** | **7** |
| 20260812 | DINO D | 355/436 = 81.42% | 84 | 15 | 10 |
|  | Factor F | **365/436 = 83.72%** | 89 | **12** | **5** |
| 20260813 | DINO D | 355/436 = 81.42% | 84 | 15 | 10 |
|  | Factor F | **365/436 = 83.72%** | 88 | **13** | **4** |

F 相对 H 的 scene-cluster bootstrap correct-support 差：

- seed 20260811：−1.38 pp，95% CI [−3.64, +1.14] pp；
- seed 20260812：0.00 pp，95% CI [−2.24, +2.32] pp；
- seed 20260813：0.00 pp，95% CI [−2.28, +2.49] pp。

因此不能声称 F 优于 H，甚至没有稳定的正向点估计。

### 3.2 Existence 与 conditional ranking

| Seed | F ROC-AUC | DINO ROC-AUC | F AP | DINO AP | F Brier |
|---:|---:|---:|---:|---:|---:|
| 20260811 | 0.9085 | 0.8878 | 0.8874 | 0.8706 | 0.1015 |
| 20260812 | 0.8983 | 0.8878 | 0.8779 | 0.8706 | 0.1021 |
| 20260813 | 0.9063 | 0.8878 | 0.8896 | 0.8706 | 0.0995 |

DINO Brier 没有列入公平主比较：报告中的 0.2438 来自 report-only min-max 映射，不是与 F
同等训练的概率校准器。

在 128 个 top-2 含 positive 的 session 上，F ranker 相对 DINO top-1：

- seed 20260811：10 win / 4 loss / 114 tie，exact McNemar p=0.180；
- seed 20260812：10 / 4 / 114，p=0.180；
- seed 20260813：10 / 5 / 113，p=0.302。

方向一致但未显著。它再次说明“选哪个 anchor”有信号，但不是当前主瓶颈。

## 4. 失败归因

### 4.1 主要损失是 abstain，不是 rank 错误

在 positive sessions 上：

| Seed | top-2 本身无 positive | top-2 有 positive 但 F abstain | F active 但选错 | F correct |
|---:|---:|---:|---:|---:|
| 20260811 | 27 | 38 | 5 | 85 |
| 20260812 | 27 | 34 | 5 | 89 |
| 20260813 | 27 | 34 | 6 | 88 |

逐 seed 查看“geometry 正确、F 错误”的 session：

- 7--10 条是 F abstain；
- 1--2 条是 F 选错 anchor；
- 2--4 条是 strict no-match 上 F 误激活。

所以最该修的不是 ranker 宽度，而是 existence 在真实 support 状态上的可观测性/覆盖。

### 4.2 保守化解决了旧问题，却产生新问题

旧 Phase-B 的主要失败是未见场景整体过度自信，导致 activation threshold 迁移损失。新 F
用 nested risk matching 将 strict FP 压到 4--7，低于 geometry 的 9；代价是 correct anchor
只有 85/89/88，低于 geometry 的 93。

这不是毫无进步：系统已经从“危险地多接管”变成“安全但接管不足”。但在 base policy
保护型 residual 方法中，安全只是必要条件，不能用覆盖损失换取论文增益。

### 4.3 状态分层只作解释，不作调参

三种子平均 correct-support 的 F−H：

- `goal_b_t0`：+1.74 pp；
- `goal_b_midpoint_t1`：−0.24 pp；
- `goal_c_t0`：−3.07 pp。

也就是说，F 在多数 no-match 的 Novel 起点更容易正确 abstain，却在真正 Revisit 为主的
`goal_c_t0` 丢覆盖。这些 state/role 字段没有进入模型，且本分层是在结果后查看，只能用于
定位下一数据需求，不能用于设计 phase rule。

## 5. 预注册判定

每个 seed 的门：

| 条件 | 结果 |
|---|---|
| F strict FP ≤ H | 3/3 通过 |
| F wrong anchor ≤ H | 3/3 通过 |
| F rank wins > losses | 3/3 通过 |
| F correct anchor > H | **0/3 通过** |

冻结结论：

```text
all_three_seeds_pass = false
branch = stop_before_action_expert_improve_memory_support_observability
deployment_approved = false
```

## 6. 这轮实验说明什么

1. **旧 learned router 并非完全没有可学习信息。** 公平、unknown-stage 的 F 稳定优于
   DINO，证明 set/geometry/pose features 可迁移。
2. **但“可排序”仍不等于“可部署”。** conditional ranker 只有 10/4--5 的非显著方向性，
   整体覆盖仍被 existence gate 限制。
3. **RANSAC 不应被删除。** 当前 geometry expert 仍有最高 correct-anchor coverage；学习
   expert 更适合补充其不完备性与校准风险，而不是直接替换。
4. **真正缺的是证据，不是模型容量。** 单个 planning state 的 top-2 观测不足以同时做到
   低风险和高 Revisit recall。
5. **当前不值得跑 8 小时闭环。** Stage-1 必要条件未过，长评测只会昂贵地验证一个已知
   不合格的 gate。

## 7. 下一步

优先级冻结为：

1. 采集 train-scene、正常 NavDP 自然运动中的 planning-stream memory evidence；不原地转圈；
2. 为稳定 anchor 累积 top-K score-distribution uncertainty、match-level uncertainty、连续
   RANSAC 与 pose/bearing agreement；
3. geometry pass 保留为 expert 正似然，reject 不作负标签；
4. 用相同 nested scene-OOF 与三种子 gate 比较 temporal support、single-state F、hard H；
5. 只有 temporal 版本三种子全部提高 correct-anchor coverage 且不增加 strict FP/wrong
   anchor，才采集 Native-vs-Memory counterfactual action advantage 并进入闭环。

禁止：在 development/consumed 20 scenes 上选时间窗、阈值或 feature；按 `goal_c_t0` 写
phase rule；继续调 logistic `C`；把本轮 AUC 当方法成功。

## 8. 产物与复现

正式命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python -u \
  -m MemNavData.analyze_unknown_goal_support_oof \
  --phase-rows .diagnostics/phase_b_train_repaired_20260808/lingbot_goal_loop_closure_rows.csv \
  --geometry-evidence .diagnostics/revisit_geometry_expert_20260811/geometry_evidence.csv \
  --output-dir .diagnostics/unknown_goal_support_oof_20260811 \
  --outer-folds 5 --inner-folds 4 \
  --seeds 20260811,20260812,20260813 \
  --bootstrap-samples 10000
```

输出：

- `report.json` SHA256：`85f79f53c701ee8d3339dea1dd258cc060a76b5ea253fd834c184293aea97ca7`
- `session_oof_predictions.csv` SHA256：`7db66f437251235df49797c9d7259dfbcc06746fba521e000c9cc85316cc6868`

限制：hard-geometry reference 是离线 decision-unit 近似，未模拟线上 two-plan latch；本轮没有
Habitat rollout，因此不能把 83.72% 写成 navigation SR。
