# Online Router Failure Audit — 2026-08-05

本文记录 20-scene 2-leg、10-scene true 3-leg、单 episode 完整候选审计，以及
LingBot-native loop-closure feasibility smoke。所有诊断只读取已有数据、checkpoint
和闭环 buffer；最终保留场景没有用于选阈值。

## 1. 结论摘要

目前证据不支持把主要失败笼统归因于“LingBot 长程转弯后 pose 全部漂掉”。更精确的
分解是：

1. 2-leg Revisit 中，geometry router 相比原生 NavDP 有明确增益；
2. 3-leg 当前首先被第二个 Novel 目标 B 卡住，导致只有一条 episode 有资格评 C；
3. 唯一可评 C 的失败中，正确历史簇真实存在，RANSAC 也能识别，但 raw DINO top-1
   选中了另一个相似区域；
4. temporal diversification 能把正确簇从 raw rank 20 提前到候选 3；
5. 冻结的 Patch+temporal pointwise 模型在这个未见过的在线轨迹上把正确候选排到
   第 1，而 listwise 模型只到第 7；
6. LingBot 点云重叠在极小 smoke 上能区分正负候选，但仅比较 pose consistency 不能。

因此近期最有效、风险最低的路径是：

```text
raw DINO complete history
        -> temporal-NMS
        -> task-aligned Patch+temporal rerank
        -> RANSAC/SIFT final verification
        -> verified anchor latch
        -> LingBot metric pose -> frozen NavDP local controller
```

LingBot point-cloud overlap暂时作为纹理弱场景的 fallback 或离线 teacher，不直接替代
快速的 RANSAC。

## 2. 20-scene / 40-episode 2-leg 闭环结果

来源：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  all_in_one_router_eval_20260805/
  all1_2leg3leg_6dc745f_20260805/
  expanded_navdp_router/summary.json
```

配对 benchmark 的 Goal A 都是 Novel，Goal B 都是在 Goal A 成功之后的 Revisit。

| 指标 | NavDP native | Geometry router |
|---|---:|---:|
| Novel A SR | 31/40 = 77.5% | 31/40 = 77.5% |
| Revisit B SR（给定 A 成功） | 4/31 = 12.9% | 19/31 = 61.3% |
| Joint SR | 4/40 = 10.0% | 19/40 = 47.5% |
| Revisit B mean final distance | 7.035 m | 2.586 m |
| Revisit B mean SPL | 0.0350 | 0.4377 |

配对结果为 geometry-only 15、NavDP-only 0，McNemar exact
`p = 6.1035e-5`。Novel A false activation 为 `0/40`。

但新增场景暴露了 router recall 问题：

- 原先 5 个 audit scene：Revisit 激活 `10/10`，成功 `8/10`；
- 新增 15 scene 中有资格的 21 条：只激活 `13/21`；
- 激活后成功 `10/13 = 76.9%`；
- 未激活时只有 `1/8` 成功。

也就是说，controller 在 router 激活后仍然相对稳定；主要泛化缺口发生在“有没有找到并
验证正确 memory”，而不是 point-goal controller 一激活就失效。

## 3. 10-scene true 3-leg 结果

协议为：

```text
start -> A (Novel) -> B (Novel) -> C (Revisit of A)
```

来源：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  all_in_one_router_eval_20260805/
  all1_2leg3leg_6dc745f_20260805/
  expanded_3leg_router/summary.json
```

正式顺序条件统计：

| 指标 | NavDP native | Geometry router |
|---|---:|---:|
| A Novel | 6/10 | 6/10 |
| B Novel，给定 A 成功 | 1/6 | 1/6 |
| C Revisit，给定 A/B 成功 | 0/1 | 0/1 |
| Joint | 0/10 | 0/10 |

这组数据不能单独判断 C 的总体 memory 能力，因为 B 只留下一个可评 C 样本。它首先
揭示的是连续 Novel exploration 的问题，而不是足够统计量下的“3-leg memory drift”。

唯一可评 C 的场景是 `cV4RVeZvu5T/episode_0000`：

- C recall gap：353 frames；
- C 规划 31 次；
- router 从未激活；
- 最终距离：4.418 m；
- online top-1 anchor：209；
- top-1 geometry：13 matches、5 inliers、ratio 0.3846，未通过。

## 4. cV4 完整候选审计

本地只读副本：

```text
/tmp/cv4_3leg_online_geo_15372729/
```

可复现脚本：

```text
MemNavData/audit_online_router_topk.py
```

输出：

```text
/tmp/cv4_3leg_online_geo_15372729/router_topk_audit.json
/tmp/cv4_3leg_online_geo_15372729/router_topk_learned_audit.json
```

候选范围严格复现线上约束：frame 39–328，共 290 帧。使用和线上完全相同的
SIFT ratio test、essential matrix、recoverPose 和阈值后：

- 32 帧通过 geometry teacher；
- 正确簇为 frame 53–90；
- frame 82 为 55 matches / 40 inliers / ratio 0.7273；
- frame 69 为 22 / 14 / 0.6364；
- raw DINO 第一个正确候选在 rank 20；
- raw top-8 和 top-16 都不包含正确候选；
- raw top-32 才包含正确候选。

问题是相邻错误帧 206–215 占据了 raw 排名头部。temporal-NMS 后：

| min frame gap | 第一个正确候选在选择列表中的位置 |
|---:|---:|
| 4 | 8 |
| 8 | 6 |
| 16 | 3 |
| 32 | 3 |

gap=16 的前 3 个为：

```text
209 (geometry fail), 125 (geometry fail), 69 (geometry pass)
```

所以本次失败不是“历史里根本没有正确图”，也不是“RANSAC 无法识别正确图”，而是
线上只验证 raw DINO top-1。

## 5. 冻结 learned router 在在线失败上的结果

模型：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  patch_router_multiscene_20260805/full_job_15315411/
  patch_temporal/diagnostic_patch_temporal_router_not_for_deployment.json
```

模型 SHA256：

```text
7a3605d9bb8891a280286408d6c54c161ae1604246b95dac4e64a0671f318723
```

在训练时一致的 temporal-NMS gap=4 / top-32 候选上：

| 排序方式 | 第一个 geometry-positive rank |
|---|---:|
| DINO + temporal-NMS | 8 |
| listwise Patch+temporal | 7 |
| pointwise Patch+temporal | 1 |

pointwise rank 1 是 frame 74：

- geometry positive；
- probability 0.9086。

但冻结模型的零误判自动接受阈值为 0.9621，所以 frame 74 仍应交给 RANSAC，而不是
自动接受。这正好支持“学习模型负责排序、几何负责最后确认”的安全部署方式。

这个单样本也说明不能仅凭 development aggregate 就规定 listwise 永远优于
pointwise。listwise 在这里仍把错误 frame 209 排在第 1；上线前需要更多 online
buffer 做分布外排序审计。

## 6. LingBot-native loop closure smoke

Job：`15387026`，commit：`57c4dcc714d2b517e37abb681eea2e8319937f32`。

输出：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
  lingbot_goal_loop_closure_20260805/smoke_job_15387026/
  diagnostic_lingbot_goal_loop_closure.json
```

该实验选取 2 个 co-visibility positive 和 2 个 hard negative，每个候选在
`-4/0/+4` 邻居 context 下完整重放 LingBot stream。

| 特征 | ROC-AUC | AP |
|---|---:|---:|
| DINO cosine | 1.0 | 1.0 |
| LingBot cloud overlap | 1.0 | 1.0 |
| LingBot pose consistency | 0.5 | 0.583 |
| LingBot pose refinement | 0.5 | 0.667 |

类别中位数：

- positive cloud-overlap F1：0.4383；
- negative cloud-overlap F1：0.0；
- positive pose translation dispersion：0.00861 normalized；
- negative pose translation dispersion：0.00953 normalized。

因此 cloud overlap 值得扩大验证，pose dispersion 暂时没有区分力。由于只有 4 个
候选，这不是部署结论；而且完整 LingBot replay 显著慢于约 50 ms 的 uncached
RANSAC，所以合理角色是 hard-case fallback 或离线 teacher。

## 7. 已实现修复

核心 commit：`20b81f4`。

1. live MemNav 保存 raw-DINO complete-history 分数；
2. 以 frame gap 做 deterministic temporal-NMS；
3. evaluator 最多验证 8 个多样化候选；
4. 同一 anchor 必须连续两次通过才 latch；
5. latch 后 pose query 强制使用已经验证的 anchor，不再退回 raw top-1；
6. 记录每个候选的 rank、score、matches、inliers、耗时和最终 selected anchor；
7. summary 使用整次 top-K verification 总耗时，而不是只记录最后一个候选；
8. 增加完整候选/learned-router 在线审计脚本和单元测试。

提交前在只包含本次改动的临时 commit snapshot 上通过 62 个相关测试，且
`py_compile`、`bash -n`、`git diff --check` 全部通过。

## 8. 正在排队的验证

### 8.1 cV4 3-leg 闭环因果复测

- Job：`15389371`；
- commit：`1db37fdd23bf3dcdbcb0eae3c14b507bd1e6eb94`；
- 对照：相同 episode / seed 的 native NavDP 与 temporal-NMS + top-8 geometry；
- 首要检查：C 段是否选择 frame 69 附近、router 是否在第二次确认后 latch、C 是否成功。

### 8.2 六场景 LingBot overlap 扩展

- Job：`15389539`；
- dependency：afterany `15389371`；
- commit：`f3ad8c1eded1e4d38dcb23f8a0b93f3202d70fe0`；
- 6 个场景，正负候选平衡，完整 replay。

两项任务当前因同一项目账户下 3 个共享 `memnav_mp3d` 长训占满 GPU quota 而排队，
不是代码 preflight 失败。共享任务最晚计划于 2026-08-06 约 07:20 结束。

## 9. Final-blind 数据完整性问题

all-in-one Job `15372729` 的 2-leg 和 3-leg 阶段已完整结束。最后 learned-router
blind 阶段失败，是因为 4 个真正 final-reserved scene 在当前只读 overlay 中只有空
scene 目录，没有两条 raw episode：

```text
2t7WUuJeko7
D7G3Y4RVNrH
HxpKQynjfin
RPmz2sHmrrY
```

这不是模型性能失败，也不能用 development scene 替代后宣称 final blind。当前代码
已经把 episode count 检查移到 all-in-one 最前面，未来会在耗费 GPU 前 fail fast。
正式 final-blind 前必须先独立生成这 4 个场景的数据和 LingBot cache。

## 10. 下一步判定规则

1. 如果 `15389371` 的 C 成功：扩展 20-scene/3-leg 闭环，重点报告 Novel false
   activation、Revisit activation、joint SR 和总 verification latency；
2. 如果 router 选中正确 anchor 但 C 仍失败：问题下移到 LingBot metric pose 或
   point-goal controller，应做 correct-anchor pose 与 oracle-point-goal 对照；
3. 如果多场景 cloud overlap 仍稳定：将其作为 texture-poor fallback/teacher；
4. learned pointwise rerank 在更多 online buffer 稳定后，再接入 live pooled patch
   cache；在此之前保留 RANSAC final verification；
5. 单独解决 Novel B exploration，再扩大 true 3-leg 的 C eligible denominator。
