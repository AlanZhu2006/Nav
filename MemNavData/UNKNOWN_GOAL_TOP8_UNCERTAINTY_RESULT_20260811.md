# Unknown-goal Top-8 Set Uncertainty：结果

日期：2026-08-11（CST）  
状态：train-only 开发实验完成；预注册门未通过。

## 0. 结论

将 top-8 仅作为集合不确定性输入，确实在两个 seeds 增加了 correct-anchor coverage，但没有
稳定超过 top-2 F2，更没有超过 hard geometry；一个 seed 还越过了 geometry 的 strict-risk
上限。按冻结协议，停止继续扩充单时刻 feature，下一步只允许采集自然 planning stream 的
时序证据。

## 1. 隔离设计

- anchor 仍只从原 deployment top-2 中选择；
- conditional pairwise ranker、模型、fold、seed、risk calibration 全部不变；
- top-8 只增加 DINO score distribution、matches/inliers/ratio 分布、hard-pass 数量/首位、
  essential/pose-recovery/pass-rate 汇总；
- 不读取 phase、state、goal role、label 或 covisibility；
- 因此这不是已经证伪的“扩大 top-K 动作候选链”。

## 2. 主结果

H 固定为 hard geometry；F2 是上一轮 top-2 factorized support；F8 是本轮系统。

| Seed | 系统 | Correct support | Correct anchor | Wrong anchor | Strict FP |
|---:|---|---:|---:|---:|---:|
| all | H | 365/436 | **93/155** | 14 | 9/281 |
| 20260811 | F2 | 359/436 | 85 | 12 | 7 |
|  | F8 | 360/436 | 90 | 12 | **11** |
| 20260812 | F2 | 365/436 | 89 | 12 | 5 |
|  | F8 | 365/436 | 89 | **11** | 5 |
| 20260813 | F2 | 365/436 | 88 | 13 | 4 |
|  | F8 | 365/436 | 90 | 13 | 6 |

Existence 概率指标：

| Seed | F2 AUC | F8 AUC | F2 AP | F8 AP | F2 Brier | F8 Brier |
|---:|---:|---:|---:|---:|---:|---:|
| 20260811 | 0.9085 | 0.9112 | 0.8874 | 0.8917 | **0.1015** | 0.1057 |
| 20260812 | 0.8983 | 0.9137 | 0.8779 | 0.8872 | 0.1021 | **0.0987** |
| 20260813 | **0.9063** | 0.9052 | **0.8896** | 0.8888 | 0.0995 | **0.0984** |

## 3. 预注册门

- seed 20260811：coverage 比 F2 高，但 strict FP 11 > H 的 9，失败；
- seed 20260812：与 F2/H correct support 持平，correct anchor 89 < H 93，失败；
- seed 20260813：correct anchor 比 F2 高 2，但 correct support 不增、仍低于 H，失败。

冻结分支：

```text
all_three_seeds_pass = false
branch = stop_single_state_feature_expansion_collect_natural_stream_evidence
deployment_approved = false
```

## 4. 说明

top-8 distribution 不是完全无信息：它在两个 seeds 将 correct anchor 提高 2--5 个，第二个
seed 的 AUC 也明显提高。但这些增益没有稳定转成 operating-point 改善；seed 1 的 outer
held scenes 反而超出 inner-train 风险预算。这再次把问题定位为**单时刻跨场景 risk/coverage
迁移**，而非缺少更多静态 scalar feature。

按协议禁止继续试 top-4/top-16、softmax temperature、MLP 或 post-hoc threshold。下一轮
必须改变观测维度：在正常 NavDP 运动产生的连续视角上跟踪同一 memory hypothesis，以时序
一致性提升 Revisit coverage；部署不需要原地转圈。

## 5. 复现与产物

协议：`MemNavData/UNKNOWN_GOAL_TOP8_UNCERTAINTY_PROTOCOL_20260811.md`  
实现：`MemNavData/analyze_unknown_goal_top8_uncertainty_oof.py`  
测试：`MemNavData/test_unknown_goal_top8_uncertainty_oof.py`

正式命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python -u \
  -m MemNavData.analyze_unknown_goal_top8_uncertainty_oof \
  --phase-rows .diagnostics/phase_b_train_repaired_20260808/lingbot_goal_loop_closure_rows.csv \
  --geometry-evidence .diagnostics/revisit_geometry_expert_20260811/geometry_evidence.csv \
  --f2-report .diagnostics/unknown_goal_support_oof_20260811/report.json \
  --output-dir .diagnostics/unknown_goal_top8_uncertainty_oof_20260811 \
  --outer-folds 5 --inner-folds 4 \
  --seeds 20260811,20260812,20260813 \
  --bootstrap-samples 10000
```

- `report.json` SHA256：`45bb907e8afbbc45d39bfc9bd0c427d9421b9fc924e2ecfb783ff3e88e9ce668`
- `session_oof_predictions.csv` SHA256：`d0aef93365426ceda0b83afbed594e75b75884b5d41e44bc6e0033619369708b`

限制：仍是同一批已被查看的 40 train scenes 上的模型开发，且没有 Habitat rollout；这些
数字不是 SR，也不是独立论文确认。
