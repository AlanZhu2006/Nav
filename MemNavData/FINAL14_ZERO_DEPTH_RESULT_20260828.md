# Final14 同 population zero-depth 结果（2026-08-28）

## 结论

在与 Final14 mono factorial 完全相同的 21-history / 42-query population 上，
冻结 NavDP 显式接收全零 depth 时仅成功 `4/42`：Novel `3/21`、Revisit `1/21`。
独立 verifier 从 raw final-distance/plan receipts 复算后给出 `verified=true`，并确认
`2,257` 个 plan 全部是 zero depth、metric sensor reads 为 `0`、monocular receipt
reads 为 `0`。

这补齐了同 population 的五行 query-stage depth/CEC 表：

| Query controller | Memory | Novel | Revisit | Overall | SPL | Mean path |
|---|---|---:|---:|---:|---:|---:|
| metric RGB-D | native | 5/21 | 6/21 | 11/42 | 0.1141 | 15.02 m |
| zero depth | native | 3/21 | 1/21 | 4/42 | 0.0586 | 15.92 m |
| causal mono | native | 6/21 | 4/21 | 10/42 | 0.1220 | 14.03 m |
| metric RGB-D | CEC | 6/21 | 20/21 | 26/42 | 0.4219 | 8.95 m |
| causal mono | CEC | 8/21 | 20/21 | 28/42 | 0.4742 | 8.78 m |

配对 zero-depth 对照：

- zero minus metric native：`+2/-9`，风险差 `-16.7 pp`，exact McNemar
  `p=0.06543`，scene-cluster 95% CI `[-38.5,-1.9] pp`；
- zero minus mono native：`+1/-7`，风险差 `-14.3 pp`，exact McNemar
  `p=0.07031`，scene-cluster 95% CI `[-31.6,-2.8] pp`。

因此正确口径是：depth 不是 NavDP 可随意丢弃的输入；causal mono 恢复了大部分
zero-depth 损失，但 native 的 mono/metric 差异在该小 population 上并不构成等价性
证明。CEC 的 Revisit 收益在 metric 和 mono query depth 下都达到 `20/21`，说明主
memory 效应不由某一种 query-depth source 单独制造。

## 协议边界

- 这是 consumed Final14 的 paired ablation，不是 fresh generalization；
- Goal-A causal history 来自原 metric-depth Goal-A replay；只有 query controller
  depth 被操纵，因此不能写成完整端到端 full-mono 系统结果；
- success radius 为 `1.0 m`，每 history 的 Novel/Revisit query、seed、budget、
  checkpoint 和历史前缀均与 verified factorial 相同；
- 表中的 overall 是 role-balanced 42-query total，不是 multi-leg joint SR。

## 基础设施审计

原 formal array `16499701` 中 index 19 在 evaluator 启动前因共享节点 TCP 端口竞争
退出（`Address already in use`），没有产生该 query outcome。partial 目录被只读归档，
其余 20 个完成单元没有重跑。加入 node-local `flock` 与 live-listener 检查的新
immutable bundle 只补跑 index 19：job `16502265`，exit `0:0`。replacement
summary/verifier `16502270` exit `0:0`。原 analysis `16499709` 及其下游从未执行，
已由 Slurm 取消。

## 权威文件

- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/final14_zero_depth_20260828/formal_4c061bd6b86da365`
- summary：`analysis/final14_zero_depth_summary.json`，SHA-256
  `263e00da60c1f9115f7f63a6074e0b7f9dedba0379eb79f7569f72a8e7bdfbba`
- independent verifier：
  `analysis/final14_zero_depth_independent_verification.json`，SHA-256
  `5f5eaf0b0393080d81096b55a0e086411be4e7b7cac9a4bd833b161634c37271`
- exact repair receipt：
  `FINAL14_ZERO_DEPTH_PORT_REPAIR_SUBMISSION_20260828.json`
