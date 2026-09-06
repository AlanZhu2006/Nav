# First-40 长程 Metric Scale 归因审计

日期：2026-08-31  
状态：已完成 post-hoc 只读复算；尚未授权替换 canonical fixed-2.5m CEC。

## 1. 审计问题

当前 full-mono 系统已经用已知相机安装高度，把 LingBot 的 first-40 causal RGB prefix
冻结为一个 episode-level metric scale receipt。短程 dense depth 使用该尺度，但 canonical
CEC 会丢弃 PnP 相对位移的模长，只输出固定 `2.5 m` 的 scale-free bearing。

本审计检验：同一个 first-40 receipt 是否也能把 CEC 的 LingBot-raw 长程位移转换为可信的
平面米制距离。

复算公式为：

```text
d_metric = scale_hat_first40 * ||pointgoal_lingbot_raw||
```

GT 对照是 eval 日志同一步记录的 Habitat 平面欧氏距离，不是 geodesic distance。

## 2. 数据与边界

- 数据：已经完成并消费的 HM3D fresh full-mono mixed-role formal evaluation；
- 主单位：每条 Revisit query 的第一次 certificate-authorized takeover；
- `28` 条 query，覆盖 `21` 个 scenes；
- 28/28 首次接管均发生在 query step 0；
- 只接受以下完整收据：
  - runtime 不读取 Novel/Revisit role；
  - CEC certificate accepted 且 adapter 实际 takeover；
  - `memory_unbounded_pointgoal_units=lingbot_raw_direction_only`；
  - first-40 scale valid、冻结且未使用 whole-episode cache；
  - simulator metric depth 未消费；
  - canonical controller 实际收到的仍是 fixed `2.5 m`。

因此这是对已存在日志的 counterfactual metric readout，不改变任何原始轨迹，也不是新的
闭环方法结果。

## 3. 首次接管结果

| 指标 | first-40 metric | fixed 2.5 m |
|---|---:|---:|
| mean absolute error | **0.361 m** | 0.433 m |
| median absolute error | **0.254 m** | 0.405 m |
| mean relative error | **13.51%** | 15.39% |
| median relative error | **9.65%** | 14.83% |
| within 20% | **23/28** | 19/28 |
| within 30% | 25/28 | **27/28** |
| P90 relative error | 31.94% | **26.90%** |

配对看，metric 更接近 GT 为 `19` 条，fixed 更接近为 `9` 条，exact two-sided sign
`p=0.0872`。Metric 的平均绝对误差低 `0.072 m`，但 scene-cluster bootstrap 95% CI
为 `[-0.224,+0.112] m`，跨零。

尺度校准整体没有明显偏置：

- `predicted / GT` mean `0.9994`；
- median `0.9672`；
- P10/P90 `[0.8519,1.1664]`；
- `scale_hat` 与每条 query 的 implied scale Pearson `r=0.736`。

因此，“first-40 高度先验无法恢复长程 metric range”不成立。它在大多数 query 上已经相当
准确，但首次接管的尾部风险尚未优于固定半径。

## 4. 尾部错误归因

3/28 首次接管的相对误差大于 30%：

| scene/episode | GT | metric | 相对误差 | floor relative IQR |
|---|---:|---:|---:|---:|
| `6D36GQHuP8H/episode_0002` | 2.882 m | 4.635 m | 60.8% | 0.927 |
| `vd3HHTEpmyA/episode_0001` | 2.040 m | 2.930 m | 43.6% | 0.776 |
| `b28CWbpQvor/episode_0003` | 3.221 m | 1.950 m | 39.5% | 0.116 |

`relative_floor_iqr` 与 metric relative error 的 Pearson `r=0.783`。前两条过估计具有异常高
的 floor IQR，说明 receipt 内已有尾部风险信号；第三条低估计没有被该信号捕获，仍可能来自
PnP translation norm、局部 depth bias 或长程 pose error。由于这个相关性是在正式结果后发现，
不能在同一集合上据此选择新 IQR threshold。

## 5. 沿导航过程的距离跟踪

28 条 query 共记录 `463` 个 certificate-authorized readouts，每条 query 先独立汇总，避免把
重复计划当独立样本：

- 28/28 query 的 metric readout 与 GT 距离相关性均为正；
- per-query Pearson median `0.9960`，mean `0.9747`；
- episode-balanced MAE：
  - height-scaled metric：`0.362 m`；
  - bounded metric `min(d_metric,2.5m)`：`0.382 m`；
  - fixed `2.5 m`：`0.562 m`；
- metric vs fixed 的 query-level MAE 配对：`21/7`，sign `p=0.01254`；
- bounded metric vs fixed：`23/5`，sign `p=0.000912`；
- 最后一次 readout 的 metric error median `0.195 m`，26/28 在 `0.5 m` 内。

这些数字说明 first-40 metric norm 不只是 episode-level 偶然校准正确；它通常会随着机器人接近
目标而稳定收缩。但所有轨迹仍由 fixed-2.5m controller 生成，因此这只能证明 readout quality，
不能直接宣称 bounded metric 会提高 SR。

## 6. 架构决策

不应直接把 canonical CEC 换成 unbounded metric PointGoal。首步 3 条大误差和更差的 P90 表明，
它会扩大当前 CEC 的最大控制权限。

下一条最小、可证伪的实验臂冻结为：

```text
v_raw = certified LingBot/PnP relative vector
d_hat = scale_hat_first40 * ||v_raw||
b     = v_raw / ||v_raw||
rho   = min(d_hat, 2.5 m)
PointGoal = rho * b
```

即 **bounded height-scaled residual**：

- 最大半径仍是早已冻结的 `2.5 m`，不扩大 memory authority；
- 不增加新的可调阈值；
- 距离较远或尺度过估计时退化为 canonical fixed radius；
- 接近目标时允许 residual 自然缩短；
- Novel rejection 与 exact native fallback 不变。

这条实验臂只能先作为 diagnostic challenger。必须经过同 history、同 seed、同机配对闭环后，
才能决定是否修改论文方法；在此之前论文仍保持 scale-free CEC 表述。

## 7. 可复现产物

脚本：

```text
MemNavData/audit_first40_longrange_metric_scale.py
```

测试：

```text
MemNavData/test_audit_first40_longrange_metric_scale.py
3 passed
```

本机只读产物：

```text
.diagnostics/first40_longrange_metric_scale_20260831/
  summary.json
  first_handoff_rows.csv
```

SHA-256：

```text
summary.json             7adeb0593e558b2ae16f003a64ddb06994200e05a659e78842e3feea2cec7205
first_handoff_rows.csv   5a890900dd11024d80b5c7806ef1148793371eb9e6b94bdd1fa7ad3cf4d3fd65
audit script             890a63f020afd3112931b0b6d5b282e2daf5e507b5432d212333bd0b95a324d2
```

注意：上述 hash 对应加入 trajectory diagnostic 后的当前脚本与产物；任何后续代码改动都必须
重新计算收据。

## 8. Bounded challenger 实现与 gate 状态

已新增独立实验 adapter：

```text
verified_bounded_metric_v1
```

实现边界：

- canonical `verified_bearing_v1` 的 schema、默认行为和 fixed `2.5 m` 均未修改；
- bounded arm 只接受 `lingbot_raw_direction_only` 与同一条 dense stream 已冻结的 first-40
  scale receipt；
- scale 缺失、非正、非有限或 units 不匹配时逐动作回退 native ImageGoal；
- 审计显式记录 `memory_metric_scale_m_per_raw`、unbounded metric distance 和 `2.5 m`
  radius cap；
- 只有 full-mono `certified_relocalization + monocular_sidecar` 合约允许选择该 arm；Pi3X
  与 phase-oracle route 均被 preflight 拒绝。

代码入口：

```text
MemNavData/revisit_bearing_adapter.py
MemNavData/eval_2leg_habitat.py
MemNavData/eval_shared_online_role_pairs.py
NavDP/baselines/memnav/policy_agent.py
```

本机核心测试为 `30 passed`。对 28 条真实 first-handoff 日志的 adapter replay 为：

- 28/28 takeover 可重放；
- 14 条缩短 residual，14 条触及 `2.5 m` cap；
- 最大 controller radius 严格等于 `2.5 m`；
- first-handoff bounded MAE `0.435 m`，与 fixed 的 `0.433 m` 基本相同。

最后一点很重要：bounded arm 的潜在价值不在第一次接管，而在随后接近目标时让 residual
收缩；这与第 5 节 trajectory readout 的结果一致，因此只有闭环 gate 能进一步判定。

本机 N=1 gate 在正式 query 前因资源竞争失败：启动检查时 GPU 空闲，但真机 MemNav/NavDP
服务随后上线并占用约 `20.4 GiB`，实验 MemNav 在 240-frame history replay 期间与其叠加后
OOM。该失败是 infrastructure-only，日志保留于：

```text
.diagnostics/cec_bounded_metric_smoke_20260831_attempt1/
```

没有删除或终止真机服务，也没有产生可计入 SR 的结果。

随后已建立并验证 HPC immutable gate bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
  cec_bounded_metric_gate_6005b4834eccaf95/

SOURCE_BUNDLE.sha256:
6005b4834eccaf95a22d5c5ac1366967af9dc95ea5acc29960137d3bd89db2eb
```

远端 exact-container contract dry-run 与 `23` 项核心测试均通过。N=1、两臂、同 server pair、
16-step Final14 consumed integration gate 已提交为 Slurm job `16666167`；提交后状态为
`PENDING (QOSGrpGRES)`，动态预计启动时间为 `2026-08-31 10:45:29 EDT`。这不是正式扩样，
也不应根据它的单条 SR 修改论文结论；它只决定是否具备提交 paired population 的技术资格。
