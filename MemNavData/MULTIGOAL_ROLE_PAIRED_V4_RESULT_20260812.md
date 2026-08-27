# Multi-goal role-paired v4：修复与本机验证（2026-08-12）

## 本次解决的问题

v3 已修复后段 Novel 无距离上限、首段路径预对齐和 Goal-B 图像/位姿不一致，但继续审计发现三个残余漏洞：

1. A、B 只共享 3–9 m 区间，没有在同一 episode 内匹配距离；旧 smoke 曾出现 `5.02 m vs 3.65 m`。
2. Goal C 虽名义上回访 leg A，leg B 却仍有 `0.25–0.67` 的最大共视，不能称为 hard-negative 最近历史。
3. 采样预算耗尽时，生成器打印 `DONE: 0/1` 后仍以退出码 0 结束，HPC 可能静默接受残缺场景。

## v4 契约

协议标识：`multileg_v4_role_paired_20260812`。

- A/B 使用同一个 3–9 m geodesic band。
- 每条 episode 的 `|geo(start,A)-geo(A,B)| <= 0.50 m`。
- A 起始 yaw 均匀采样，不做路径预对齐。
- A/B 的 metadata 目标位置绑定专家真实终点；Goal B yaw 和 JPEG 也绑定同一终点帧。
- Novel B 对 leg A 的完整 co-visibility curve 严格低于 `novel_covis`。
- Revisit C 的 anchor 必须位于 `[anchor_margin, switch_A)`。
- Revisit C 对全部 leg-B 帧的最大共视必须不超过 `covis_pos_lo=0.10`。
- 生成不足默认非零退出；只有显式 `--allow_incomplete` 才允许诊断性部分输出。
- 每个输出目录写 `generation_summary.json`，记录 outer/candidate attempts 和逐原因 rejection counts。
- 3-leg A/B 距离带不同或距离匹配容差大于 0.50 m 时，命令在加载 Habitat 前直接报错。
- 2-leg Revisit 的默认协议和采样行为保持不变。

严格契约还检查 switch、curve 长度、C anchor 归属、hard-negative tail、stored/measured geodesic、首帧/终点位姿、B 终点 JPEG 以及所有 NaN/Inf。

## 新场景真实数据 smoke

场景：`gTV8FGcVJC9`，seed `20260814`，RTX 4090。

计划生成 2 条；第一条约两分钟内成功，第二条在更严格的 C hard-negative rejection 下继续约两分钟仍未接受，因此主动停止临时任务并保留完整的第一条。没有把部分数据伪装成 2/2 完成。

加入 rejection telemetry 和延迟 ESDF 构建后，用相同场景/seed 从最新源码重新生成 `1/1`：使用 2 次 outer calls（共 93 个内部 episode candidates）完成。新旧成功 episode 共 869 个文件，文件名集合和 SHA-256 均逐一相同，证明诊断/效率修复没有改变样本语义或渲染结果。

成功 episode：

- A geodesic：`4.8046 m`。
- B geodesic：`4.9917 m`。
- 配对误差：`0.1871 m < 0.50 m`。
- 初始相对 heading offset：`-51.31°`。
- C anchor：frame `39`，位于 leg A。
- C 对整个 leg B 的最大共视：`0.0495 < 0.10`。

独立审计不再复用 metadata 距离，而是重新加载 scene、重建 evaluator navmesh，并调用 Habitat pathfinder：

- contract：`1/1` valid。
- metadata start 对首个存储帧误差：`0 m`。
- A/B metadata 目标对专家终点误差：均 `0 m`。
- Goal B yaw 误差：约 `4.2e-7°`。
- Goal B JPEG 与专家终点 JPEG：完全一致。
- Goal C JPEG 按 metadata pose/yaw 重新渲染：完全一致。
- Parquet、RGB、depth 的行数和连续帧索引：完整。

小场景 `i5noydFURQK` 的 1-attempt 负测试生成 `0/1`，现在正确返回退出码 1；不同 A/B band 的命令正确返回 argparse 退出码 2。

该负测试的 rejection telemetry 也正常落盘：60 个内部候选中，56 次在 A clearance gate 被拒；进入 Novel-B 采样的两次主要因同楼层/clearance 候选不足而耗尽。由此可见该场景的失败不是 C hard-negative gate 造成的，后续不能通过放松 C 定义来“修”接受率。

## 冻结 NavDP 闭环 smoke

官方冻结 checkpoint：`navdp_checkpoint.ckpt`；每段最多 600 steps；确定性 plan seeds。

- carry：A `1/1`，B|A `0/1`，C 因 B 失败被因果 censor。
- reset before B：A `1/1`，B|A `0/1`。
- A 在两臂的 steps 和 final distance 完全一致。
- role-paired 反事实：`b2_executed=0/1`，`b1_role_matched=0/1`；final distance `9.044 vs 9.038 m`，配对 `+0/-0`。

这些数字只证明严格数据能够贯通冻结控制器和配对评测器，不估计 SR，也不支持 FIFO reset。

## 回归与测试

- 相关 compile：通过。
- unit tests：`20/20` 通过。
- post-v4 2-leg Revisit 实际生成：`1/1`，确认没有改变其默认路径。
- NavDP server 已关闭，GPU 无残留进程。

最终源码生成与审计：`.diagnostics/multigoal_v4_final_source_smoke_20260812/`。冻结 NavDP 闭环及配对结果：`.diagnostics/multigoal_v4_second_scene_20260812/`；两者 episode 已逐文件验证完全相同。

## 下一步门槛

在扩到 20 scenes / 40 episodes 之前，应先解决生成接受率的资源问题，但不能放宽 hard-negative 或距离配对契约。合理动作是记录 rejection reason，并按场景统计 acceptance rate，再为低接受率场景提高采样预算或并行化；不能用生成成功与否筛选最终评测场景。

## 2026-08-12 Revisit controller 接线修复

上述冻结 smoke 使用 `--server_backend navdp`，因此只验证了数据与原生控制器，并未执行
Revisit 方法。后续审计发现旧 3-leg evaluator 在 hybrid 模式下强制所有腿使用 automatic
router，无法表达 benchmark 已知 role 的最强 Revisit reference。

现在逐腿契约已固定并写入结果 receipt：

- native arm：A/B/C 均为原生 NavDP；
- known-role direct arm：A/B 由原生 NavDP 控制、同时写入因果长记忆；只有 C 使用
  raw-DINO top-1 + LingBot metric pose + mixed NavDP residual；
- role-free certified arm：A/B/C 都执行一次 certificate，reject 当步原生，accept 才输出
  scale-free bearing，经固定 2.5 m adapter 进入 mixed NavDP。

`multigoal_policy_contract.py` 对三条路径 fail closed；Habitat Python 3.8 的真实 CLI 入口已通过，
相关 28 项测试通过。当前 v4 单条 episode 的 native B 失败，因此 C 被正确因果 censor；尚未把
“接口测试通过”误写为 v4 Revisit SR。下一步需生成足够多的 v4 episode，使 `A∧B` 分母非零，
再比较 `SR_C|AB` 和 A/B false takeover。

## 2026-08-12 四场景 strict-v4 快速验证

为确认修复不是单场景特例，预先固定四个场景和四个 seed，各生成一条数据；没有根据策略成败
筛选或重采样 episode。四条均通过独立 Habitat pathfinder、Parquet、JPEG 和契约审计：

| scene | geo A | geo B | 配对误差 | 初始 heading | C 在整个 leg B 的最大共视 |
|---|---:|---:|---:|---:|---:|
| `e9zR4mvMWw7` | 3.114 m | 3.006 m | 0.108 m | +59.85° | 0.0636 |
| `gxdoqLR6rwA` | 5.215 m | 4.887 m | 0.327 m | -17.02° | 0.0380 |
| `dhjEzFoUFzH` | 6.281 m | 6.159 m | 0.122 m | -119.07° | 0.0651 |
| `gTV8FGcVJC9` | 3.817 m | 3.651 m | 0.166 m | -37.96° | 0.0509 |

四条的 A/B 目标位置均与各自 expert terminal 一致，Goal-B yaw 误差小于 `1e-6°`，Goal-B
JPEG 与 terminal JPEG 逐字节一致，Goal-C 按 metadata pose/yaw 重渲染也逐字节一致。由此可见
旧数据中的 B 距离无上限、目标帧错绑，以及 C 同时属于 leg B 的三项混杂，在真实多场景数据上
均已被消除。

### 额外发现并修复：evaluator 错误压平目标高度

`gxdoqLR6rwA` 的离线审计为合法，但旧 `eval_3leg_habitat.py` 把 A/B/C 的 Y 坐标强行替换为
episode 起点高度，在缓坡或不平地面上制造了 A `0.0534 m`、B `0.0904 m` 的假终点误差，随后
fail-closed 拒绝正确数据。现已改为使用生成器记录的完整 Habitat floor position；同类修复同步到
`eval_3leg_symmetry_habitat.py` 和 `eval_novel_b_habitat.py`。修复后 gxdo 数据通过真实 evaluator
并完成闭环。全局审计还发现 `eval_2leg_habitat.py` 以同样方式压平 A/B geodesic，现也改为
metadata 的完整 3D floor pose；这不改变正式 server-selector 路径或 2D SR，但修正 geodesic/SPL。
四个入口 compile 通过，相关契约、路由和汇总回归 `37/37` 通过。该修改只去掉假拒绝和距离污染，
不改变成功阈值或策略。

### 冻结 native 的有效分母 smoke

官方冻结 `navdp_checkpoint.ckpt`，每 leg 600 steps，四场景结果：

| scene | A | B\|A | C\|AB | 备注 |
|---|---:|---:|---:|---|
| `e9zR4mvMWw7` | 1 | 0 | censored | B 600 steps 后距目标 3.158 m |
| `gxdoqLR6rwA` | 1 | 0 | censored | B 600 steps 后距目标 17.692 m |
| `dhjEzFoUFzH` | 0 | censored | censored | A 未到达 |
| `gTV8FGcVJC9` | 1 | 1 | 0 | 提供一条合法 `A∧B` 的 Revisit-C 分母 |

汇总为 A `3/4`、B given A `1/3`、C given AB `0/1`。这是实现 smoke，不是 SR 估计；尤其不能
从三条可评 B 推断顺序效应。但它证明修复后的协议能够产生非零且语义干净的 C 条件分母。

### 同一 gTV episode 的三臂因果审计

A/B 在三臂中使用完全相同的 seed、server-selected candidate、steps、路径长度与终点距离：
A 为 81 steps / 2.9724 m，B 为 453 steps / 16.1458 m。因此差异只发生在 C。

| C controller | C | steps | path | final dist | 自动激活 A/B/C |
|---|---:|---:|---:|---:|---:|
| native ImageGoal | 0 | 518 | 17.608 m | 11.145 m | — |
| known-role raw-DINO direct + metric residual | 1 | 135 | 4.028 m | 0.996 m | known C only |
| role-free certified relocalization + fixed-bearing adapter | 1 | 86 | 2.824 m | 0.970 m | `0/11`, `0/57`, `11/11` |

certified arm 在 A 没有因果候选时拒绝，在 Novel B 因 fundamental-inlier precheck 不足而拒绝，
两段都精确回退 native；在 Revisit C 的 11 次 plan 全部通过 certificate 并接管。这个单例同时
验证了 open-set、fail-closed、scale-free bearing 和 mixed NavDP 执行链路，但 **N=1 只属于集成
证据，不能写成方法增益或成功率结论**。

最初 gTV 的多样本生成进程在第一条完成后被人工中断，因此另用相同固定 seed 正常生成 `n=1`
完成副本。生成器退出码为 0，并写出 `generation_summary.json`；新旧 episode 各 733 个文件，
相对路径和 SHA-256 全部一致。闭环结果因此可追溯到完整生成回执。

证据目录：

- 四场景生成、审计和 native：`.diagnostics/multigoal_v4_quickpilot_r1_20260812/`
- gTV 完整生成回执：`.diagnostics/multigoal_v4_quickpilot_r1_clean_20260812/`
- known-direct / certified：前一目录下的 `method_eval/`

### 当前唯一合理的正式下一步

不应把四条 smoke 扩写为结论，也不应返回旧 3-leg 数据。下一阶段应冻结 strict-v4 manifest，
在不相交多场景上扩充 episode，并以 `A∧B` 为唯一 C 分母，同机同进程比较 native、known-direct
和 role-free certified；同时报告 A/B false takeover。正式规模由先导数据的 `A∧B` 分母决定，
而不是预先把“40 episode”当成充分样本量。
