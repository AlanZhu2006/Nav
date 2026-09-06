# 2026-09-06：论文 SPL 真实位移复算与修正

本次属于审计后的计分修正，不是新的策略实验。没有改变任何 SR、配对样本、
成功阈值、随机种子、模型、统计检验或原始冻结结果。

2026-09-07 最新决定：作者要求“不要删除 SPL，还是保留”。活动论文 Table III 已恢复
SPL 列，采用下表实际位移积分的上下界，向外取三位小数；没有恢复不准确的旧单值。
全部 SR 不变，Table II 继续使用精确 SPL。最新实施见
[PAPER_SPL_RETAINED_20260907.md](PAPER_SPL_RETAINED_20260907.md)。

## 1. 确认的问题

旧 `eval_2leg_habitat.py::pursuit_step` 在正常/creep 分支返回指令步长
`v` / `0.3*v`，但实际新位置已经经过 `snap_point`。障碍附近实际位移可能小于
指令位移；后续累加这个返回值会使路径长度偏大、SPL 偏低。

旧 verifier 确认的是 CSV 与 trace 中的同一累计量一致，没有再次积分逐动作位置。
因此“旧 verifier 通过”不能排除这个共同来源的计分错误。

当前 worktree 的执行器已由此前工作修为实际 x-z 位移返回值；本轮保留该改动，
不把其他运行时变更混入已完成的正式实验。

## 2. 复算定义

`executed_path_metrics.py` 按连续 `step` 的实际 x-z 位置计算折线路径长度。
每条保存的 pose 是动作前位置，因此必须补上单独记录的最终落点；不能把最后一次
动作当成零。如果有更长诊断轨迹，则只计到首次成功的 step，不计之后的动作。

SPL 使用原 geodesic 和原 success 标签：

`success × geodesic / max(geodesic, executed_planar_path)`。

沿用本项目平面路径约定，不把本次处理包装成新 benchmark 指标或新性能增益。
缺失不止一个动作、step 不连续或非有限坐标会报错，不估算完整轨迹。

## 3. Table II：全部可精确修正

196 条 A + 183 条 B + 80 条配对 C，共 **459 条**，都有完整终点。

| 表格单元 | n / 成功数 | 旧 SPL | 实际位移 SPL | 论文三位小数 |
|---|---:|---:|---:|---:|
| 第一目标 Novel，共享前缀 | 196 / 131 | 0.601056484 | 0.618217330 | 0.618 |
| 第二目标 Novel，共享前缀 | 183 / 54 | 0.192110783 | 0.200019183 | 0.200 |
| 第三目标 Novel，native | 20 / 4 | 0.172272538 | 0.175580274 | 0.176 |
| 第三目标 Novel，CEC | 20 / 4 | 0.172272538 | 0.175580274 | 0.176 |
| 第三目标 Revisit，native | 20 / 8 | 0.142183244 | 0.148914175 | 0.149 |
| 第三目标 Revisit，CEC | 20 / 17 | 0.674960108 | 0.761541225 | 0.762 |

CEC/native 的 `8/20 → 17/20`、`+10/−1`、`p=0.0117` 全部不变。
已修改活动论文 `tables/continual_meeting.tex` 和 `sec/6_results.tex`；
原始 sealed CSV、旧 summary 和旧 verifier 没有覆盖。

## 4. Table III：SR 有效，精确 SPL 暂不可恢复

Final14 四臂 + zero-depth 共 210 条日志没有保存最终位置。逐动作位置完整到
最后一次动作之前，但不能从一个轨迹 SHA 或最终到目标距离唯一恢复最终坐标。

冻结执行器的 `v_max=0.0376 m`，允许的平面 snap 偏移不超过 `0.06 m`。
因此利用三角不等式，把唯一缺失动作长度限定在 `[0,0.0976] m`；由此可得
**数值上下界，而不是置信区间，也不是精确 SPL**：

| 深度 / 历史 | SR（不变） | 旧 SPL | 实际路径 SPL 下界 | 上界 |
|---|---:|---:|---:|---:|
| Metric / native | 11/42 | 0.114059 | 0.118064 | 0.119403 |
| Zero / native | 4/42 | 0.058597 | 0.060141 | 0.060220 |
| Mono / native | 10/42 | 0.121984 | 0.121642 | 0.122565 |
| Metric / CEC | 26/42 | 0.421858 | 0.443794 | 0.453478 |
| Mono / CEC | 28/42 | 0.474209 | 0.492901 | 0.503471 |

2026-09-07 初次处理删除了 SPL 列；作者随后要求保留，当前表格已改为明确标注的
`SPL bounds`，未用区间中点替代单值。额外追查覆盖原 210 个 plans JSON 与 105 个 CSV，哈希均与修正产物一致；
210/210 的 rollout 与 memory 位置都截止于 `steps - 1`，未找到额外终点字段。
样本的 evaluator / MemNav / NavDP 日志也没有终点坐标。结论限于已检查的保存产物，
不声称穷尽所有机器上可能存在的副本。上表上下界已用于论文的 SPL 列，但不称精确 SPL。

无需为恢复一个次要指标，立刻重跑已完成的 210 次闭环。是否重跑应由作者决定。

## 5. 代码、原始依据与验证

- 通用积分：`MemNavData/executed_path_metrics.py`。
- 回归测试：`MemNavData/test_executed_path_metrics.py`，10 passed。
- 全量离线重算：`MemNavData/recount_paper_executed_spl.py`。
- Table II 前缀仍先通过原 `independent_verify_hm3d_table2_stage_spl.py`，
  绑定原 parent/union/completion/trace SHA，然后独立积分位置。
- 修正文件包含全部 669 行的源文件路径、SHA、原数值和重算值/边界。

远端原始修正文件：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/paper_executed_spl_correction_20260906/correction.json
```

SHA-256：

```
41bd2c68c498863d5cf92452b2a7564d517698c38d26918ce7820b39cd49a942
```

本地同 SHA 副本：`.diagnostics/project_repair_20260906/correction.json`。

本次没有修改 title、abstract、introduction，没有 commit 或 push。
