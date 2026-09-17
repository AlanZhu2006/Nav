# 新版native-B归档：连续导航前缀可复用性检查

2026-09-09。仅检查已完成的固定query index 1，不构造C、不提交新实验，
也不改变当前90个查询。目标是核实后续Table II是否一定要重新采集A/B。

对象：source index 100，b28CWbpQvor，`novel_rear`，作业`17263230_1`。
顺序为native/GEM/raw，GEM没有接管，独立verifier确认与native逐动作一致。
这里的“验证通过”不表示导航到达：native和GEM该项均未到达。

## 已实际找到的数据

| 内容 | 原始证据 |
|---|---|
| 实际A历史 | query plans中保存192个A观测位姿，与封存新A绑定 |
| 实际native-B轨迹 | 267个逐动作观测，含step、x/y/z、yaw与`jpg_sha256` |
| native-B的全部RGB | 267/267个观测SHA均在已逐成员回读的archive索引中匹配到JPEG |
| 最终动作后的状态 | `terminal_measurements.json`保存`end_position`和`end_yaw_rad` |
| A/B及方法版本 | 原query manifest与population绑定，同一修复版c8运行栈 |

RGB不位于`evaluation/native/`子目录：该处只有首张查询PNG。
真正历史帧位于公共buffer的独立episode目录，本项匹配到例如
`buffer/ep_0001/192.jpg`。应按native逐动作日志中的图像SHA恢复，
不能因目录名没有native就认定RGB丢失，也不能直接取最后一个方法的buffer替换它。

这说明归档格式具备恢复真实A→native-B前缀的必要信息；
不是已经验证全部45条Novel的像素完整性，也不是已经得到可用C population。

## 为什么本项没有到达

该项实际行走4.4164m，在第267步结束。日志统计的runtime failure和geometry-stream stop均为0。
从原始`executor_actions.jsonl`重建冻结stuck规则：

- `stuck_window=150`、`stuck_dist=0.10m`；
- 首次满足该窗口条件恰为第267步；
- 窗口两端的平面净位移为0.0664163m。

因此这里有实际轨迹支持的停滞/循环终止证据，不是依赖崩溃或任务没有跑完。
0.0664m是窗口两端净位移，不是该窗口内的总路径长度。
没有改变这个所有方法共用的冻结终止规则，也不会为导航失败重跑同一个目标。

## 对必要补跑清单的影响

后续若更新连续导航，应先检查本轮固定45个Novel查询的native臂：

1. 只考虑原协议允许作为前缀的实际A/B结果，并完整报告失败与构造损耗。
2. 按逐动作SHA恢复原生B的全部RGB、位姿和终点，核对与A的连续性及输入版本。
3. 完整之后再冻结新的C构造/配对协议；不根据GEM-vs-raw成绩挑前缀。
4. 有了足够合法前缀时，可复用这批已经执行的新A/B，无需再跑相同的A/B导航；
   若不足，再明确额外来源和规模，不能放宽旧规则或直接拿失败B凑数。

**本项B失败，不能作为“成功A/B”的新C样本。**它只完成归档结构与恢复能力检查。
本轮没有新增连续导航结果，没有将其写入论文表格。

归档：
`/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_fullmono_actual_a_20260909/prefix_series_after_scene0_20260909/evaluation_cb812054dc470e53/task_001/artifacts.tar.gz`

归档SHA：`d30977c4ac606d842c6f373a7651f12f672f3b4d30bb0430fb980d2ab6c118dd`。
