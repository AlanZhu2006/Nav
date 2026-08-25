# Multi-leg 帧上限审计(2026-08-22)

5-leg 正式评测的第一个前置:搞清 episode 超过 `MEMNAV_MAX_FRAME_NUM` 时
LingBot 流到底发生什么。

## 结论:不是崩溃,是静默损坏

`max_frame_num` 同时决定 3D-RoPE 频率表行数(`stream.py:262`)和 FlashInfer
KV 管理器的 `max_total_frames`(+100)。RoPE 前向用 Python 切片取时间维频率
(`rope.py`,`freqs[0][f_start:f_end]`),**切片越界不抛错、静默截断**。

CPU 单元复现(按 stream.py 的真实调用形状,max_seq_len=48,pph=ppw=37):

| 情形 | 结果 |
|---|---|
| 界内 (40..41) | OK,32 个频率分量 |
| 边界 (47..48) | OK |
| 过界一帧 (48..49) | **不报错**,只剩 22 个分量——时间维整体为空 |
| 跨界 (47..49) | 27 个分量,部分截断 |

即超界帧的位置编码丢失时间维,下游要么在注意力深处炸出难归因的形状错误,
要么静默降级:此后检索、PnP、位姿全部建立在无时序信息的位置编码上,而所有
receipt 显示正常。附带发现:f/h/w 共用同一张表,`max_frame_num` 还必须
≥ 空间 patch 数(518/14=37),实际配置远超,无碍。

## 修正(操作者指出):flow-gate 使上限对 5-leg 实际不构成约束

gatecurr 时期的 interval 改进(flow-gated keyframe,`FLOW_TIERS` 按
episode 长度分档,>2048 步取 60px,`FLOW_GAP=30` 强制提交)正是为此设计:
`add_frame` 对非 keyframe 帧评估后整体回滚 KV 与
`total_frames_processed`(policy_agent 911–915 行),因此 **RoPE 位置只被
committed keyframe 消耗**;稠密 DINO/pose 记忆按原始帧号累积,不占 RoPE
预算。一个 2500 步的 5-leg episode 实际只消耗几百个 RoPE 位置,离 2048
很远;历史运行中实际使用过的 RoPE 位置也远低于原始帧数,"RoPE 外推未经
验证"的担忧同步缩小。

## 已落的护栏(按正确计数器)

`policy_agent.add_frame` 在 `lb.agg.total_frames_processed >=
max_frame_num` 时 fail-closed 抛明确错误——静默损坏路径被封死,同时不会
误杀被门控的长 episode。属性缺失时自动退化为无操作;39 tests passed。

## 对 5-leg 的剩余前提(收窄后)

1. reset 时必须传**整个 multi-leg 的真实总长**作 `episode_len`,否则
   auto 档退回 20px 软门,keyframe 率显著升高(仍安全,但余量变小);
2. `flow_gate=off` 的配置不允许用于 >2048 步的 episode(护栏会拦住);
3. 帧上限不再是 5-leg 的阻塞项;剩余前置只剩 k 段构造合约与 population
   决策。
