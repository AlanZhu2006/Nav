# Shared-online double-Revisit pilot — 2026-08-12

## 结论

已从冻结 NavDP **真实执行的 online Goal-A rollout** 构造出 4 场景配对的
double-Revisit pilot。它不再把 expert A 当作运行历史。

- online A：4/4 到达，1059 帧；全部 RGB 重渲染哈希与原 rollout 一致。
- V0：B/C 都是 online A 中实际观察过的原始 JPEG + depth。
- V1：围绕相同历史位置做受控小位姿重渲染；保持强共视，但不是重复 JPEG。
- 构造和独立审计均通过；确定性复跑逐文件一致。
- **尚未产生闭环 SR**。当前结果只证明 benchmark asset 合法，不证明方法有效。

## 路径与哈希

- online-A source：
  `.diagnostics/shared_online_a_v0v1_pilot_native_20260812`
- paired V0/V1 benchmark：
  `.diagnostics/shared_online_double_revisit_v0v1_pilot_20260812`
- benchmark manifest SHA-256：
  `6101137a4655b5b457c38c46818130d3401fcf3f5897a211fcecfa3db58f4f5f`
- materializer：`MemNavData/materialize_online_a_traces.py`
- V0/V1 builder：`MemNavData/build_shared_online_double_revisit.py`
- independent audit：`MemNavData/audit_shared_online_double_revisit.py`

## 冻结构造口径

1. A 必须是冻结 NavDP 自己控制并成功完成的在线轨迹；router 不得接管。
2. B 取较晚的 online-A 记忆位置，C 取较早的位置，降低 A-end→B 穿过 C
   区域的风险。
3. B/C source frame 至少相隔 32 帧；候选记忆从 frame 39 开始，与部署时
   可用候选集合一致。
4. A→B 与 B→C 测地距离均至少 2.0 m。
5. V0 必须与 source online frame 的 JPEG/depth 逐字节一致。
6. V1 必须同时满足：
   - 平移 0.20–0.50 m；
   - yaw 改变 10–25°；
   - source-frame co-visibility ≥ 0.45；
   - eligible online-A 最大 co-visibility 在 [0.50, 0.98]；
   - 最大共视 anchor 距 source frame 不超过 20 帧；
   - RGB pixel MAE ≥ 5 且 JPEG 不同于 V0。

## Pilot 数值

| scene / episode | online-A 帧数 | source B/C | V1 B/C anchor gap | V1 A→B / B→C geo | V1 B/C max covis |
|---|---:|---:|---:|---:|---:|
| gxdoqLR6rwA / 0000 | 240 | 134 / 39 | 78 | 3.46 / 3.46 m | 0.818 / 0.750 |
| pLe4wQe7qrG / 0000 | 405 | 313 / 79 | 237 | 3.24 / 4.05 m | 0.808 / 0.815 |
| yqstnuAEVhm / 0001 | 213 | 123 / 39 | 63 | 3.43 / 2.59 m | 0.870 / 0.811 |
| mJXqzFtmKg4 / 0001 | 201 | 118 / 39 | 68 | 2.97 / 3.07 m | 0.832 / 0.552 |

V1 最终平移均约 0.30 m；yaw 为 12°或 18°；像素 MAE 范围为 6.46–63.06。

## 一次重要的审计修正

`pLe4wQe7qrG` 的 C 候选在全轨迹上对 frame 0 有更大共视，但 frame 0 不在
部署时可用的 memory candidate domain。若用全轨迹 argmax，构造会被错误拒绝；
从实际候选下界 frame 39 复算后，最大值位于 frame 82–83，与 source frame 79
一致。修正的是审计集合，而不是放宽 V1 阈值。

## 尚未回答的问题

- B、C 的闭环 SR 尚未运行。
- 实际 B rollout 是否再次观察到 C、从而把“长时记忆 C”变成 recent-memory C，
  必须在闭环后用 pose+depth 共视曲线审计。
- N=4 只是 pipeline pilot，不作统计或论文主张。
- V0 是 identity sanity arm；正式 benchmark 应以 V1 为主。

## 下一步唯一合理的小测试

对每条 episode 回灌同一份已哈希的 online-A 决策历史，再运行 B→C：

1. 先各跑 1 条 V0/V1 native，验证 evaluator、位置目标和 C 污染审计；
2. 通过后再跑 4 条 V0/V1 的 native 与 certified memory arm；
3. 只有确认 shared-A 完全一致且 online-B 对 C 是 hard negative，才报告 SR；
4. pilot 通过后再扩展场景，不需要为数据构造本身使用 HPC。
