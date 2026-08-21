# Role-free online-memory benchmark（冻结候选协议，2026-08-14）

## 1. 要回答的唯一问题

在冻结 NavDP 完成一次真实 Novel Goal-A、形成因果 RGB 历史后，同一个自动记忆模块能否：

1. 在有历史视觉支持的 Revisit 查询上带来闭环收益；
2. 在无历史视觉支持的 Novel 查询上拒绝错误接管；
3. 相对“永远相信 raw-DINO 检索”的简单强基线，形成更好的 risk–coverage 取舍。

这不是顺序 `A -> B -> C` joint benchmark。Novel 与 Revisit 查询都从完全相同的 frozen
online-A 终点独立启动，以免 Novel-B 失败缩小 Revisit-C 分母，也避免 B 轨迹污染 C 的记忆。

## 2. 数据单元与因果契约

每个样本单元是一条由 native NavDP 实际执行并逐帧哈希复现的 Goal-A 历史。expert 轨迹不进入
运行时记忆。每条合格历史只保留一个匹配对：

- `Revisit`：在 online-A 历史中的 eligible 区间最大 3D co-visibility 位于 `[0.50,0.98]`；
  目标相对历史帧有 `0.20–0.50 m` 位移、`10–25°` 视角扰动，不是重复 JPEG；
- `Novel`：同楼层、全 online-A 历史最大 3D co-visibility `<0.10`；
- 二者从 online-A 终点的 geodesic distance 相差不超过 `0.50 m`；
- 二者最短路第一段 bearing 相差不超过 `30°`；
- 查询 geodesic 位于 `[2,9] m`。

如果较晚的受控 Revisit anchor 无法构造匹配 Novel，则按冻结顺序尝试较早 anchor。该选择只看
几何/可见性构造约束，发生在任何 policy rollout 之前；所有失败尝试和随机种子写入 receipt。
一个历史不重复塞入多个相关 pair，扩样依靠增加独立 online-A 历史。

## 3. Role-free 的严格含义

`analysis_role`、co-visibility 曲线、GT 目标位置及构造诊断仅存在于 evaluator sidecar。
运行时投影只允许：query id、目标 RGB/depth 及其哈希、渲染位姿（仅 evaluator 计算 SR）。
policy/server 请求只收到当前 RGB、goal RGB、depth 和自身因果记忆，绝不收到 Novel/Revisit 标签。

每个 `(history, query, arm)` 都执行：fresh server reset → exact online-A replay → 单个 query
rollout。online-A replay 不采样 diffusion；NavDP 只恢复原始 decision frames，MemNav 恢复所有
物理帧。三臂的起点、goal bytes、seed、checkpoint 和 diffusion plan seeds 必须相同。

## 4. 冻结四臂

1. `native`：纯 frozen ImageGoal NavDP，无长期记忆。
2. `raw_direct`：每个查询都使用 raw-DINO top-1 memory proposal 和现有直接 pose residual；
   不读取 role，也不做 certificate。它刻画最大 coverage 及其 Novel 风险。
3. `raw_fixed_bearing`：与 `raw_direct` 使用同一个无证书、always-on raw-DINO top-1 pose
   proposal，但丢弃 metric norm，归一化到与 certified 相同的 fixed 2.5 m bearing，再调用相同
   frozen mixed controller。它只用于隔离 controller input scale，不是安全方法。
4. `certified`：DINO temporally-diverse top-8 → SuperPoint/LightGlue/Fundamental ranking →
   LingBot depth + PnP → 固定原子 certificate；仅 accept 时输出 scale-free bearing，经固定
   `2.5 m` residual 交给 frozen mixed NavDP；reject 当步精确回退 native ImageGoal。

禁止 graph rescue、CDEC learned rescue、X-NavDP、oracle selector、frontier/bearing oracle、阈值扫描。

## 5. 预注册指标

### Primary：risk–coverage 分解

- Novel harm：相对 native 的 paired gain/loss、McNemar、false-takeover episode/plan rate；
- Revisit utility：相对 native 的 paired gain/loss、conditional SR、certificate activation recall；
- certified 相对 raw-direct：分别在 Novel 和 Revisit 子集报告 paired gain/loss；
- balanced mixed SR：每个角色等权，而不是被某个角色样本数主导。

Novel 上 certificate reject 的 episode 必须与 native 在 executed rollout 和 memory trace 上精确一致；
只比较 success bit 不足以证明 fail-closed。统计以 query 配对，置信区间按 scene cluster bootstrap。

### 解释边界

- 构造 smoke 不形成 SR 结论；
- train/consumed 场景只能用于接口、功效和运行时估计；
- final-reserved 场景只允许在代码、arm、阈值、分析脚本和排除规则全部冻结后一次性运行；
- 不以 final 结果回改方法。

## 6. 2026-08-14 consumed-scene construction smoke

在四条已消费 online-A 历史上构造 `4 pairs / 8 queries / 4 scenes`，独立 renderer-free 审计通过：

| 审计量 | min | median | max |
|---|---:|---:|---:|
| role distance error | 0.005 m | 0.237 m | 0.411 m |
| initial path bearing error | 0.00° | 0.72° | 29.68° |
| Novel max online-A covis | 0.000 | 0.0006 | 0.0957 |
| Revisit max eligible covis | 0.750 | 0.813 | 0.832 |

所有源/目标 RGB、depth、online-A receipt/trace、sidecar 和 manifest 哈希一致；运行时投影不含 role
或 co-visibility。该结果只证明 benchmark 可构造，不说明任何方法 SR。

## 7. 冻结产物

- contract：`MemNavData/shared_online_role_pair_contract.py`
- builder：`MemNavData/build_shared_online_role_pairs.py`
- independent auditor：`MemNavData/audit_shared_online_role_pairs.py`
- contract tests：`MemNavData/test_shared_online_role_pair_contract.py`
- builder tests：`MemNavData/test_build_shared_online_role_pairs.py`
- smoke：`.diagnostics/shared_online_role_pair_heading30_v3_smoke_20260814/`
- manifest SHA256：`123ddfcb047653d0fceed1be51aacba32a58b6f8e2f5656dbac47f672993de88`

下一门不是 held-out SR，而是用这四个 consumed pair 做三臂 closed-loop integration smoke，确认
exact replay、role hiding、raw-direct 语义和 certified fallback 后，再冻结正式 runner/auditor。

## 8. 单场景 closed-loop integration smoke 更新

`gxdoqLR6rwA/episode_0000` 的 120-step 三臂 smoke 已通过独立审计：raw-direct 对 Novel 和
Revisit 均 `15/15` takeover；certified 对 Novel `0/15` accept/takeover 且 executed rollout 与
native 逐帧一致，对 Revisit `15/15` accept/takeover；无 runtime failure。所有 arm 在截断预算内
均未成功，故不形成 SR 结论。完整结果见
`MemNavData/SHARED_ONLINE_ROLE_PAIR_SMOKE_RESULT_20260814.md`。

该 smoke 同时暴露了正式比较的最后一个设计缺口：raw-direct 是 legacy metric input，而 certified
是 fixed-2.5m bearing。协议现已加入 `raw_fixed_bearing` 接口消融；必须在 consumed smoke 验证后
才可冻结 final runner，避免把 controller input scale 差异误归因给 certificate。
