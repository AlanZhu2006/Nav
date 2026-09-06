# Monocular Route-Tangent Compass：冻结确认协议

状态：**已冻结，尚未读取确认行的任何 route-tangent 输出；闭环 SR 未授权。**

日期：2026-09-03

## 1. 要回答的问题

长距离 Revisit 中，逐帧单目相对运动本身很准，但将这些增量积分成一个全局查询位置后，横向漂移会不断累积。旧的 pose-chord readout 从“漂移后的当前位置”指向历史路线前视点；因此即便路线进度估计正确，横向漂移仍会旋转最终 bearing。

本实验只检验一个更小、更符合任务需求的假设：

> 保留二维投影来估计当前处于历史路线的哪个进度，但从该进度处的历史路线切向直接读取方向，不再让累计横向位置漂移进入 bearing。

该实验不是导航成功率实验，也不训练模型。只有预注册机制门通过后，才允许把该 readout 接入 frozen NavDP 并建立新的闭环配对池。

## 2. 固定方法

设由 causal RGB、LingBot 单目深度和相邻帧 PnP 得到的反向历史路线为 \(R(s)\)，总弧长为 \(S\)。查询阶段只累计相邻帧单目 SE(2) 增量，得到估计位置 \(p_t\)、朝向 \(\theta_t\) 和实际视觉路程预算 \(L_t\)。

路线进度仍由单调二维投影决定：

\[
s_t = \arg\min_{s\in[s_{t-1},\,\min(S,L_t)]}
\lVert R(s)-p_t\rVert_2.
\]

方向不再使用 \(R(s_t+\rho)-p_t\)，而固定为路线自身的前向 chord：

\[
\hat b_t = \operatorname{normalize}\!\left(
R(-\theta_t)\,[R(\min(S,s_t+\rho))-R(s_t)]
\right),\qquad \rho=2.5\,\mathrm m.
\]

到达路线末端时，保持最后一段长度不超过 \(\rho\) 的路线切向。传给 NavDP 的仍然只是 \(2.5\hat b_t\)。

固定契约：

- 单目 causal RGB；
- 唯一尺度先验为固定相机安装高度；
- 不读取 simulator depth、GT pose、轮速里程计、Novel/Revisit 标签或成功结果；
- 不增加距离分档、confidence gate、endpoint fallback 或 native fallback；
- CEC 只负责初始历史 anchor 的既有严格授权；
- frozen NavDP 不参与本机制审计。

## 3. 已消耗的开发证据

### 3.1 Index 16：开发行

- translated bearing：67/72 在 30° 内；
- 中位误差 7.383°，P90 22.977°；
- terminal-turn 中位误差 8.629°；
- artifact SHA256：`d666d48094dbdf4dbd51c6375d115b65ecfbea4dc97909d127279e74954d4b60`。

### 3.2 Index 40：仅 post-hoc 诊断

Index 40 已被旧方法的 prospective gate 消耗，不能作为确认结果。Route-tangent 的事后回放为：

- translated bearing：81/99 在 30° 内；
- 中位误差 17.437°，P90 41.230°；
- artifact SHA256：`dcfb5d1684d123045c97fb7e19bfd72108ec56acd35e368473df2f4ead1a49a6`。

## 4. 确认行在输出生成前的冻结

来源 population manifest：

- 路径：`.diagnostics/long_range_path_field_20260902/source/population/manifest.json`
- SHA256：`cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451`

选择规则在读取任何新方法输出前固定为：

1. 仅考虑冻结的 `30_to_50_m` 档；
2. 排除已用于 route-tangent 开发/诊断的 population index 16 和 40；
3. 选择 `online_a_steps` 最大者；
4. 如并列，选择最小 population index；
5. 选定后才读取旧实验中已经冻结的首步 strict-certificate anchor；若它未通过严格 certificate，则本实验直接 blocked，不另挑样本。

冻结结果：

- population index：39；
- history ID：435；
- scene：`LT9Jq6dN3Ea`；
- causal history：1948 帧；
- construction/scorer-only 回程长度：35.9106 m；
- 既有 strict-certificate anchor：67；
- freeze receipt：`.diagnostics/mono_adjacent_motion_20260903/route_tangent_confirmation_freeze_v1/confirmation_freeze.json`；
- freeze receipt SHA256：`3fb0c2d1401107e43230bf077f3fa8d383f4b41c99da80db59fd7c56059ba3cc`。

冻结 receipt 明确记录：`route_tangent_output_read=false`、`navigation_outcome_read=false`、`sr_read=false`。

## 5. 预注册机制通过门

所有条件必须同时满足：

1. history 与 query 的相邻运动链均完整；
2. 所有运行时信息边界标志均为 false：metric depth sensor、global pose、wheel odometry、role label、navigation outcome；
3. 历史路线预测长度和查询视觉路程相对 scorer 路程的绝对偏差均不超过 15%；
4. 历史路线 endpoint error 不超过 1.25 m；
5. translated query 上，route-tangent bearing 至少 80% 在 30° 内；
6. translated bearing 中位误差不超过 20°；
7. 路线进度严格单调，且从不超过累计查询视觉路程预算；
8. 所有输出给 controller 的 PointGoal 范数严格为 2.5 m；
9. 不存在 distance regime、visual gate、endpoint fallback 或 native fallback。

查询累计 absolute endpoint/cross-track drift 仍完整报告，但不作为单独门槛：它不再直接进入 route-tangent bearing；第 5、6 条直接检验最终方向是否仍被投影或 yaw 误差破坏。该口径在确认输出生成前冻结，不能根据结果修改。

## 6. 固定执行顺序

1. 用 Habitat pose 只构造物理连续的 reverse-route RGB scorer 序列，并封存 query-set SHA；
2. 只向单目运行时发送 RGB，生成 query adjacent-motion receipt；
3. 独立重放 causal history，生成 anchor-to-tail 历史路线 receipt；
4. 在已封存的两个 motion receipt 上运行一次 route-tangent replay；
5. 独立 gate 脚本一次性判定第 5 节全部条件；
6. 若失败：停止，不调阈值、不运行 controller、不报告 SR；
7. 若通过：只授权实现 runtime integration 和构建新的 fresh paired population。当前单行不得冒充论文闭环确认。

## 7. 结果占位

尚未运行。任何结果必须同时给出：样本身份、全部输入/输出 SHA、通过门逐项布尔值、方向误差分布、路径尺度偏差、endpoint/cross-track 描述量，以及是否授权后续闭环。
