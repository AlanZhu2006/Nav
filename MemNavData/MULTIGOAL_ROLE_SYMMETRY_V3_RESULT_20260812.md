# Multi-goal role-symmetry v3：修复与本机验证（2026-08-12）

> 已被后续的 role-paired v4 契约取代。v3 没有逐 episode 匹配 A/B
> geodesic，也没有保证 Revisit C 对整个 leg-B 历史为 hard negative。
> 最新结果见 `MULTIGOAL_ROLE_PAIRED_V4_RESULT_20260812.md`。

## 结论先行

旧 3-leg 结果不能用于证明“NavDP 做第二个 Novel 会天然退化”。第一段 A 和后段 Novel 在距离、初始朝向、目标图生成、历史状态和数据种子上并不等价；其中最严重的是后段 Novel 没有最大距离限制。

本次已修复所有已定位的**基准构造与归因问题**，并用真实 MP3D/Habitat 数据和官方冻结 NavDP checkpoint 做了本机闭环小测。修复后的生成数据 4/4 通过 fail-closed 契约。小样本没有复现“清空 NavDP FIFO 会提升 B”的现象，因此 FIFO reset 只保留为消融，不能改成默认方法。

这次修复提高的是实验的可解释性，而不是宣称获得了新 SR 增益。N=4 仅是实现冒烟测试，不是统计确认。

## 旧实验为什么不公平

第三方标准 NNR 数据曾得到：A `69/100`，Novel B|A `16/69=23.2%`，Revisit C|AB `13/16=81.3%`。角色交换 NRN 曾得到：A `66/100`，Revisit B|A `54/66=81.8%`，Novel C|AB `9/54=16.7%`。但两批数据分别使用 seed 777 和 4242，不是配对角色交换。

对原始 episode 的审计显示：

- 标准 NNR 的 A 测地距离均被限制在 3–9 m；B 平均 16.41 m、中位数 11.64 m、最大 67.49 m，`42/69` 条可评 B 超过 9 m。
- B 在 9 m 内为 `14/27=51.9%`，超过 9 m 只有 `2/42=4.8%`。距离本身已能解释大部分表面落差。
- A 初始朝向被路径预对齐，初始 bearing 绝对误差均值 28.4°；B 到达 A 后的初始 bearing 绝对误差均值 115.2°。
- A 的目标图是专家轨迹真实终点帧；旧 B 目标图按采样目标位姿另行渲染，而且生成器实际不执行终点朝向对齐。B 的渲染 yaw 与专家真实到达 yaw 平均相差 27.2°。
- A 没有历史，严格说应标为 `initial_imagegoal`；B 才是相对已有历史的 `novel`。旧标签把两个不同操作任务都称为 Novel。
- 角色交换实验使用另一批 seed；后段 Novel C 还要对更长的 A+B 历史满足 non-covisibility，因此也不是对 A 的单变量交换。

第三方 Revisit 与本项目属于同一构造族：leg-A anchor、1.5 m 位置扰动、45° heading envelope、co-visibility gate、long-term gap 和因果 LingBot 历史。但第三方 C 位于长 Novel B 之后，使用旧数据、旧 router/checkpoint 和不同上下文；其约 81% 不能和当前 fresh 2-leg Revisit 的约 92–93% 直接混算。

## 已完成的修复

### 1. role-symmetric 3-leg 数据协议

`generate_twoleg.py` 的 3-leg 默认协议现在是 `multileg_v3_role_symmetric_20260812`：

- A 与 B 使用相同的 3–9 m 测地距离带；B 新增 9 m 上限。
- 3-leg 的初始 yaw 默认均匀采样，不再预对齐第一段最短路径；2-leg Revisit 默认行为保持不变。
- A 的 metadata 起点绑定到第一个真实保存帧，不再使用未写入数据的“前一步”位姿。
- A 的目标位置绑定到专家真实终点帧。
- B 的目标位置、目标 yaw 和目标 RGB 全部绑定到专家真实终点帧。
- B 重绑定后重新检查距离和对历史的严格 non-covisibility，失败则丢弃整个 episode。
- Habitat simulator、pathfinder 与 NumPy RNG 都记录并设置 seed。
- metadata 写入明确的 role sequence、距离带、起点/目标来源、初始 heading offset 和 generation seed。

修复起点定义时额外发现：旧 metadata 从未保存的 pre-step pose 计算 A 距离，和实际评测首帧约差 3.8 cm；边界 episode 会实际低于 3 m 却被接受。v3 现在从真实首帧重新计算并重新过 3–9 m gate。

### 2. fail-closed 契约与独立审计

新增 `multigoal_benchmark_contract.py`，逐 episode 检查：

- 协议版本和角色顺序；
- 初始 yaw 模式；
- A/B 相同距离带及实测距离；
- metadata 起点与首个存储帧一致；
- A/B metadata 目标与专家终点一致；
- B 目标 yaw 与专家终点 yaw 一致；
- `goal_1.jpg` 与 B 专家终点 JPEG 逐字节一致；
- stored/measured geodesic 一致；
- 任意 NaN/Inf 均 fail-closed。

新增 `audit_multigoal_role_symmetry.py`，可在不运行策略时审计 Parquet、metadata 和 JPEG。`eval_3leg_habitat.py` 默认拒绝旧/不合规 3-leg 数据；只有显式指定 `--allow_legacy_multigoal_data` 才允许复现旧混杂基准，并会在输出中记录 contract failure。

### 3. 正确角色标签与配对反事实

3-leg 输出现在明确标记：

- A：`initial_imagegoal`
- B：`novel`
- C：`revisit`

`eval_3leg_symmetry_habitat.py` 新增 `b1_role_matched`：把同一个 B 作为 fresh-FIFO 的第一目标，在专家 leg-2 上选择与 A 距离匹配的起点，并复制 A 相对首段路径的 heading offset。它同时匹配距离、目标图构造、初始朝向难度和 diffusion seed，用于隔离“目标本身”与“顺序/on-policy 状态”。

### 4. FIFO 状态变成显式消融，不暗改方法

保留三种清晰配置：`carry`、`before_b`、`every_goal`。reset 只清 NavDP 的 8-frame 短期 FIFO，不清 LingBot/MemNav 长期记忆。由于当前小测不支持 reset 增益，默认仍为 `carry`。

## 本机真实数据验证

环境：RTX 4090；MP3D scene `e9zR4mvMWw7`；seed `20260812`；官方冻结 checkpoint `navdp_checkpoint.ckpt`；每段最多 600 steps；server 内同进程按确定性 plan seed 配对。

### 数据契约（4 条 3-leg）

- 有效：`4/4`，无 contract issue。
- A geodesic：min 3.041、mean 3.889、max 5.024 m。
- B geodesic：min 3.144、mean 3.986、max 5.288 m。
- metadata 起点对首帧误差：4 条均 0 m。
- A/B 目标位置对专家终点误差：4 条均 0 m。
- B yaw 误差：最大约 `2e-6°`。
- B 目标 JPEG 与专家终点 JPEG：`4/4` 逐字节一致。
- 初始 heading offsets：`129.4°, -32.3°, 53.1°, 24.0°`；这只证明实现没有继续逐条路径预对齐，N=4 不用于检验总体均匀性。

另生成 1 条历史 2-leg Revisit episode 成功，确认 3-leg 修复没有破坏 2-leg 默认生成路径。

### 冻结 NavDP 闭环冒烟

| 配置 | A | B\|A | C\|AB | joint |
|---|---:|---:|---:|---:|
| carry FIFO | 2/4 | 1/2 | 1/1 | 1/4 |
| reset before B | 2/4 | 1/2 | 1/1 | 1/4 |

A 的逐 episode 结果在两臂完全一致。B 成败也一致；失败 episode 的终点距离会因 FIFO 不同而变化，但没有跨过 1 m 成功阈值。因此之前另一批 N=4 smoke 中 reset 的表面增益没有稳定复现。

### role-matched 冒烟

- `b1_role_matched`：`3/4`。
- 顺序执行的 `b2_executed`：A 成功后可评 `2` 条，其中 B `1/2`。
- 在这 2 条共同可评 episode 上：两臂一条都成功、一条都失败，配对 `+0/-0`。

这个结果只说明新反事实执行链路有效，且当前 N=4 没有证据支持“纯顺序效应”或“FIFO 是主因”。它不推翻旧 100 条数据中的巨大构造混杂，也不能估计正式 SR。

## 测试清单

- Python compile：相关 7 个脚本全部通过。
- Unit tests：`17/17` 通过，覆盖协议、旧数据拒绝、距离不对称、目标位姿/JPEG、真实首帧起点、NaN/Inf、goal-switch reset 路由及错误 server fail-closed。
- 真实 Habitat 3-leg 生成：`4/4`。
- 离线严格审计：`4/4`。
- 冻结 NavDP carry/reset 闭环：均完成 `4/4`。
- role-matched 闭环：两臂均完成。
- NavDP server 已正常关闭，GPU 无残留进程。

证据目录：`.diagnostics/multigoal_v3_final_smoke_20260812/`。

## 当前可以和不可以下的结论

可以：

1. 旧的“第一段约 69%，第二段 Novel 约 17–23%”不是干净的 NavDP role/order 效应；后段任务在旧协议中远距离、背向和目标图不一致等方面显著更难。
2. v3 已消除目前定位到的静态构造不对称，并能 fail-closed 阻止旧数据混入正式比较。
3. 当前没有证据把 FIFO reset 升格为方法修复。

不可以：

1. 不能从 N=4 报告修复后的正式 A/B SR，也不能称 role symmetry 已统计确认。
2. 不能断言 NavDP 完全没有 goal-order/state degradation；需要在 v3 的不相交多场景样本上做配对检验。
3. 不能把第三方 3-leg Revisit 的 81% 与 fresh 2-leg 的 92–93% 当成同分布升降。

## 唯一合理的下一步

正式实验只需运行一次冻结设计：在不相交的 20 scenes / 40 episodes 上生成 v3 数据，同机同进程运行 `b2_executed` 与 `b1_role_matched`，并附带 `carry`/`before_b` 作为预注册消融。主检验是共同可评 episode 的配对增减与 scene-cluster bootstrap；在此之前不修改冻结 NavDP，也不再解释旧 100 条混杂数据的 SR 差为模型缺陷。
