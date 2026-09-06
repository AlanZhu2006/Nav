# 单目 CEC 长程 Revisit：根因审计与解决路线

日期：2026-09-03（Asia/Shanghai）  
状态：**进行中；静态、数据、测量、三臂机制归因和可部署 continuous local-route-tangent
实现均已完成；23-history fresh population 已在真实 HPC 数据上完成结果盲精确预审计，新正式
DAG 已提交，闭环结论尚待 GPU 与最终独立 verifier。**  
边界：本文档区分已经成立的事实、已消费样本上的机制线索、尚待确认的假设。任何单条
episode、oracle readout 或事后视频都不能成为论文结果。

## 0. 当前结论

长程失败目前不能被诚实地归结为“LingBot 漂移”或“2.5 m 太短”。证据更支持一条分层因果链：

```text
CEC 正确认证一个历史目标
        ↓
把整条长且弯曲、通常跨层的可行路线压缩成一个终点方向
        ↓
2.5 m geodesic lookahead 又被压缩成起点到 lookahead 点的一条弦
        ↓
弦在拐角处可能与第一个可行局部切向相差约 90°，甚至指向墙体
        ↓
mixed ImageGoal + PointGoal 条件下，NavDP critic 大量进入低分侧向搜索模式
        ↓
执行器重复转向/贴墙；旧 motion receipt 还会把未实现的命令位移记成前进
```

在一条已消费的 33.791 m 跨层失败上，冻结的三臂实验已经确认：缺失量不是 endpoint
range，而是**当前位置沿可行路线的第一个局部切向**。`chord / continuous tangent /
tangent-then-native` 的 3-D success 分别为 `0/1 / 1/1 / 1/1`。这是一条机制归因，不是
论文级扩样结果；它逐层区分了：

1. route geometry 的弦是否是主要原因；
2. 正确局部切向能否被 frozen NavDP 兑现；
3. PointGoal 是否应只负责纠正方向，而不应持续替代 ImageGoal 的局部控制作用。

continuous 与 selective tangent 的最终距离只差 `0.013 m`，critic 低分比例只差
`0.24 pp`，没有证据支持 20° 门控或切回 native。因此实现选择冻结为**连续 local
route tangent**；在 fresh 扩样完成前，canonical 论文方法仍是 fixed-2.5 m
endpoint-bearing CEC，不得把单条 oracle 机制结果写成方法增益。

## 1. 被审计的系统到底是什么

Canonical CEC 的长程分支不是路径规划器。它执行：

```text
Goal image + causal RGB history
  -> DINO top-8 temporal proposal
  -> SuperPoint + LightGlue + Fundamental-MAGSAC
  -> historical LingBot depth + PnP
  -> operational certificate
  -> current-to-goal relative vector v_t
  -> discard ||v_t||, retain b_t = v_t / ||v_t||
  -> PointGoal = 2.5 b_t
  -> frozen NavDP(ImageGoal, PointGoal, current RGB, monocular depth)
```

因此 canonical CEC 证明的是“这个历史目标假设有足够几何支持”，输出的是目标端点的二维
方向。它没有编码：走廊从哪侧绕、下一个门在哪、楼梯入口在哪、应沿哪个 homotopy 走。

这一区分解释了为什么短程结果很强，而长程可能崩溃：短程中 endpoint ray 常近似第一段
可行路径；长程中二者不再等价。

## 2. 长度压力测试的数据构造审计

### 2.1 冻结总体

- manifest：`.diagnostics/long_range_path_field_20260902/source/population/manifest.json`；
- SHA-256：`cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451`；
- 48 histories，三档各 16；
- 每档至少 10 个 scene clusters，单场景每档最多 2 条；
- runtime 不读取 Novel/Revisit role；
- Revisit query 要求 online history covisibility `>=0.55`；
- Novel query 要求最大 covisibility `<0.10`。

### 2.2 它不是 actual NavDP Goal-A rollout

history contract 是 `controlled_causal_rgb_geodesic_survey`：沿冻结的可行测地路线采集因果
RGB，而不是让 NavDP 自主完成 Goal-A 后留下的自然 history。这样做适合隔离“如果历史路线
确实存在，跨度增长会怎样”，但它不能冒充完整端到端 lifelong rollout。

所以该 48-history 总体的合法用途是：

- 长度/路线信息压力测试；
- 机制归因与方法淘汰；
- 为新的 fresh benchmark 冻结设计原则。

不合法的用途是：

- 宣称 actual-online A 后的端到端长程 SR；
- 把其结果直接并入 Final14 或 HM3D full-mono 主表；
- 在同一总体上反复开发后再称为泛化确认。

### 2.3 距离与跨楼层严重混杂

对冻结 manifest 直接复算起点 `online_a_endpoint.y` 与 Revisit 目标 `floor_position.y`：

| 冻结档位 | N | 高差中位数 | 高差范围 | 高差 >=0.5 m | 高差 >=1 m |
|---|---:|---:|---:|---:|---:|
| 约 7--10.6 m | 16 | 0.038 m | 0--4.02 m | 3/16 | 3/16 |
| 约 20.1--24.9 m | 16 | 3.00 m | 0--8.72 m | 13/16 | 13/16 |
| 约 30.1--39.7 m | 16 | 3.34 m | 0.60--11.60 m | 16/16 | 14/16 |

可复现产物：

- `audit_hm3d_longrange_floor_confound.py`；
- `.diagnostics/longrange_floor_confound_20260903/audit.json`；
- result SHA-256：`d2b52889ecca0325b426150e0282e535b41054fc90397cfed4fab71379205433`；
- test：`test_audit_hm3d_longrange_floor_confound.py`，`1 passed`。

这意味着旧表不能被解释为单变量的“长度曲线”。距离越长，跨层比例也越高；同时拐角数、
路径曲率和 homotopy 难度通常同步增加。准确表述应是：

> CEC 在一个随 geodesic span 增长、且越来越多跨层/高曲率路线的压力总体上退化。

后续 fresh 设计必须至少按 `same-floor / multi-floor` 分层；若论文只主张室内地面机器人，
主结果应优先使用 same-floor long-range，跨层另列诊断，不能把两者混成“距离”。

此外，canonical CEC、旧 episodic path field 和 NavDP PointGoal 都是二维 `[forward,left]`。
旧 `geodesic_lookahead_world()` 与 route arc 也只累计 x/z 长度，不累计 vertical displacement。
NavMesh snap 可能让一条平面轨迹沿坡面换层，但 policy 输入本身没有“目标楼层”或竖直动作。
因此把 3-D success 修正确保了评分不再跨楼板误判，却不会凭空赋予 controller 跨层语义。
若 multi-floor 子集继续失败，这必须作为独立能力边界，而不是用更长的 planar residual 掩盖。

## 3. 已有闭环现象

### 3.1 Canonical 长度压力结果

| Revisit geodesic 范围 | native | canonical CEC |
|---|---:|---:|
| 约 7--10.6 m | 2/16 | 4/16 |
| 约 20.1--24.9 m | 1/16 | 2/16 |
| 约 30.1--39.7 m | 2/16 | 0/16 |

这证明 canonical endpoint bearing 在该困难总体上不够，但还不能告诉我们是哪一层失败。
尤其不能仅凭 `0/16` 推断：

- CEC 没找到目标；
- 单目尺度完全失真；
- frozen NavDP 没有长程控制能力；
- 只需把 2.5 m 改得更大。

### 3.2 主任务近距离并未显示相同问题

- Final14 Revisit：CEC `20/21`；
- HM3D full-mono Revisit：CEC `25/28`；
- Fresh160 高支持 Revisit：CEC `112/120`。

这些总体主要位于约 2--9 m，并且 supported Revisit 接近饱和。因此长程扩展应是一个明确的
外推问题，不能因为压力总体失败而推翻已确认的短程方法。

## 4. 测量合约审计：旧长程数字有两个隐藏混杂

### 4.1 旧 success 是平面 x/z 距离

`eval_2leg_habitat.py` 的传统 success 只比较当前和目标的 x/z。对同楼层任务这通常足够；
对多楼层任务，它可能把“平面位置接近、楼层错误”计为成功。当前已消费结果不能事后重写，
但后续长程实验必须同时记录：

- 3-D Euclidean final distance；
- vertical error；
- 最好再记录 pathfinder geodesic distance / floor identity。

冻结的 v2 单条机制实验已使用 3-D Euclidean `<1 m`，并保留 planar 与 vertical 分量。它比
旧 x/z 合约严格，但 fresh benchmark 仍应把 geodesic success 作为首选，避免薄楼板或坡道的
边界歧义。

### 4.2 旧 path length / executor receipt 不是实际位移

旧 `pursuit_step` 在 pathfinder 把候选点 snap 回原位置、但 snap 与微小命令点仍小于 6 cm 时，
返回的是命令速度 `v`，不是 snap 后真实位移。已消费 index 32 的四臂复算显示：

| arm | 日志 path | post-snap 实际 x/z path | stationary transitions | receipt mismatch |
|---|---:|---:|---:|---:|
| action-coordinate mixed | 83.665 m | 17.366 m | 1092/2365 | 1976 |
| oracle historical-route mixed | 85.209 m | 15.271 m | 1736/2365 | 1984 |
| oracle geodesic mixed | 74.619 m | 24.926 m | 11/2365 | 2110 |
| oracle geodesic PointGoal | 72.962 m | 24.041 m | 3/2365 | 2139 |

这不会改变已执行的物理 pose，但会污染 route clock、stuck 诊断、SPL 和“已经走了多远”的
解释。尤其 action-coordinate 方法直接消费该 receipt，因此它的离线机制通过与闭环弱增益
之间存在一个明确测量因果混杂。

v2 通过 monkeypatch 保持原运动完全不变，只把 receipt 改为：

```text
||post_snap_position[x,z] - previous_position[x,z]||
```

runner 再从 rollout pose 独立复算，并要求逐帧严格一致。

## 5. 分层根因审计

### 5.1 目标 proposal / certificate 不是当前失败点

在 index 32 的已消费归因中，真实 DINO proposal、local matching、PnP 和 strict certificate
在 296/296 planning decisions 上通过；各臂使用相同 anchor 与 proof。也就是说，对这条失败：

- 目标在历史中的地址已经找到；
- 目标假设有真实局部几何证据；
- 失败发生在“已认证目标怎样变成沿途控制”之后。

因此继续调 DINO top-K、LightGlue 阈值或 certificate coverage 不能解释这条长程失败。

### 5.2 Metric range 有信息，但不是缺失的关键控制量

First-40 相机高度尺度在 28 条实际 mono history 上的只读审计显示：

- 首次接管 metric range MAE `0.361 m`，median `0.254 m`；
- 28 条 query 内，metric distance 与 GT 的相关系数逐 query 中位数 `0.996`；
- 最后一次 readout 的 metric error 中位数 `0.195 m`。

所以“系统完全没有尺度”不成立。但把完整 metric magnitude 交给 NavDP 的配对闭环没有提升：

| arm | success | SPL | both-success mean path | mean steps |
|---|---:|---:|---:|---:|
| fixed 2.5 m | 25/28 | 0.715 | 3.557 m | 116.0 |
| full metric | 24/28 | 0.646 | 3.790 m | 123.3 |

配对为 `+1/-2`，McNemar `p=1.0`，scene-cluster CI `[-16.0,+7.69] pp`；479 次 metric takeover
均未触及 NavDP 的 10 m input cap。冻结决策是 `stop_keep_fixed_2p5m`。

这说明 metric norm 可估计，不等于 metric norm 对 frozen controller 有额外价值。更重要的是，
无论半径是 2.5、4 或 10 m，一个 endpoint ray 仍没有门、走廊、楼梯入口与绕障 homotopy。

### 5.3 历史路线本身确实含有缺失信息，但旧上限仍不是 first tangent

在 48 条已消费 history 上，用 evaluator pose 只做信息上限：

| 距离档 | endpoint bearing median error | ordered-route 2.5 m chord median error | <=30° coverage |
|---|---:|---:|---:|
| 约 7--10.6 m | 44.59° | 6.95° | 5/16 -> 14/16 |
| 约 20.1--24.9 m | 42.27° | 25.93° | 5/16 -> 8/16 |
| 约 30.1--39.7 m | 54.33° | 31.60° | 5/16 -> 8/16 |
| overall | 44.59° | 19.40° | 15/48 -> 30/48 |

33/48 改善、15/48 变差。实现复核表明，这里的 ordered-route readout 同样取路线弧长
`2.5 m` 处的参考点，再形成一条 chord；它不是第一段 path segment 的微分切向。因此该结果
只证明**有序历史路径比终点弦包含更多中程路线信息**；它不证明当前单目系统能恢复路线，
不证明 first tangent 已正确，也不保证原路返回总是最短 homotopy。

### 5.4 旧 “route-tangent confirmation” 实际仍是 2.5 m route chord

Index 39 的单目 route-tangent 确认得到：

- query adjacent chain：214/214；局部运动方向误差中位数 `0.837°`；
- history chain：1880/1880；预测长度 `31.637 m` vs scorer `34.011 m`；
- route position error median/P90：`0.490/0.819 m`；
- history endpoint error：`1.552 m`，超过预注册 `1.25 m`；
- 最终 translated route readout：90/105 在 30° 内，median `11.643°`。

预注册门因此严格记为失败，不能事后改判。更重要的是，代码审计确认
`SE2ProjectedRouteCompass` 取的是 `progress + 2.5 m` 处的参考点，再从 estimated position
指向该点。它仍是 **2.5 m arc-ahead chord**，不是 `epsilon -> 0` 的 first feasible tangent。
90/105 比较的是“预测 route chord”与“真值 route chord”的一致性，不能被引用为 local
geodesic tangent 已通过。

该实验仍然有价值：它证明单目相邻运动、route-coordinate 与坐标变换能在一条 34 m route 上
产生相当一致的 readout；同时显示全局 endpoint accuracy 与局部 readout accuracy 可以解耦。
若最终方法只消费第一段局部路线导数，1.552 m 的全局终点误差可能不是必要状态；但这个更强
主张必须由当前 first-tangent v2 和后续 fresh gate 重新验证，不能从旧命名反推。

### 5.5 稀疏相邻边不能自动消除相关漂移

把 history edge cadence 固定为 stride 8：

- sparse edge chain 236/236；
- 2.5 m route-readout within 30° 从 90/105 变为 92/105；
- median 从 11.64° 变为 5.55°；
- endpoint error 仍是 1.557 m，而 dense 为 1.552 m。

因此问题不是简单的“边太多”。相邻视觉误差具有相关性，只减少积分次数不会自动得到全局
一致轨迹。

### 5.6 反向单帧重定位不可观测

在历史 heading ±15° 的受控 query 上，DINO top-1 可在 1 m 内定位 12/12；换成真实回程常见
的 `180°±15°` 视角后：

- top-1 within 1 m：0/12；
- top-8 contains within 1 m：4/12；
- top-1 median position error：5.295 m；
- 还出现 6 次大幅 temporal-order violation。

所以“每一步 DINO 找最近历史帧，再 PnP”不是阈值没调好，而是视角几何下观测不到。最终
方法必须利用 goal switch 时已知的 causal tail 初态和序列连续性，不能依赖独立单帧定位。

### 5.7 纯视觉里程计也不够

在 22.96 m 的反向 route tape 上，只积分 LingBot motion：

- predicted progress 仅 8.60 m；
- final progress error 14.35 m；
- bearing median error 53.25°；
- within 30° 仅 19/72；
- 六帧原地转向产生 2.27 m 伪平移。

因此不能只靠 odometry 积分；需要“连续动态模型 + 路线结构/序列观测”，但不能退回独立
单帧硬匹配。

### 5.8 Action-coordinate 只解决了 route clock 的一部分

Action-coordinate 用执行器 translation/yaw receipt 代替帧号作为 route coordinate。它在
一个冻结离线 gate 上达到 99/99 address within 1 m、99/99 bearing within 30°，说明
“帧密度不是距离”这一修正是对的。

但旧闭环到 46 条时 endpoint/action 均为 7/46，配对 `+3/-3`，平均 final distance
`7.69 -> 5.91 m`；完整 N=48 尚未形成合法 summary。更关键的是，旧 translation receipt
包含大量 commanded-not-realized motion，导致 route coordinate 会在机器人原地时继续前进。
因此不能从离线 99/99 直接宣称该方案闭环有效。

### 5.9 Frozen NavDP 下游仍是未排除变量

已消费 index 32 的旧四臂都失败：action-coordinate mixed、oracle historical-route mixed、
oracle current-geodesic mixed、oracle current-geodesic pure PointGoal 均为 0/1。Oracle
geodesic 两臂把 33.79 m 降到约 25.2 m 后停滞，且 critic `<-0.5` 分别占约 77.7% 与 88.9%。

这条 N=1 不能证明 NavDP 无能力，因为旧 oracle direction 是 2.5 m path chord，并且成功与
运动测量均有上述混杂。它只说明：**“给一个周期性 geodesic chord”仍不足以解决这条多层
长程任务。**

代码层还有一个明确的表示边界：`NavDP/baselines/navdp/policy_agent.py::process_pointgoal`
把 PointGoal 的 forward 分量裁到 `[0,10]`。对 signed heading：

- `|heading| <= 90°`：方向完整可表示；
- `|heading| > 90°`：负 forward 被抹掉，输入被投影成侧向分量；
- 接近 180°：处理后 PointGoal 范数趋近 0。

这与已测 point-token 在 165°--195° 零输出一致。因此“oracle direction 正确”不等于它已被
完整送入 frozen decoder。v2 的预冻结分析会同时报告 `>90°` 与 `>=165°` 的请求比例及裁剪后
PointGoal 范数；若 tangent 频繁进入后方不可表达区，失败必须归因到 controller interface，
不能错记成 route geometry 失败。

Critic 的含义也必须准确：`predict_critic()` 只读取 candidate trajectory 与当前 RGB-D
embedding，不读取 ImageGoal 或 PointGoal embedding。它主要评价局部可行性/碰撞风险，不是
“这条轨迹是否朝向目标”的价值函数。当所有 16 个候选的最大 critic 低于冻结阈值 `-0.5`
时，agent 会把选中轨迹的 forward 分量清零，并把 lateral 分量改成其均值符号。经 pure-pursuit
执行后，这表现为侧向搜索/原地转向；不同 replan 的符号变化会产生左右摆动。因此：

- 低 critic 不能解释成“CEC proof 低置信度”；
- 不应调 `-0.5` 来修长程目标语义；
- 若 first tangent 减少低-critic 比例，说明它把生成分布推回局部可行走方向；
- 若方向正确且可表达但 critic 仍低，才说明 frozen decoder 对该条件组合的 trajectory support
  不足。

## 6. 关键几何错误：2.5 m lookahead 不是局部切向

对一条折线路径 `gamma(s)`，旧 readout 使用：

```text
q = gamma(s + 2.5 m)
b_chord = unit(q - gamma(s))
```

这是一条跨过未来 2.5 m 路径的弦。真正需要的局部方向是：

```text
b_tangent = unit(gamma(s + epsilon) - gamma(s)), epsilon ≈ 0.3 m
```

当 2.5 m 内含拐角时，两者可以完全不同。在 index 32 的 step 320：

- first traversable tangent heading：约 `+100.1°`；
- 2.5 m chord heading：约 `-3.6°`；
- mismatch：约 `103.7°`。

296 次计划的 mismatch 中位数约 `90.6°`。因此之前所谓的“oracle geodesic bearing”并非
oracle local action direction；它仍丢失了拐角前的局部可行性。

这也解释了为什么旧 N=40 Novel oracle-bearing 结果与本次不矛盾：旧探针使用首个至少
0.3 m 的 path waypoint/周期 yaw 纠正，接近局部切向；它不是持续注入 2.5 m chord。

## 7. 已完成的冻结三臂判别实验

### 7.1 身份

- consumed history：manifest index 32，`5jp3fCRSRjc/episode_table3_survey_080`；
- start geodesic：33.791 m；起点 y `3.209 m`，目标 y `0.009 m`；
- protocol：`LONG_RANGE_LOCAL_TANGENT_ATTRIBUTION_PROTOCOL_20260903.md`；
- protocol SHA-256：`3e48de6deeff53da7ed4e7732efbee4ecf3e02691480e35021ea2e29abc4a3ba`；
- source bundle：`hm3d_longrange_local_tangent_00bee7fb60cef334`；
- bundle receipt SHA-256：
  `00bee7fb60cef33454c1b4b65b41a47e129b0a74bf2cd24b9e6abddd0805a8b3`；
- Slurm：`16822227_32`；
- 预揭封 analysis receipt：`longrange_local_tangent_analysis_freeze_20260903.json`，SHA-256
  `015f47fef8b0d85cd7564303b809d56b7dfff858e54d66947bf1d3b680c3b8a4`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_local_tangent_20260903/consumed_00bee7fb60cef334`。

所有 navigation outcome 必须等三臂全部完成后一起读取。

### 7.2 唯一处理差异

三臂均先执行完全相同的真实 CEC proposal/certificate，并保持同一 ImageGoal、RGB/depth
transaction、FIFO、diffusion seed、8-action horizon、2.5 m PointGoal norm 与预算：

1. `oracle_chord_mixed_realized`：当前 Habitat path 的 2.5 m chord，mixed NavDP；
2. `oracle_tangent_mixed_realized`：第一个 x/z baseline `>=0.30 m` 的 ordered path tangent，
   每步 mixed NavDP；
3. `oracle_tangent_then_native_realized`：同一 tangent；heading residual `>20°` 时 mixed，
   已对齐时用同 observation/seed/FIFO 的 read-only native ImageGoal resample。

第三臂只是定位下游冲突的机制诊断；20° 是冻结诊断阈值，不是候选论文方法。它没有依据
失败、距离或 role 做 fallback。

### 7.3 修正后的测量

- `path_len` = post-snap 实际 x/z 位移；
- 逐帧 motion receipt 必须与 pose diff 一致；
- success = 与冻结 goal-floor position 的 3-D Euclidean distance `<1 m`；
- 同时记录 planar final distance、vertical error、zero-motion count；
- runtime/server error 中止，不计作导航失败；
- RGB buffer 写入 `$SLURM_TMPDIR`，避免 scratch file-count quota 再次污染实验。

### 7.4 结果分叉

| 观察 | 因果解释 | 下一步 |
|---|---|---|
| continuous tangent 明显优于 chord | 主要缺失量是局部路线切向 | 构造可部署 local route-tangent readout |
| selective tangent 优于 continuous tangent | tangent 应纠正 heading；持续 PointGoal conditioning 与 ImageGoal 局部控制冲突 | 不再调路线半径；改 controller 接口/候选选择 |
| continuous ≈ selective 且二者都改善 | route tangent 可持续作为唯一 residual | 直接进入单目实现与 fresh replication |
| 两个 tangent arm 都弱 | 正确几何方向仍无法由当前 frozen mixed decoder 兑现 | 停止 route-geometry 微调，审计 NavDP conditioning/critic/action distribution |
| planar 接近但 vertical 不降 | 核心是跨层能力，不是水平路线 | same-floor 与 multi-floor 分开；不再叫纯长度问题 |

### 7.5 解封结果与冻结决策

Slurm `16822227_32` 于 `00:37:12` 正常完成；completion sidecar 校验通过。事前冻结的
analyzer 一次性复算得到：

| arm | success | final 3-D / vertical | min geodesic | realized path | critic < -0.5 |
|---|---:|---:|---:|---:|---:|
| 2.5 m chord + mixed | 0 | 12.773 / 3.200 m | 25.157 m | 25.720 m | 78.04% |
| first tangent + mixed | 1 | 0.990 / 0.000 m | 1.166 m | 33.089 m | 58.91% |
| first tangent then native | 1 | 0.977 / 0.000 m | 1.164 m | 35.142 m | 59.15% |

三臂都保持真实 CEC proof；连续 tangent 把 minimum geodesic 相对 chord 降低 `23.991 m`，
并真实跨过 `3.200 m` 高差。selective 与 continuous 的 final distance 只差 `0.013 m`，
critic 低分比例只差 `0.24 pp`。因此按 7.4 的事前分叉选择第三种解释：**route tangent
可持续作为同一个 mixed NavDP 的唯一 residual；没有证据支持 20° 门控或切回 native。**

这仍只有一条已消费失败且 tangent 来自 Habitat shortest path，所以它只授权实现，不能成为
论文 SR。完整结果见 `LONG_RANGE_LOCAL_TANGENT_ATTRIBUTION_RESULT_20260903.md`；本地分析
SHA-256 为 `cf5c47944fca307df99008d6a8584335a7005f8a6c656e1f6f608a206a87e9fd`。

## 8. Tangent 机制通过后的最小可部署解

目标不是建全局 metric map，也不是加入 graph planner。推荐的方法对象是
**proof-carrying local route-tangent compass**：

```text
CEC target certificate
  -> freeze target anchor and authorized causal interval
  -> represent history as ordered adjacent local SE(2) edges
  -> initialize query coordinate at the known causal tail
  -> update a monotone route-coordinate belief from adjacent query motion
  -> read only the first feasible local route tangent
  -> discard route length and global endpoint pose
  -> fixed 2.5 m unit-direction token to the same frozen NavDP
```

### 8.1 必须保留的设计原则

1. **一个目标 proof。** DINO/LightGlue/PnP 只在 goal session 开始时认证目标 anchor；不在
   反向回程每步重新寻找单帧地址。
2. **局部状态，不维护脆弱全局终点。** 历史只存相邻局部 motion 与 temporal order；消费
   route derivative，而不是长链积分后的绝对 endpoint。
3. **从已知 tail 启动。** Goal switch 时当前状态与 causal history 尾部连续，这是系统真正
   已知的初始条件。
4. **单调进度。** 进度由实际实现的局部平移推进，不能由 frame count 或 commanded velocity
   推进；原地转向的 route progress 为零。
5. **无距离分支。** 同一个算子覆盖短程与长程；短 route 自然靠近 endpoint，长 route 输出
   下一段 tangent。
6. **无失败触发 rescue。** Certificate 接受后不根据 stuck、距离档或结果切回 endpoint/native；
   geometry transaction 损坏则明确 stop。Certificate 初始 reject 的 exact native 行为仍是
   canonical open-set 合约，不属于长程方法内部 fallback。
7. **高度先验只校准局部 arc。** Camera height 可以给相邻 motion 与 dense depth 定尺度；
   不把长基线 PnP norm 当作已认证距离。
8. **PointGoal 仍是局部语义 token。** 固定 2.5 m 只是 frozen NavDP 的条件输入尺度，不代表
   一次直线执行 2.5 m。NavDP 每次输出短轨迹、执行 8 steps 后重规划。

### 8.2 若 selective arm 才有效

这将否定“持续 mixed PointGoal 是合适接口”，但不否定 route tangent。最终实现不应照搬
20° hard switch。更干净的两个候选只允许做一次机制门后再选择：

- **native trajectory re-ranking**：让 frozen ImageGoal NavDP 生成原生候选，用 route tangent
  只排序候选，不再把 PointGoal 持续注入 decoder；
- **continuous residual authority**：将 tangent 作为连续 heading residual，而非另一目标；其
  权重由几何量连续决定，不使用 distance/stuck/role gate。

前者最保守，因为不改变原生 trajectory support；后者更统一，但需要新增 adapter 或训练，
不能在没有闭环证据时声称更优。

## 9. Fresh 证明必须怎样构造

当前 48-history population 已被多轮诊断消费。任何新方法只能在新的 immutable population 上
确认。建议两阶段：

### 9.1 Deployable mechanism confirmation

这一层已经由实现契约与 result-blind construction audit 完成，而不是另挑 6--8 条看起来
成功的 episode：相邻 RGB motion、frame-40 尺度边界、route-coordinate 单调性、fixed-radius
direction payload 与 geometry-stop 语义均有纯测试；真正的导航效果只允许由 9.2 的完整 fresh
population 决定。这样避免再次把小样本机制录像误报为 SR 证据。

### 9.2 Paired closed-loop confirmation

- 现有未消费 20--30 m same-floor capacity 为 23 histories / 8 scene clusters；全部纳入主分析，
  并另报每 scene 最多 2 条的 15-history sensitivity；
- scene multiplicity 为 `1/3/2/2/2/2/6/5`，因此主分析必须使用 scene-cluster bootstrap，且不能
  只凭 23 条 episode-level 比例下结论；max-two sensitivity 正是为了检查两个高频 scene 是否
  驱动效应；
- 结果盲 population 描述为：geodesic `20.37--24.66 m`（中位 `23.62 m`），causal history
  `963--1313` 帧（中位 `1182`），Revisit max-covis `0.556--0.968`（中位 `0.607`），目标高差
  中位 `0 m`、最大 `0.40 m`；它明显比 Final14/Fresh160 的高共视近距离总体更难；
- 本轮只能使用未消费 controlled causal-RGB survey；因此即使通过也只确认路线 readout，不能
  冒充 actual mono Goal-A 的最终端到端结果；
- 处理维度显式拆开：`geodesic span × same/multi-floor × route curvature`；
- 两个主臂：canonical endpoint-bearing CEC 与 frozen route-tangent CEC；
- native 只作上下文，不参与方法选择；
- 同机同进程、同 prefix、proof、seed、FIFO、depth receipt、budget；
- success 使用与冻结 goal-floor position 的 3-D Euclidean `<1 m`，并同时报告 planar 与
  vertical final distance；geodesic 仅作为路径难度/进展诊断，不作为终点跨楼层判定；
- 路径长度只累计 post-snap realized motion；
- 次要指标：SPL、minimum geodesic、route progress、critic 分布、zero-motion fraction、turning。

必须预注册一个停止规则：若 route-tangent 没有正向 paired net 或出现明显损失，不再进行
lookahead、radius、threshold 网格搜索；将 canonical CEC 的论文 claim 限定在已确认的 2--9 m
Revisit 范围，把长程作为 limitation。

## 10. 当前实现与验证状态

新增/修改的可部署核心：

- `longrange_local_tangent_attribution.py`：纯 tangent、heading、realized motion、3-D distance
  合约；
- `eval_hm3d_longrange_local_tangent_attribution.py`：真实 CEC 后的 read-only privileged
  resample；
- `run_hm3d_longrange_local_tangent_attribution.py`：三臂配对、proof/depth/FIFO/输入一致性和
  独立运动复算；
- `eval_2leg_habitat.py`：仅在显式传入 `success_goal_position` 时启用 3-D success，默认旧
  evaluator 行为不变；
- `run_hm3d_fullmono_server_scene.sh`：runtime JPEG buffer 改到 node-local temporary storage；
- `monocular_adjacent_motion.py`：相邻 RGB + frame-bound height-scaled LingBot depth 的局部
  `(forward,left,yaw)` witness；
- `monocular_route_tangent_runtime.py`：从已认证 goal image 到历史 anchor、再到 causal tail 的
  稀疏有序 route tape；
- `path_budgeted_route_compass.py`：累计真实视觉运动预算下的单调 route progress，以及第一条
  baseline `>=0.30 m` 的 forward tangent；
- `NavDP/baselines/memnav/policy_agent.py`：只在初次 CEC proof 后建立一次 route，随后持续读
  tangent；不读取 evaluator pose、Habitat path、executor odometry、sensor metric depth、距离档或
  stuck 状态；
- `hm3d_longrange_route_tangent_experiment.py`、fresh freezer、runner、analyzer 与两个独立
  verifier：绑定三臂、3-D success、proof equality、exact reject fallback 与 post-accept stop。

截至最终 fresh 提交，新增 route-tangent closure 验证为：

- 最新 source bundle 在工作区与隔离目录内各 `45/45` tests；此前完整相关回归为
  `49/49`；
- analyzer + independent verifier 的完整 23-history synthetic closure；
- Habitat evaluator/runner CLI、Python compile，以及远端真实 Habitat 环境中显式禁止
  `cv2` 的 runner import smoke；
- 三个 Slurm template 的 shell/lint 审计；
- remote `sbatch --test-only` 六级依赖链；
- source bundle 逐文件 SHA-256 与只读部署。

### 10.1 Fresh capacity 预审计错误与 fail-closed 修复

第一版协议错误预登记 `64 histories / 14 scenes`。CPU population job `16825614` 在任何 GPU
query 前因 exact count mismatch 退出；依赖链自动取消，`query arms executed = 0`。独立重算发现
错误来自把不相关位置的高度作比较。部署时正确的 same-floor 条件必须比较：

```text
online-A endpoint y  vs.  Revisit query goal y
```

在不读取任何 policy outcome 的情况下，正确容量为：

| stratum | unused same-floor histories | scenes |
|---|---:|---:|
| 0--20 m | 15 | 15 |
| 20--30 m | 23 | 8 |
| 30--50 m | 0 | 0 |

v2 没有放宽 `vertical <= 0.5 m`、没有换距离段、没有改方法；它纳入全部 23 条合法
20--30 m history。两个场景各有 5--6 条，因此预注册一个 `max 2 / scene` 的 15-history
sensitivity，主统计同时使用 scene-cluster bootstrap。

### 10.2 首次进入完整 evaluator 的 fresh 链（已作废）

- v2 protocol：`hm3d_longrange_route_tangent_freeze_protocol_v2_20260903.json`；
- protocol SHA-256：
  `435fe26646a8c42229230213ff70d9fc07c3d7b746b44641251d4eb4ce84227e`；
- immutable bundle：`hm3d_longrange_route_tangent_c472a57c07f261f5`；
- bundle receipt SHA-256：
  `c472a57c07f261f5384c839e5eefae9adc24b15f77b7685f410f92dd226d96fd`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_tangent_20260903/fresh_c472a57c07f261f5`；
- population freeze / verify：`16829676 -> 16829677`；
- technical gate / remaining：`16829678_[0] -> 16829679_[1-22%4]`；
- analysis / independent verify：`16829680 -> 16829681`。

gate 与 remaining 在读取 gate outcome 前一次性提交。这里的 gate 只是完整运行栈技术检查，
不是方法中的距离/失败门控。提交前，新审计器在真实 HPC 文件系统上对上一条零结果失败链
留下的同一确定性 population 做了逐文件结果盲预演：`23 histories / 8 scenes / 仅 20--30 m /
same-floor marker 全部有效 / role 不可见 / query outcomes 未读取`，耗时 `272.53 s`。新的
population 仍由相同冻结协议重新生成并独立复核，不直接复用旧 run root。

该 `c472a57c07f261f5` 链最终没有产生正式结果。technical gate `16829678_0` 用时
`00:22:06`，三个 evaluator 都写出了各自的底层文件，但统一 runner 在写
`completion.json` 之前退出；`16829679--16829681` 因依赖自动取消。由于没有 completion、
aggregate summary 或 independent result verification，这一链不能作为科学结果。修复期间没有
读取任一 arm 的 metric、summary、success、final distance 或 trajectory；只读取了触发异常所需
的 route-plan 结构。具体边界问题及最终新链见 10.5。

### 10.3 GPU 前完成的 early-anchor 双边界审计

第一次生产闭包复核注意到 proposal contract 允许 candidate anchor 从 frame 8 开始，而 route
最初把 frame 39/40 误当成统一下界。为避免潜在的“certificate 已接受、route 随后因早期
anchor 停止”，`16826238/16826240` 在仍为 `PENDING` 时被保守取消；四个 GPU/summary job 均为
`00:00:00`，没有 query arm 执行。随后曾实现一个额外的 first-prefix depth forward 和 reverse
bridge；继续追踪完整调用链后，确认这两个部件都不需要。

真实合约有两个不同的时间边界：

1. `self.S = LingBot num_scale = 8`。DINO shortlist 和 canonical CEC depth replay 都允许
   `anchor >= 8`；`_certified_reference_depth_impl()` 的实际合法区间是 `[S, n-1]`；
2. frame 40 是单独的 camera-height metric-scale receipt 冻结时刻，只约束 query-time metric
   depth 何时可用，不是历史 anchor 的下界。

现有固定 stride writer 在八帧初始化后自然写入 frames `8,16,24,32,...` 的相对深度；不需要
在 scale block 上另跑一次 prefix depth。若目标 anchor 是 9 这类非 stride frame，初始 CEC 已经
通过 canonical exact replay 得到该 anchor 的 depth，route 第一条边直接使用这个同一 depth 做
`anchor -> next_stride_frame` 的 PnP。到 query 开始时，frame-40 receipt 已冻结，再统一把这些
历史 relative depth metricize。因此既不需要未来 query observation，也不需要
`successor -> anchor` 的求逆 bridge。

最终闭包因此是：route 对所有已授权的 `target_anchor >=8` 提供 total readout；selected target
depth 来自与 endpoint arm 相同的 canonical CEC replay，中间节点来自原有 sparse causal writer，
尺度来自一次冻结的 frame-40 receipt。运行审计强制 `target_anchor >=8` 且
`pre_metric_anchor_bridge=false`。这不是结果后的阈值或方法变更，而是在任何 GPU query/结果
产生前消除错误的边界假设，同时保持 endpoint 与 tangent 完全相同的初始 proof。

第二、第三个中间 bundle 的 GPU/summary 任务
`16827507/16827509/16827510/16827511` 与
`16828843/16828844/16828845/16828846` 也都在 `PENDING`、`0:00` 时取消。CPU population
freeze/verification 可以保留作 immutable construction receipt，但最终闭环必须由上述双边界
修正后的新 content-addressed bundle 运行。

其后的 `f16d230d74df2aeb` bundle 完成了最终科学闭包，但 technical gate `16828945_0` 在任何
episode evaluator 启动前退出：轻量 arm-audit 模块为读取 `CERTIFIED_MINIMUM_ANCHOR=8`，错误地
导入了需要 OpenCV 的完整 certificate runtime；远端 Habitat Python 没有 `cv2`。该 gate 仅完成
server startup，没有 `evaluation/`、`completion.json` 或 query outcome；remaining/analysis/
verifier 被依赖自动取消。修复没有安装或替换环境依赖，而是新增 dependency-free
`certified_relocalization_contract.py`，让 runtime 与 evaluator 共享同一常量源。新 bundle 在
远端相同 Singularity + Habitat Python 中通过 `runner import` 且确认 `cv2_loaded=false` 后才
继续。

随后 `c47bec283de0db94` bundle 的 population freeze/verify `16829093/16829094` 成功，technical
gate `16829095_0` 又在任何 arm 完成前 fail closed。原因是通用 Table-III 审计器默认要求原始
`0--20 / 20--30 / 30--50 m` 三个桶同时达到 power gate，而该正式协议有意只冻结全部合法的
same-floor `20--30 m` stratum。目录中只有 episode contract 与首臂日志，没有 metric、plan、
completion 或 outcome；因此这仍是零结果基础设施失败。修复保留通用三桶默认行为，只为
`table3_longrange_route_tangent` 增加严格的 exact-selected audit：恰好 `23 histories / 8 scenes /
20--30 m`，并逐条要求 fresh marker 与 vertical error `<=0.5 m`。补丁经过本地双层测试和上述
真实远端 population 预演后，才生成 `c472a57c07f261f5` 并提交 10.2 的链。科学 population、
三臂、阈值、半径、budget 与统计停止规则均未改变。

任何科学代码变化都必须生成新 content-addressed bundle；当前运行 bundle 不允许就地修改。

### 10.4 Scratch file-count quota 的可审计清理

一次正式提交在创建 run root 时触及个人 scratch file-count quota。清理前先验证一项已封存的
Final14 mono factorial run：21/21 task complete、POSTHOC receipt 和 independent verifier 均通过；
随后只删除可由 receipt 重建的 task-local RGB `buffer/`，保留所有 summary、metric、plans、
日志、receipt 与 verifier。共删除 `106,810` 个 JPEG/目录项、`5,411,961,295` bytes；不读取
navigation outcome，也不改变任何科学结果。远端清理 receipt：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/maintenance/
  final14_mono_factorial_runtime_buffer_cleanup_20260903.json
SHA-256: 9bdcc7a9b0c6c6999c0b791b32b43ff23942ca326f9f7ebb41afb02af91192d2
```

新的 route-tangent 运行把逐帧 JPEG 全部放在 `$SLURM_TMPDIR`；scratch 只保留小型审计结果，
因此该 quota 问题不再随 23-history eval 线性累积。

另一次只读 diagnostic bundle smoke 在 login node 的共享 `/tmp` 遇到 `No space left on device`；
随即改在个人 scratch 的 `maintenance/` 临时目录按包结构复测并在退出时清除，测试通过。正式
GPU evaluator 使用各 compute node 的 `$SLURM_TMPDIR`，因此该 login-node 临时盘事件没有影响
任何 formal arm 或结果文件。

### 10.5 Atomic geometry-stop 审计修复与最终 fresh 链

`c472a57c07f261f5` 的唯一异常为：

```text
ABORT: route tangent escaped the canonical CEC depth boundary
```

结果盲结构审计显示，route arm 共留下 75 个已授权 plan：目标 anchor 始终为 111；前 74 个
是正常 active readout，且都显式记录 `pre_metric_anchor_bridge=false`；最后一个是预注册的
atomic `geometry_failure` stop。Stop packet 按合约不会伪造一个成功 route readout，因此省略了
只属于 active packet 的 `pre_metric_anchor_bridge` 字段。旧 runner 在先识别 stop 之前就要求该
字段必须为 `false`，错误地把一个应计为导航失败的 method outcome 升格成基础设施异常。

修复只改变审计顺序，不改变方法或 outcome：

1. 所有 accepted packet 仍必须满足 canonical `target_anchor >= 8`；
2. active packet 仍必须显式满足 `pre_metric_anchor_bridge=false`；
3. stop packet 只允许该字段缺失或为 `false`，同时必须证明 controller 未被调用、native 和
   endpoint fallback 均未执行、状态为 `geometry_failure`；
4. 该 stop 在 metric 中仍必须记为 `reached=0` 和 `geometry_stream_failure`。

新审计器在不读取旧 arm outcome 的条件下重放完全相同的 plan，得到：

```text
certificate_accept_plans=75
active_route_plans=74
geometry_stream_stop_plans=1
history_route_receipt_hashes=1
canonical_anchor_boundary_verified=true
```

最终正式链使用全新、只读、content-addressed bundle，从零重跑三臂，不复用 `c472` 的任何 arm
文件：

- source bundle：`hm3d_longrange_route_tangent_751efdcbb436c99b`；
- source receipt SHA-256：
  `751efdcbb436c99b5cc92d69633e812e9d1edc381dcf224af9042113ee8e874e`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_longrange_route_tangent_20260903/fresh_751efdcbb436c99b`；
- population freeze / independent verify：`16830966 -> 16830967`（均已完成）；
- technical gate / remaining：`16830968_[0] -> 16830969_[1-22%4]`；
- aggregate analysis / independent verify：`16830970 -> 16830971`；
- 预冻结 failure audit：最终为 `16834088`，严格依赖 `afterok:16830971`。先前的
  `16831043` 在 `PENDING / 00:00:00` 时取消并替换；它没有执行或读取结果。

最终 failure panel 在新 run 尚未创建、任何新 outcome 尚不存在时冻结。V3 freeze SHA-256 为
`938f5060d837c375ea49181f1512a9b2cf1c791f836923282dfe01de46fe13d3`。在 formal gate 已提交但
仍未读取任何 outcome 时，审计实现又补齐冻结 panel 中原已要求、但旧脚本漏输出的
selected-anchor 分布、initial reject 数和逐 episode final/max route progress；panel、分区规则和
科学方法均未改变。最终 diagnostic source receipt SHA-256 为
`00294ff9b817fb1c601721f97f9689cae2864f71048549df23ab7d46a8d807a4`。它固定检查 final/minimum
3-D distance、真实运动复算、critic、blocked/stationary motion、certificate accept/reject、
geometry stop、route progress/cross-track/live-PnP，以及 endpoint 与 tangent 的全部 discordant
pairs；只在 formal verifier 通过后运行。

在正式 array 仍运行、aggregate outcome 尚未读取时，又发现上述 analyzer 把“minimum 3-D
distance”实现为每 8 步一次的 planning sample；正式 evaluator 虽然逐动作判断 success，但该
诊断可能漏掉两次规划之间的最近物理位置。固定 panel 与正式判据均不变，只把失败审计补成：
从完整 query pose trace 加 terminal pose 逐动作复算 minimum 3-D/planar distance，并核对
actionwise terminal 3-D distance 与正式终值一致；旧 plan-time sample 另名保留。V5 diagnostic
source receipt SHA-256 为
`9e2cda80afabc6d5c414216e0c3d3b3b9e9849b6376931dd163aca003d13afef`。旧作业
`16831766` 在 `PENDING / 00:00:00` 时取消，替换为 `16833895`；远端临时包结构复测为
`1 passed`，没有读取任何 policy outcome。

同一结果盲窗口内又补入一个不参与方法选择的 operational panel：逐臂总 wall time、首次与
后续 relocalization、CEC total decision、NavDP controller 和 monocular depth sidecar latency。
原因是路线 tangent 在 goal switch 时需要物化整条 sparse edge chain；即使 SR 成立，也必须
诚实报告其一次性计算代价。V6 receipt SHA-256 为
`3fbcd4b62fda14af2a42130d523390217d509eda698bf089e69ef366f892376b`；远端包结构复测仍为
`1 passed`。随后按第 9.2 节早已列出的次要指标，再补齐 SPL、trace 复算的 3-D 路程与累计
绝对转角；这些只读量不进入确认规则。最终 V7 receipt SHA-256 为
`92b5c38f6e268248275a019beb5feb6c5fc95608d6d33dd9d6ed584ebf1fa9e0`，并明确冻结“此后不再
添加结果依赖字段”。`16833895`、`16833929` 及其下游视频作业都在
`PENDING / 00:00:00` 时取消；最终诊断作业为 `16834088`。

GPU gate `16830968_0` 随后在 A100 上以 `00:22:29 / exit 0` 完成；唯一 completion sidecar
校验通过，aggregate 仍未揭封。它证明先前的 stop-packet 审计错误已关闭，并且完整三臂 runtime
能写出统一 completion。剩余 array 已解除依赖，提交时因项目级 `QOSGrpGRES` 等待额度释放，
Slurm 估计 `2026-09-03 04:41 EDT` 启动；这不是代码或依赖错误。Slurm 已用 `--test-only` 接受
全部六级提交形状，population 也已由冻结协议重新生成并独立验证为 `23 histories / 8 scene
clusters / same-floor 20--30 m`。

在等待 formal verifier 期间又完成一次不读 outcome 的执行口径复核：

1. `pursuit_step()` 返回的路程是 navmesh `snap_point` 之后的实际 x-z 位移，碰撞或 snap
   拒绝时记为零，不是未执行的命令位移；
2. `table3_longrange_route_tangent` 显式向 `run_policy_leg()` 传入完整的 goal xyz，
   每个实际动作后以三维欧氏距离严格 `<1 m` 判定，不再使用旧的平面 x-z 判定；
3. rollout trace 保存每个动作前的真实 pose，failure analyzer 再并入 formal 终点 pose，
   因此逐动作 minimum 3-D distance 不会漏掉最后一步；
4. route-tangent 相关合同、population、runner、result verifier、failure analyzer 与视频渲染共
   `81 passed`；V7 failure-audit receipt、V2 visualization receipt 及两份方法 protocol 的
   SHA sidecar 全部复验为 `OK`。

这些证据只确认评测与审计实现没有退回旧口径，不预告也不改变最终 SR 结论。

## 11. 可视化失败证据

对照视频：

```text
.diagnostics/longrange_failure_video_20260903/
  longrange_failure_comparison.mp4
```

视频用于观察路线、heading、critic 与实际运动，不用于统计。关键帧与渲染脚本位于同目录。

Fresh 结果的可视化选择也在解封前冻结：不挑“好看案例”，而是按 manifest 顺序渲染
endpoint 与 route-tangent 的**全部**配对不一致样本。每个视频使用相同布局，展示 native / endpoint /
tangent 的真实 x-z rollout、逐动作 3-D goal distance、route progress、请求 heading 与 critic；
只重放 sealed pose/plan receipt，不重新运行策略。最终 renderer V2 source receipt SHA-256 为
`acea4d89ff2f6846e3ed385e9efd928ed06380060f33aca774fa7be959668e91`，代码相对首个 receipt
未改变，本机与远端包结构 smoke 均为 `1 passed`。作业 `16834097` 依赖最终 failure audit
`16834088`，输出到正式 run root 的
`posthoc_discordant_videos/`；即使没有 discordant，也会写出带 SHA 的零条目 manifest。

## 12. 今晚的完成定义

今晚不是以“再得到一个更高 SR”作为完成条件，而是以下顺序：

1. 三臂任务完整结束，completion SHA、输入/proof/depth/FIFO/实际运动审计全部通过；
2. 只在三臂齐全后读取 outcome、3-D/planar/vertical final distance、minimum progress、critic 与
   zero-motion；
3. 按第 7.4 节的冻结分叉确定根因层，不做半径/阈值 sweep；
4. 只有 tangent 机制成立，才实现 deployable local route-tangent；
5. 冻结并独立复算全部合法 fresh population，再运行 paired closed-loop；
6. fresh verifier 通过后，才决定它进入论文方法、ablation，还是作为明确 limitation。

在第 6 步之前，最诚实的系统结论仍是：CEC 已可靠解决高支持的短中程 Revisit；长程压力
测试揭示了 route topology 与跨层控制的未解决边界，而不是 certificate 本身失效。

## 13. Fresh 结果后的 motion-chain 归因与停止结论

### 13.1 正式 fresh 结果

最终 `23 histories / 8 scene clusters / same-floor 20--30 m` 的三臂结果已由独立 verifier
复算：native、endpoint-bearing CEC、route-tangent CEC 分别为 `0/23、0/23、4/23`。Tangent
相对 endpoint 为 `+4/-0`，scene-cluster risk-difference CI 为 `[+5.56,+31.82] pp`，但 exact
McNemar `p=0.125`。它证明局部 route derivative 确实比单一 endpoint direction 携带更多
控制信息；它没有把当前实现提升为已确认的长程方法。14/23 条 tangent 运行发生 typed
geometry-stream stop，其中 13 条发生在 live query、1 条发生在历史路线初始化。

正式结果和 verifier 见：

```text
MemNavData/HM3D_LONGRANGE_ROUTE_TANGENT_FORMAL_RESULT_20260903.md
/scratch/yz11502/Research/Nav-axis-uturn-results/
  hm3d_longrange_route_tangent_20260903/fresh_751efdcbb436c99b
```

### 13.2 “删掉 epipolar prefilter”已被排除

对全部 14 条 exact failed edge 的 same-edge no-authority shadow 显示：直接 depth-PnP 仅
`1/14` 有效且准确，其余 13 条不能形成可靠 pose。一个 H100 exact retry 中，未过滤 PnP
甚至产生 `155.6 deg` yaw error。Aggregate 的决定为
`simple_epipolar_explanation_not_established`，独立 verifier 为 `verified=true`。因此
Fundamental-MAGSAC 不是可以简单删除的冗余工程门；它正在阻止错误局部位姿进入累计路线。

### 13.3 Dense live-query PnP 的冻结 gate 未通过

下一项冻结 consumed gate 保持初始 CEC proof、历史 stride-8 route、历史
Fundamental-MAGSAC→PnP、固定 2.5 m payload 和 NavDP 全部不变；只把 live query 的每个
planning interval 分解成逐动作 direct-PnP。选择是 formal parent 中前三条真正的 live-query
failure：indices `0,1,6`。正式 DAG `16842471 -> 16842472 -> 16842473` 完成并独立复算：

```text
geometry-stream failures          1/3   (frozen pass rule: 0/3)
descriptive navigation successes  0/3
decision                          do_not_relax_thresholds_inspect_failed_atomic_transition
```

Indices 0 和 1 的 live chain 分别完整消费 `1,096` 与 `896` 条连续 motion edges，且都把
geometry stop 降到零，但最终仍 stuck 在 `11.42 m` 与 `4.44 m`。Index 0 的估计 route
coordinate 在真实目标仍远 `6.63 m` 时提前达到 100%，随后 cross-track 从 `3.15 m` 增长到
`7.04 m` 并沿 terminal tangent 离开目标。这不是“再提高采样频率”能修复的 failure。

Index 6 在 H100 上于历史 route initialization 提前停止；其 formal parent 在 A100 上能初始化
并于 live query 才停止。该跨 GPU 数值差异保留在正式失败结果中。A100-only job
`16848570_[6]` 仅作为 consumed sensitivity，不能替换正式 aggregate，也不能生成 SR claim。

完整状态见：

```text
MemNavData/HM3D_LONGRANGE_DENSE_QUERY_GATE_STATUS_20260903.md
```

### 13.4 与旧实验去重后的根因边界

下面这些方向已经有直接反证，不能作为“新方案”重跑：

| 候选修复 | 已有证据 | 结论 |
|---|---|---|
| hard DINO route address | reverse-view top-1 `0/12`、top-8 `4/12` within 1 m | 180° 外观反转时不可观测 |
| soft DINO sequence filter | 两条 consumed route 通过，prospective route final address error `5.16 m` | 帧密度与路程混淆仍会累积 |
| pure LingBot route odometry | 22.96 m return 只推进 8.60 m；bearing median error 53.25° | 无视觉校正不足 |
| action-coordinate clock | 离线 `99/99` address/bearing gate；闭环 partial `7/46 vs 7/46`, `+3/-3` | 正确时钟不等于闭环沿路 |
| delete Fundamental-MAGSAC | direct PnP reliable `1/14` | 会把错误 pose 放进长链 |
| denser adjacent PnP | formal gate `1/3` stop，且两个无 stop 样本仍 `0/2` success | 消除缺测但不消除漂移/控制偏离 |

因此现有 evidence-backed 结论是：**当前 forward-only causal monocular stream 同时缺少稳定的
反向 route observation 和无漂移的长程 motion reference。** CEC 能证明“目标在历史中”并给出
初始 target anchor；它不能凭同一 proof 自动提供整段回程中的连续 route coordinate。当前
长程 failure 不是 certificate threshold、DINO top-K、2.5 m radius 或 PnP cadence 的单变量
问题。

### 13.5 合法的下一步必须增加哪一种信息

若继续做长程方法，只剩三类结构上不同的选择：

1. **双向可观测的 episodic memory**：在首次 traversal 时采集或由局部 3-D history 合成
   return-facing route references；这改变 memory-write contract，但保留纯视觉部署。
2. **可部署的 proprioceptive/VIO route state**：用真实轮式/机身 odometry 或成熟 VIO 约束
   live pose，再用视觉证据校正；这不再是严格的 one-RGB-stream claim。
3. **真正的 visual-route follower**：让 controller 消费一列 receding historical visual
   subgoals，而不是要求 frozen NavDP 用一个最终 ImageGoal 加 bearing token 隐式执行整条
   route；这改变 controller interface，并接近 topological navigation。

三者都属于新的信息或任务合约，不是今晚可以通过调阈值“修复”的同一方法。若论文保持
“one causal RGB stream, one frozen ImageGoal policy”的最小主张，最严谨的决定是把 canonical
CEC claim 限定在已确认的短中程范围，把 20--30 m 结果作为压力测试和 limitation，而不是
继续在同一 consumed population 上追逐 SR。Table-3 本来也属于会议清单的 low-priority 项；
它不能挤占 Table-1 controller/dataset breadth、Table-2 continual legs 和真机 paired evidence。
