# M2P S-1 因果角色分层结果

日期：2026-08-13（Asia/Shanghai）

状态：**10-scene futility-only 门已完成并触发停止；不运行 full40，不授权训练或闭环。**

## 1. 要回答的问题

当前 LingBot/GCT 的完整因果 prefix read-only goal query，能否在不给历史 anchor 的情况下，
替代 DINO-addressed GCT query，为真正的长程 Revisit-C 输出可执行 bearing？

这不是 SR 实验。主指标是离线 bearing error `<=30 deg`；唯一主单位是
`goal_c_t0/factual`。Novel-B 起点、Novel-B midpoint 和 counterfactual goal 仅作因果对照，
不与 Revisit-C 混算。

## 2. 冻结运行与完整性

- Slurm job：`15684078`；H100；完成，exit `0:0`；运行 `17m59s`；
- train-only：10 scenes、20 episodes、120 role/variant sessions；
- development/blind：未读取；
- source receipt：
  `10bc0517a2e865bf95a5235df864383782c3972ada4a10b44658a534622dc0f4`；
- LingBot commit：`7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2`；
- weights SHA256：
  `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`；
- 本地复算确认 `120/120` unique sessions、20 episodes、10 scenes，全部角度有限；
- 每次 goal query 前后 cache identity 均逐 tensor 相同；
- collector report SHA256：
  `d31c62ce58da2086a632b96b9d1cb3417d29a22c01274bc8eeb622ed0d41d4cc`；
- raw rows SHA256：
  `7a3b3c6decfd0a14626903124fe3adb209e39823e865ff73a50210e36a0db306`。

## 3. 主结果：真正 Revisit-C

20 条 factual Revisit-C 全部属于 long-gap：latest positive support 距 decision
`153--876` frames，中位数 `269`。

| `<=30 deg` 四格 | 数量 |
|---|---:|
| both good | 5 |
| DINO-anchored only | **13** |
| full-prefix only | **0** |
| neither | 2 |

因此：

| query | CDF@30 |
|---|---:|
| DINO-anchored GCT | **18/20 = 90%** |
| full-prefix anchor-free GCT | **5/20 = 25%** |
| oracle union | **18/20 = 90%** |

- full-prefix minus anchored：`+0/-13`，风险差 `-65 pp`；
- exact McNemar：`p=0.000244140625`；
- 10-scene cluster bootstrap 95% CI：`[-85,-45] pp`；
- oracle union 相对 anchored 的 headroom：`0/20`，cluster CI `[0,0]`；
- 10 个 scene 中 full-prefix 没有一个 scene 优于 anchored。

这远超预注册停止条件：`full-prefix only=0`、`anchored only>=5`、full-prefix rate
不高于 anchored。故停止 full40。

## 4. 为什么一条 smoke 曾经看起来相反

因果对照复现出完全不同的时间角色：

| factual role | anchored CDF@30 | full-prefix CDF@30 | latest support gap |
|---|---:|---:|---:|
| Novel-B midpoint | 4/20 | **17/20** | median `0` frame |
| Novel-B t0 | 9/20 | 6/20 | 无历史正 support |
| true Revisit-C t0 | **18/20** | 5/20 | median `269` frames |

所以早期 `global error=0.29 deg / DINO-anchor error=168.3 deg` 的成功例属于
Novel-B midpoint 的近期 support，不是长期 Revisit。把三种 causal role 汇成 155 个
positive session 会得到错误的架构结论。

## 5. 机制迹象与边界

在 factual Revisit-C 上：

- anchored raw direction norm 中位数 `1.785`；
- full-prefix raw direction norm 中位数 `0.254`，仅为 anchored 的约 `12.9%`；
- full-prefix 成功样本的 norm 中位数 `0.448`，失败样本为 `0.241`。

相反，近期 Novel-B midpoint 的 full-prefix norm 中位数为 `0.719`，并取得 `17/20`。
这支持以下解释：当前接口能处理近期/工作记忆关系，但在 decision 末端注入一个很久以前的
goal 时，没有完成可靠的长程内容寻址，输出常收缩成短小且方向不稳的 residual。它证明的是
**当前 GCT cache/interface 对长程 candidate-free retrieval 不可观测**，不是证明所有 learned
candidate-free memory 架构在原则上不可能。

## 6. production 复现边界

- exact DINO anchor：`120/120` 相同；
- 所有 role 的新旧 `CDF@30` row-level agreement：`81/120=67.5%`；
- factual Revisit-C：新旧均为 `18/20`，但 row-level agreement 是 `18/20`。

因此不能把跨进程连续角度或逐行 actionability 当作完全可复现。正式对比采用本次同一
H100、同一 streaming process 内的 anchored/full-prefix 配对；production 只作独立 aggregate
参考。这个差异本身也是停止理由之一，但即使完全忽略 production，same-process 的
`+0/-13` 仍独立触发 futility 门。

## 7. 决策

1. 不运行剩余 20 scenes；已有 10-scene 结果的用途只是止损，不能写成效果确认；
2. 不训练当前 full-prefix goal-query adapter；扩大模型容量不能修复已证实的长程寻址缺失；
3. 保留 DINO 作为冻结的 episodic content address，GCT 只在被寻址的历史区域估计 bearing；
4. 不采用简单 anchored/full-prefix agreement gate：两者 disagreement 的正确分支随 causal
   role 改变；
5. 下一步先用现有 train40 production candidates 做 **candidate-set oracle headroom** 审计：
   若 DINO top-1 已接近候选集 oracle，就不训练 reranker；只有存在足够的可恢复错误，才做
   zero-init、scene-OOF 的小型 residual pointer；
6. 任何后续 learned 模块都必须从 raw-DINO direct 严格退化起步，并以当前 certified
   relocalization 为安全/闭环基线，不再以旧 geometry router 为主要比较对象。

## 8. 可复查文件

- 原始 collector：
  `.diagnostics/m2p_s1_dual_context_futility10_hpc_15684078/collection/`；
- 角色分层复算：
  `.diagnostics/m2p_s1_role_stratified_futility10_hpc_15684078_v2/`；
- 分析器：`MemNavData/analyze_m2p_s1_role_stratified.py`；
- 完整协议：`MemNavData/UNIFIED_MEMORY_TO_POINT_PROTOCOL_20260813.md`。
