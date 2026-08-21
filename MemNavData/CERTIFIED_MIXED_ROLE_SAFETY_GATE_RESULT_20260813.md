# Certified mixed-role safety gate：结果与论文含义

日期：2026-08-13（Asia/Shanghai）  
状态：四场景本机因果安全门完成；**不是 SR、specificity 或 blind 确认**。

## 一句话结论

在四条 strict-v4 `initial ImageGoal -> Novel -> Revisit` episode 中，role-free certified
route 对实际执行的 7 条 Novel 腿均 fail closed，并与同进程 native reference 在 2,098 个执行帧
上逐项完全一致；0 次 certificate accept、0 次 adapter takeover、0 次 runtime failure。唯一可评
的 Revisit C 正控完成一次独立 certificate accept、11 次 mixed-controller 接管并成功。

这补上了“自动方法能否在同一自然流中拒绝 Novel、接受 Revisit”的**集成/因果安全证据**，
但 `N=4` 且仅一个 Revisit 正控，不能据此声称 open-set specificity 或方法 SR。

## 1. 冻结设置

数据为已消费的四条 strict-v4 smoke，各一条 episode：

- `e9zR4mvMWw7/episode_0000`
- `gxdoqLR6rwA/episode_0000`
- `dhjEzFoUFzH/episode_0000`
- `gTV8FGcVJC9/episode_0000`

两臂在同一对已加载 server 中运行，场景间平衡 arm 顺序：

1. `known_c_reference`：A/B 为 native NavDP 并写入因果记忆；只有 benchmark 已知的
   Revisit C 使用 direct metric residual。
2. `certified`：A/B/C 均为 `navdp_auto`；certificate accept 才输出 scale-free bearing 和
   fixed 2.5 m residual，reject 当步调用相同 native ImageGoal NavDP。

共同设置：seed `20260830`、deterministic plan seeds、server selector、每腿 600 steps、
exec horizon 8、冻结 MemNav/NavDP/LingBot 权重。未读取 blind。

## 2. 合并结果

| 项目 | 结果 |
|---|---:|
| scenes / episodes | 4 / 4 |
| 实际执行并审计的 Novel 腿 | **7**（A 四条、B 三条） |
| Novel planning requests | **266** |
| 独立 uncached Novel certificate 决策 | **7** |
| 逐帧比较的 Novel rollout / memory rows | **2,098 / 2,098** |
| Novel certificate accepts | **0** |
| Novel adapter takeovers | **0** |
| certificate runtime failures | **0** |
| Revisit C 正控 eligible / activated / success | **1 / 1 / 1** |

这里的 7 而不是 8 很重要：`dhjEzFoUFzH` 的 A 失败后 B 被因果 censor，空 B 槽位没有被
算成一次安全拒绝。旧分片审计最初把槽位数写成 8；合并器已按实际 plan execution 修正为 7，
不改变任何路径或接管结论。

### 2.1 Novel exact-fallback

| scene | 实际腿 | steps | 独立拒绝原因 | rollout / memory exact |
|---|---|---:|---|---|
| `e9zR4mvMWw7` | A | 65 | `no_causal_candidate` | yes / yes |
|  | B | 600 | `precheck_fundamental_inliers` | yes / yes |
| `gxdoqLR6rwA` | A | 113 | `no_causal_candidate` | yes / yes |
|  | B | 600 | `minimum_inliers` | yes / yes |
| `dhjEzFoUFzH` | A | 186 | `no_causal_candidate` | yes / yes |
| `gTV8FGcVJC9` | A | 81 | `no_causal_candidate` | yes / yes |
|  | B | 453 | `precheck_fundamental_inliers` | yes / yes |

每条腿的 success、steps、path length、final distance 也与 reference 数值完全相同。这说明
结果不是“两个 arm 最后碰巧同成同败”，而是 certificate rejection 后的整个物理执行前缀和
因果记忆写入一致。

### 2.2 Revisit positive control

只有 gTV 的 A/B 都成功，因此只有它拥有合法的 C 条件分母：

- 1 次独立 uncached LightGlue/PnP/certificate 计算；
- reason 为 `certificate_accepted`；
- 后续 10 次为同一定位的 cached bearing 更新；
- 11/11 plans 为 `revisit_adapter_takeover=True`；
- certified C：86 steps、2.824 m、final distance 0.970 m、成功；
- known-C reference 同样成功，因此该单例只验证自动执行链，不用于声称性能优势。

## 3. 它解决了什么、没解决什么

### 已解决到 implementation/causal-smoke 强度

1. 同一 automatic route 确实没有读取 causal-role 标签：A/B/C 均走 `navdp_auto`。
2. certificate reject 的“回退 native”不是文字承诺，而是 2,098 帧 exact counterfactual。
3. 系统不是因为永远 abstain 才安全：在同一自然流的 Revisit C 上确实 accept、接管并成功。
4. 三种拒绝边界均被真实触发：无因果候选、fundamental 支持不足、PnP 最小内点不足。

### 仍未解决

1. `N=4`、7 条 Novel 腿和 1 个 Revisit 正控不能估计 false-accept rate、recall 或置信区间。
2. 本实验没有 native/direct/certified 的大样本 mixed-role SR 对比。
3. fresh160 中 certified `112/120` 相对简单 raw-DINO direct `106/120` 只有 `+9/-3`、
   `p=0.146`；“显著优于最强简单 baseline”仍未成立。
4. Revisit 的正式结果仍是 supported known-role population；本实验没有把它升级为开放集结果。
5. 四条均是 consumed/internal smoke，不是 scene-disjoint paper confirmation。

## 4. 对“工程拼接”风险的诚实判断

风险**仍然存在，但比此前小了一层**。

如果论文方法被描述为“DINO + LightGlue + LingBot + NavDP”，审稿人完全可以认为这是组件拼接；
本次 N=4 不能改变这一点。真正可形成方法贡献的对象应是一个统一契约，而不是组件清单：

> **Certified Episodic Compass**：从因果视觉记忆提出一个历史位姿假设，只在单个原子几何
> 假设可自认证时输出 scale-free bearing；否则对冻结 ImageGoal policy 保持 exact identity。

在这个抽象下，各组件只实现四个明确角色：proposal、atomic verification、bearing recovery、
frozen local control。方法性来自：

- 把错误的 Novel/Revisit 语义分类改写成 open-set history support / relocalizability；
- 只认证方向、不声称单目 metric scale；
- residual 接口与 frozen controller 解耦；
- fail-closed 不只是门控，而有可审计的 exact-identity 契约。

目前证据三角为：

| 问题 | 当前最强证据 | 状态 |
|---|---|---|
| certificate 能否认证可执行定位 | HPC v2：8 TP / 0 FP / 1 FN / 15 TN，19 scenes | 支持，样本仍小 |
| 已知 Revisit 时能否提升闭环 | 112/120 vs native 27/120；vs geometry 91/120 | 强 |
| 自动 mixed-role 是否安全且非永远 abstain | 本次 7/7 Novel exact reject；1/1 Revisit accept+success | 仅 smoke |
| 是否显著优于 raw-DINO direct | `+9/-3, p=0.146` | **未建立** |

因此最危险的写法是把前三项拼成“完整 open-set 方法已经证明”；最稳的写法是把前三项分层，
明确第四项仍是论文主缺口。

## 5. 唯一高价值的下一正式实验

下一次昂贵闭环不应再重复 known-Revisit，也不应继续训练 selector。它应在新的 scene-disjoint
自然 mixed-role 流中比较：

1. native ImageGoal；
2. 最强简单 memory baseline（raw-DINO proposal 的无证书/弱认证版本）；
3. frozen Certified Episodic Compass。

同时报告两个轴，而不是只报 joint SR：

- Novel safety：false takeover、相对 native gain/loss、exact fallback；
- Revisit utility：certificate activation recall、conditional SR、相对 raw-direct gain/loss。

这项实验才能回答 certificate 是否带来有意义的 **risk-coverage Pareto improvement**，也是把
“工程拼接”变成“有原则的开放集 residual 方法”的决定性证据。

当前 scene-role inventory 的事实边界是：3-leg 526 episodes 只有 36 个非空 scene clusters；扣除
consumed/train/development/final-reserved 后剩余 16 scenes，恰好全部属于 blind16。没有可继续
读取的非 blind、scene-disjoint pool。因此：

- 可以在 consumed/train scenes 生成更多 strict-v4 仅用于调试和功效估计；
- 不能把它们写成新泛化证据；
- one-shot blind mixed-role confirmation 必须在方法、arm、阈值和分析全部冻结后，由用户显式
  授权，不能自动启动。

## 6. 产物与校验

- 冻结协议：`MemNavData/CERTIFIED_MIXED_ROLE_SAFETY_GATE_PROTOCOL_20260813.md`
- runner：`MemNavData/run_certified_mixed_role_safety_gate_local.sh`
- 独立审计：`MemNavData/audit_certified_mixed_role_safety_gate.py`
- 分片合并器：`MemNavData/combine_certified_mixed_role_safety_gate_audits.py`
- 两个原始分片：
  - `.diagnostics/certified_mixed_role_safety_gate_local_20260813/`
  - `.diagnostics/certified_mixed_role_safety_gate_remaining2_local_20260813/`
- 四场景合并 receipt：
  `.diagnostics/certified_mixed_role_safety_gate_combined4_20260813.json`
- combined receipt SHA256：
  `380a5cfd801dc0796626c7bec026764607ade28c603340aea9f8024f5bf8dffd`
- 修正后 auditor SHA256：
  `9acfa7af64c8d12747bc57150b5fed936785bbf1074fa1b4dc495c3f3a35e1df`
- combiner SHA256：
  `1a0b0f76a6f35de0a8eb50f546d7c6931ec6bec09f674403842fbc7b9712c2c6`
- 新增审计/合并测试 `5/5` 通过；runner 内相关回归 `26/26` 通过。

