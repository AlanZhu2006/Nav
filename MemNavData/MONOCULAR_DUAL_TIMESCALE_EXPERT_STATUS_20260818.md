# 单目双时间尺度专家：2026-08-18 今晚状态总账

更新时间：2026-08-18（Asia/Shanghai）  
科学状态：**Gate A/B 的接口/可优化性已通过；发现并修复 whole-episode scale 泄漏；
causal Gate C 尚未产生结果；尚无闭环 SR 结论。**

## 1. 今晚真正要回答的问题

当前 CEC 已经证明长程 Revisit history 有用，但完整系统仍依赖 NavDP 的 metric-depth
observation input。今晚不再叠加一个 action gate，而是检验一个更基本、更有价值的问题：

> 冻结 LingBot-Map 从单目 RGB 得到的短程几何，能否被一个小型 adapter 翻译成冻结
> NavDP decoder/critic 真正可消费的 observation latent，同时保留 CEC 的长程记忆能力？

若答案为真，系统从“单目 sidecar + RGB-D controller”升级为部署输入完整单目的双时间
尺度组合；若答案为假，CEC v1 的论文口径保持不变，不能靠叙事掩盖。

## 2. 收敛后的完整架构

```text
causal RGB stream
    |
    v
frozen LingBot-Map (single shared stream)
    |
    +-- dense local-control readout ------------------------------+
    |   recent 8 x 6 special/pose tokens                          |
    |   current 37x37 visual patches                              |
    |   current 37x37 frozen depth features                       |
    |   explicit scale-quality token                              |
    |          |                                                   |
    |          v                                                   |
    |   Geometry Token Adapter A_phi (6.02M trainable)            |
    |          |                                                   |
    |          +--> z_mono [128,384] -------------------------+   |
    |                                                         |   |
    +-- sparse episodic-proof readout                          |   |
        causal RGB history -> DINO top-8 address              |   |
        -> local geometry + historical depth + PnP            |   |
        -> atomic certificate                                  |   |
             reject: abstain                                   |   |
             accept: scale-free unit bearing                   |   |
                    |                                          |   |
                    +--> typed goal residual ------------------+---+
                                                               |
                                        frozen NavDP image/point goal encoder
                                        + diffusion decoder + critic
                                                               |
                                                           trajectory
```

这不是两个 controller 的 action-level MoE，而是 **one map, two readouts, one policy**：

- LingBot 是共享的**状态估计专家**；
- CEC 是稀疏触发的**长程记忆证明专家**；
- NavDP 是唯一的**动作生成专家**；
- 两个上游专家只能通过 `[128,384] latent` 或 `unit bearing/abstain` 两种窄接口影响动作。

Dense readout 的输入 token 仍然来自 LingBot 的 causal sparse-map state，因此可能注意到旧
keyframes；“local”指它每步服务局部控制，而不是假装其上游完全无历史。CEC 的区别是显式
检索、几何自认证和 abstention，不是维护第二份地图。

因此系统不会学习一个不透明的“选 LingBot 还是选 NavDP”门，也不需要 Novel/Revisit
role 标签。

部署闭包已有明确复用点：`NavDP/baselines/memnav/policy_agent.py::add_frame` 在线维护与
离线 precompute 同构的 scale/sparse-anchor/camera state；dense readout 复用
`_last_tokens/_last_agg/_psi`，CEC 复用 `_live_cache()`。后续不得另启第二个 LingBot map，
也不得把旧 MemNav learned decoder 混回唯一变量测试。

但现有 `_get_metric_scale()` 不是新短程分支的部署接口：它会使用调用时的全部累计历史，
而 formal Gate C 使用固定 RGB `0..39`。若 Gate C 授权，runtime 必须在第 40 帧生成一个
不可变 `first40_scale_receipt` 并终身复用；这是一处必要的契约实现，不是允许改用更长
未来历史来优化结果。

## 3. 短程分支为什么这样设计

### 3.1 输入与压缩

正式 adapter 输入：

```text
recent_specials       [B,8,6,2048]
current_patches       [B,1369,2048] -> deterministic pool -> [B,256,2048]
depth_features        [B,1369,256]  -> deterministic pool -> [B,256,256]
scale_quality         [B,6]
```

source 共 `48 + 256 + 256 + 1 = 561` tokens。128 个从官方 NavDP query slots 初始化的
query，经 2-layer Transformer resampler 输出 `[B,128,384]`。初始化只复制 slot 坐标，
不复制 teacher observation。

离线 shard 保存池化后的 tokens，不保存 37x37 原网格；池化无可学习参数，已有 exact
equivalence test，单样本存储约从 6.5 MB 降到约 1.3--1.6 MB。

### 3.2 训练目标

唯一训练参数为 adapter。LingBot、NavDP RGB-D teacher、goal encoder、diffusion decoder、
critic 和 CEC 全冻结：

```text
L = L_token + L_epsilon + 0.25 L_critic + 0.25 L_rank
```

- token：对齐 teacher observation latent；
- epsilon：同 goal、action、timestep、noise 下对齐 frozen diffusion function；
- critic：同候选轨迹下对齐分数；
- rank：对齐候选排序。

训练不使用 SR、Novel/Revisit、CEC accept 或最终场景标签。

### 3.3 冷启动

- 第 1--7 帧：冻结 NavDP RGB-only/zero-depth bootstrap；不读取 metric depth，不强制转圈；
- 第 8 帧后：LingBot scale block 因果可用，但 metric scale 仍标为 missing；
- 第 8--39 帧：必须使用真实 prefix stream，不能离线未来补帧；
- 第 40 帧后：只使用最早 40 帧估计的一次性 causal scale，进入 mature gate；
- prefix 是闭环前单独的强制门，不与成熟流平均隐藏。

## 4. 长程分支怎样与短程分支组合

CEC 不输出 metric waypoint，只输出：

```text
{certified unit bearing, abstain}
```

- reject：由同一个 `pi_mono` 继续 ImageGoal；
- accept：把 `2.5 m * unit_bearing` 作为有限残差，与 ImageGoal 一起送入冻结 NavDP；
- certificate reject 不等于语义 Novel，只表示当前历史证据不足；
- history expert 不覆盖短程几何 latent，两个时间尺度的职责不重复。

长程 translation 继续丢弃；短程分支需要局部控制尺度时，仅使用已报告的相机高度/尺度
质量。论文可以声称“无深度传感器”，不能声称“完全无尺度先验”。

## 5. 今晚发现并修复的关键数据问题

首次 smoke 显示 `zero_depth` 与所谓 metric teacher 几乎完全相同。审计确认：

- `generate_twoleg.py` 保存深度为 `uint16 = metres * 10000`；
- 旧诊断 loader 漏掉 `/10000`；
- NavDP `process_depth` 随后把几乎全部值当成 `>5 m` 并裁为 0。

因此旧的首次 Gate A/B、local shard 和 local Gate C receipt 已全部作废，见：

```text
.diagnostics/monocular_geometry_adapter_20260818/INVALID_DEPTH_UNIT_RECEIPT.json
```

修复后加了 fail-closed 单元测试；PNG 中位编码约 25539，正确解码约 2.55 m。

第二次 HPC smoke 又发现一个独立问题：旧 cache 的 `ground_h_est/ground_dbg` 是整条
episode 的 floor 统计。RGB/KV 虽然按 prefix 注入，但 scale token 和 raw-depth baseline
可能在 frame 40 使用未来帧尺度。v3 smoke 因此在运行 2:09 后被人工取消，formal 从未
启动；v3 不产生任何可用 Gate C 结论。

v4 修复为：每条 episode 只用 raw RGB `0..39` 和对应 causal camera poses 估计一次尺度，
后续 state 固定复用；whole-episode ground cache 被代码与 receipt 显式禁止。新增 prefix
边界测试后，当前相关测试为 22 passed。

## 6. 修正后的本地证据

### Gate A：通过

- teacher/student shape 均为 `[1,128,384]`；
- adapter 参数：6,023,424；
- 54/54 adapter tensors 有梯度；
- NavDP 与 LingBot 都是零梯度；
- 官方 NavDP SHA：`3bb3ad4a...011947`；
- LingBot SHA：`832bc82c...6f409`。

结果：

```text
.diagnostics/monocular_geometry_adapter_20260818/real_preflight_metric_depth.json
```

### Gate B：通过，但只代表可优化性

16 个固定真实 states、300 steps：

| loss | final / initial |
|---|---:|
| total | 0.164 |
| token | 0.128 |
| denoise | 0.106 |
| critic | 0.414 |
| rank | 0.539 |

结果：

```text
.diagnostics/monocular_geometry_adapter_20260818/metric_depth_gate_b/overfit_receipt.json
```

这些旧 receipt 使用了 whole-episode scale feature，因此只保留“接口有梯度、模型可过拟合”
的结论，不具有 causal 泛化解释权。v4 smoke 必须在 first-40 causal scale 下重新确认训练
闭包后，才授权 formal Gate C。

### 3-scene metric smoke：基线方向恢复合理

唯一 validation state（明确 underpowered）：

| representation | token cosine error | epsilon MSE |
|---|---:|---:|
| zero depth | 0.466 | 0.01235 |
| raw LingBot depth | 0.121 | 0.00166 |

这至少证明 teacher 不再是 zero-depth，并表明 raw-depth 必须保留为简单 baseline；但该
旧 smoke 的 scale 来自 whole-episode cache，数值只能作方向性诊断，不能当 causal 结果。
2 个训练样本上的 adapter 数字没有科学解释权，gate 正确返回
`underpowered_diagnostic_no_gate_decision`。

### 5090 v4 causal first-40 smoke：通过执行闭包，仍不作方法选择

使用与 HPC 完全相同的 v4 source、NavDP checkpoint 和 LingBot weights，在空闲 RTX 5090
上重建 3 scenes / 3 states。三条 scale receipt 均满足：

```text
scale_evidence_contract = causal_first_prefix_rgb_only_v1
scale_prefix = frames 0..39
whole_episode_ground_cache_consumed = false
```

唯一 validation state：

| representation | token cosine error | epsilon MSE | critic Spearman |
|---|---:|---:|---:|
| zero depth | 0.31964 | 1.4894e-4 | 0.20 |
| causal raw LingBot depth | 0.07539 | 1.1619e-5 | 0.80 |
| adapter | 0.97747 | 2.0626e-4 | 0.20 |

Raw path 在这个 state 上同时改善 representation、denoiser 和 critic；adapter 仅有两个
训练样本，数字没有泛化解释权。独立 verifier 返回 `verified=true`、
`authorized=false`、`underpowered_diagnostic_no_gate_decision`，符合冻结协议。

5090 输出：

```text
/home/cv/memnav_eval/mdtec_v4_causal_smoke_20260818
manifest SHA256 = b9246f9ee9d0b038c5c09d3a28497e1d650b3caac7300390e68d1846e31cd178
receipt SHA256 = a139c1f7fa6770924444eb2ae74aebec5e6e632a5e06b215c023afa5829c8339
verification SHA256 = 871bc1cfdaeed2e3a138bb28829c2f273216d7d631946ac569f7ef8ed1d09eda
```

## 7. Gate C 决策树（结果读取前冻结）

正式 validation 至少 4 scenes / 32 states。相对 zero-depth，候选必须同时满足：

1. token cosine error `<= 0.80x`；
2. epsilon MSE `<= 0.90x`；
3. critic Spearman 与 top-1 agreement 均不恶化超过 `0.05`；
4. 至少一个 critic 指标绝对改善 `0.02`。

选择规则：

- 仅 raw 通过：正式短程接口改为 raw LingBot metricized depth；
- 仅 adapter 通过：继续 latent adapter；
- 两者都通过：除非 adapter epsilon 比 raw 再低至少 10% 且 Spearman 不差，否则选 raw；
- 两者都失败：停止单目 controller 升级，CEC v1 保持 RGB-D-controller 口径。

这保证新方法不会因为“看起来更 learned”而保留不必要的复杂性。

## 8. HPC 冻结任务与基础设施审计

### 8.1 v2 失败尝试：未产生 Gate C 结果

v2 smoke `15954184` 在 shard 构建前 fail-closed。原因不是模型或数据内容，而是
enumerator 把 episode identity 写成 `(scene, episode)`；PT1 的 `mp3d_2leg` 与
`mp3d_3leg` 合法存在同名 episode，因而被错误判为重复。依赖任务 `15954185` 从未运行，
也没有产生任何 formal sample、checkpoint 或 Gate C 数字。

修复后 identity 改为 `(group, scene, episode)`，并加入跨 2-leg/3-leg 同名 episode 的
回归测试。旧 v2 bundle 和输出只保留作失败审计，不再复用。

### 8.2 v3 因果尺度审计：已取消、无有效结论

不可变 source bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
mdtec_gate_c_bundle_v3_20260818
archive SHA256 = dd9b84575fcc343588f7f19ce81d3896a67459e8275ab1220f10946554a4d623
SOURCE_BUNDLE.sha256 file SHA256 = 1b65e072f754bac9bf39ddb0d1c16c62e93c514601de85c83a5ad4b5694c857b
```

远端逐文件 `sha256sum -c` 已通过，目录随后设为只读。

最后一组 v3 任务：

| job | ID | 作用 |
|---|---:|---|
| environment smoke v3 | 15955321 | 运行 2:09 后因 whole-episode scale 泄漏主动取消 |
| formal Gate C v3 | 15955323 | 从未运行，已取消 |

早先提交的 v3 IDs `15954667/15954668`、`15955139/15955140` 均在零运行、零输出
状态下因资源碎片取消。v3 source/output 全部只作失败审计，不允许恢复或用于 Gate C。

### 8.3 v4 causal-first-40：teacher 输入缺失，未产生 Gate C

v4 唯一科学变化是删除 whole-episode scale input，改为
`causal_first_prefix_rgb_only_v1`。模型结构、scene split、样本规模、epoch、Gate C 阈值和
raw-vs-adapter tie-break 全部保持不变。必须使用新的 immutable bundle 与新输出根。

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/
mdtec_gate_c_bundle_v4_causal_scale_20260818
archive SHA256 = 41630c58219639a14f842c06185e93a0a51ef3d3c325100a81a7f2c696cb4e41
SOURCE_BUNDLE.sha256 file SHA256 = 057173e7ef8d201a900913679787072e6719f5e5301fd09eea80bdb02c1074bb
```

| job | ID | 作用 |
|---|---:|---|
| formal Gate C v4 | 15956337 | H100/H200/A100，40 scenes、最多 640 states |

`15956337` 于 2026-08-18 在 `ga002`（A100-SXM4-80GB）启动。official NavDP/LingBot
均正确加载，并写出 49 个完整 shard；随后在
`mp3d_2leg/82sE5b5pLXE/episode_0012/frame 240` 因 privileged teacher depth 全零而
fail closed。任务 exit 1，未训练、未产生 manifest/checkpoint/Gate C receipt。

CPU population audit `15957535` 在同一冻结 selection 上完成：40 scenes、160 episodes、
640 planned states 中 639 个合法，只有上述 1 个 state 为 270x480 全零 PNG；attrition
跨 1 scene/1 episode，且该 scene 属于 train，不改变 validation 的 8 scenes/128 states。
audit SHA256 为
`e6d44c9956ffabdb7ccd43716eb686de9634fbeb6adf24342e3366595ee489a4`。

v4 的 49 个完整 shard 已独立写入 `PARTIAL_SHARDS.sha256` 并校验，receipt SHA256 为
`7ff05b0fef86b87df1f446dad18adf35379b5670207507b8980432a7622b17d1`。它们只是 v5 的
verified resume cache，不是 Gate C 结果。旧 dependent audit `15957123` 因
DependencyNeverSatisfied 已取消。

L40S 尝试因当前 project account 的 QoS/account association 不兼容而在创建任务前被拒绝；
没有运行或输出。HPC smoke `15956023` 在零运行状态下取消，由完全哈希匹配且已独立验证的
5090 v4 smoke 取代；原 dependency formal `15956025` 同步取消。当前 formal 自身仍执行全部
source/asset/causal-scale fail-closed 检查，不依赖本地输出。

Formal fixed split 为 32 train scenes / 8 validation scenes；不读取 development、blind、
Final14，不运行 Habitat，不产生 SR。

Gate C 完成后还必须由
`independent_verify_monocular_geometry_gate_c.py` 在不导入 trainer 的条件下独立复算：
scene split、source/sample counts、全部冻结阈值、authorization 和 raw/adapter tie-break。
verifier 对正结果和负结果一视同仁；`verified=true` 不等于 `authorized=true`。

总量 Gate C 决策冻结后，另用
`audit_monocular_geometry_gate_c_strata.py` 对选中的同一 checkpoint 做只读 post-hoc
诊断：按 `scale_invalid / scale_valid_clamped / scale_valid_unclamped` 分层，并报告
critic 选中候选相对 RGB-D teacher 的 cumulative endpoint L2 与 endpoint-bearing error。
该审计明确写入 `posthoc_diagnostic_not_authorization=true`，不得重选 checkpoint、调阈值
或推翻冻结 Gate C 算术。相关纯函数与既有 Gate C/verifier 测试共 `8 passed`。

该 auditor 已在 5090 的 v4 causal smoke 真实 shard/checkpoint 上完成端到端演练：
zero/raw/adapter 的 token error、epsilon MSE、critic MSE、Spearman、top-1 五类 aggregate
与原 Gate C receipt 的绝对差均为 `0`，
`aggregate_reproduction_all_within_tolerance=true`。该 smoke 只有 1 个 validation sample，
只证明审计执行闭包，不增加方法证据。

### 8.4 v5 teacher-input attrition repair：当前有效任务

```text
source: mdtec_gate_c_bundle_v5_teacher_attrition_20260818
archive SHA256 = 840355211245660793e09b165db3eb25605f60ec53f6363f670f4787eb106d89
SOURCE_BUNDLE.sha256 file SHA256 = 244b7d640739d337e4f2d84f63f3be09329d930b4b11efe69a877572d3ed9009
formal = 15957726
afterok independent + strata audit = 15957727
```

v5 不更换 episode、不补 oracle depth、不改变模型/epoch/split/阈值。它只允许 pinned audit
中的一个全零 state 被显式排除；任何新 invalid state 仍失败。manifest 和 independent
verifier 必须复算 `640 selected = 639 valid + 1 invalid`。v5 在逐文件校验后复用 49 个
v4 shard；新 formal 从剩余人口继续。dependent audit 先校验 formal `OUTPUTS.sha256`，再写
独立 arithmetic verifier、stratified audit、`POSTHOC_RECEIPT.json` 与
`POSTHOC_OUTPUTS.sha256`；不训练、不跑 Habitat、不更改 Gate C。

启动审计已实际看到 `preseeded 49 verified shards`，并越过原失败点，输出
`[50/160] 82sE5b5pLXE/episode_0012 states=3`。相关 depth/identity/causal-scale/
Gate-C/verifier/strata 回归共 `28 passed`。

输出根：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/
mdtec_monocular_geometry_20260818/
  smoke_v4_causal_first40/
  formal_v4_causal_first40/
  formal_v5_causal_first40_attrition1/
```

## 9. Gate C 后唯一允许的下一步

若 Gate C 有路径通过：

1. 先补 `8..39` causal prefix/missing-scale gate；
2. 在已消费 MP3D 场景做 Gate D：RGB-D teacher / zero / raw / adapter 的严格同进程配对
   Novel base；
3. 只对 Gate D 非劣的 winner 测 CEC Revisit bearing 是否仍兑现；
4. 再测 latency、显存与双机 real-world 部署；
5. 所有闭环通过前，MDTEC 不升级为论文主方法。

若 Gate C 失败：不继续长训调参，不动 CEC 已成立主线。

Gate D 还必须防止一个新的 trajectory-source 泄漏：student 的 first-40 尺度只能来自该
arm 自己 bootstrap 后真实观察到的 RGB/pose，不能复用 PT1 teacher/expert prefix。除 SR
外需报告 frame-40 survival、scale-valid/clamped 比例和前缀运动量；否则 Gate C 的离线
有效尺度可能掩盖 zero-depth bootstrap 在闭环中缺少视差的问题。
