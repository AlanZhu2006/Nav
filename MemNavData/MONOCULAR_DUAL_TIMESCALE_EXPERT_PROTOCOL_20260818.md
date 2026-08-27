# 单目双时间尺度专家组合：冻结协议与今晚执行门

更新时间：2026-08-18（Asia/Shanghai）  
状态：架构/开发协议；不改变已冻结的 CEC Final14 方法与结果。

## 0. 目标与边界

目标是在**部署时不使用 metric-depth 传感器**的前提下，复用已经验证的两类能力：

1. LingBot-Map 从单目 RGB 流中提取短程几何、相机状态和长程视觉历史；
2. 冻结 NavDP 保留 ImageGoal 条件、diffusion trajectory prior 与 learned critic；
3. CEC 继续只在历史证据通过时输出 scale-free bearing，失败时回退同一个单目
   NavDP base，而不是另训一个 fallback policy。

本协议区分：

- **无深度传感器**：部署输入没有 Habitat/RealSense/双目 metric depth；这是目标；
- **完全无尺度信息**：单目绝对尺度在几何上不可观，不能保证米制避障；这不是可实现
  的声明。固定相机高度、相机内参及可选机器人里程计必须明确报告；
- simulator metric depth 可在训练中作为 privileged teacher，但不得进入 student 部署。

本轮不重新训练 NavDP decoder，不替换 CEC，不读取 Novel/Revisit role，不使用 Final14
做模型选择。

## 1. 新架构

```text
                            causal monocular RGB stream
                                      |
                              frozen LingBot-Map
                     +----------------+----------------+
                     |                                 |
          dense local-control readout       sparse episodic-proof readout
       recent pose/special tokens + current      history + DINO address
       geometry patches/depth feature/conf              |
                     |                              frozen CEC proof
          Geometry Token Adapter A_phi                 |
                     |                        {unit bearing, abstain}
        NavDP-compatible [128,384] latent               |
                     +----------------+----------------+
                                      |
                         frozen NavDP goal encoder
                         + diffusion decoder + critic
                                      |
                                  trajectory
```

这不是传统 action-level Mixture-of-Experts，而是 **one map, two readouts, one policy**：
LingBot 的同一个 causal sparse-map state 只计算一次；dense readout 每步生成局部控制表征，
sparse readout 仅在历史证据充分时生成可审计证明。NavDP 始终是唯一动作生成器。二者通过
类型受限的 latent/bearing 接口组合。论文暂称
**Monocular Dual-Timescale Expert Composition (MDTEC)**；在形成闭环结果前不改 CEC
论文标题。

“dense/local”描述的是 readout 的职责和输出频率，不表示其上游 token 完全看不到旧历史：
LingBot 当前 token 仍可注意到 causal sparse keyframes。长程分支的独特职责是显式检索、
几何自认证与 abstention，而不是再维护第二份地图。

## 2. 为什么不是直接替换深度图

NavDP 的原生 `process_depth` 把输入解释为米，并裁剪到 `[0.1,5.0] m`。已经观察到
一条 runtime 中 LingBot scale 为 `4.10`，与 GT 距离一致的 scale 约 `1.75`，导致
`15.29 m` request 对应真实 `6.54 m`。因此未校准 raw LingBot depth 只能做诊断。
使用固定早期 RGB prefix、已知相机安装高度和显式质量字段得到的 causal metricized depth
仍可参加 Gate C/D；只有同时通过功能门和闭环门后，才有资格成为正式的简单接口。

旧 MemNav 也不是纯 depth-swap：它同时替换了 RGB-D backbone、goal encoder、decoder、
critic 和 selector，Novel 从原生 NavDP 的 `9/10` 降到 `4/10`，不能用于归因 depth。

## 3. 短程 Geometry Token Adapter

### 3.1 输入

每个决策只消费因果可用量：

```text
recent_specials : [B, 8, 6, 2048]
current_tokens  : [B, 6+37*37, 2048]
depth_features  : [B, 37*37, 256]
scale_quality   : [B, 6]
```

`scale_quality` 固定字段：

```text
[log(camera_height_m), log(scale_hat), scale_valid,
 valid_frame_ratio, relative_floor_iqr, scale_clamped]
```

尺度不可用时使用显式 missing state：`log(scale_hat)=0, scale_valid=0`，不得填 pooled
oracle constant 后伪装为有效。

Mature-stream Gate C 的尺度证据冻结为每条 episode 的最早 40 帧 RGB prefix：

- 只读取 raw frames `0..39` 及其 causal streaming camera poses；
- 估计一次后对该 episode 的全部 `frame>=40` states 固定复用；
- 禁止读取预计算 cache 中由整条 episode 汇总的 `ground_h_est/ground_dbg`；
- `frame<40` 的 prefix gate 使用 missing-scale 状态，单独报告。

在线实现不得直接调用现有 `MemNavAgent._get_metric_scale()` 作为该字段：旧 helper 会按
调用时已经累积的全部历史重新估计尺度，虽然因果，却不再与 Gate C 的 fixed-first-40
分布相同。Gate C 通过后的 runtime 必须在收到第 40 帧时只计算一次并冻结
`first40_scale_receipt`；后续 dense readout 和 raw-depth baseline 均复用它，CEC 的
scale-free bearing 不依赖它。

为控制内存并与 NavDP patch resolution 对齐：

- 当前 LingBot `37x37` patch 自适应池化为 `16x16`；
- 当前 depth feature `37x37` 同样池化为 `16x16`；
- 最近 8 帧各保留 6 个 special/pose/register/scale tokens；
- 加入一个尺度质量 token 和 modality embeddings；
- 共约 `48 + 256 + 256 + 1 = 561` 个 source tokens。

### 3.2 输出

使用 `128` 个查询进行两层 cross-attention resampling：

```text
A_phi(source) -> z_mono [B,128,384]
```

输出形状严格等于官方 NavDP `rgbd_encoder`。128 个 query 可以从官方 checkpoint 的
`rgbd_encoder.former_query.position_embedding.weight` 初始化；这只是坐标/slot 初始化，
不是拷贝 teacher observation。

### 3.3 冷启动（修正版）

LingBot 的稳定流式坐标需要首个 scale block。为避免“先移动才能估几何、但先有几何
才能移动”的循环：

1. LingBot 的冻结 scale block 需要 8 个真实观察；在第 8 帧之前，使用冻结 NavDP 的
   RGB-only/zero-depth bootstrap，不读取 metric depth，也不强制原地转圈；
2. 第 8 帧后才允许构造 LingBot scale block；不足 32 帧时使用真实因果 prefix forward，
   不能用未来帧补齐，也不能提前读取离线 `scale_k/v`；
3. 第 8--39 帧 scale token 仍显式 unavailable；从第 40 帧起才允许使用冻结的 first-40
   RGB-only floor scale，不能为了提前激活读取未来帧；
4. Gate C 的成熟流结果与 `8..31` prefix 结果必须分开报告；今晚先验证成熟流表示，
   prefix 是闭环前的独立强制门；
5. 若短 prefix 无法逼近 teacher，系统仍可称为无深度传感器，但必须把启动阶段明确写成
   RGB-only bootstrap，不能宣称 adapter 从第一步接管。

## 4. Teacher/student 训练契约

### 4.1 冻结范围

冻结且 `requires_grad=False`：

- LingBot-Map 全部权重；
- NavDP RGB-D teacher；
- NavDP image/point goal encoders；
- NavDP diffusion decoder、action head、critic；
- CEC retrieval、matcher、PnP 与 certificate。

唯一可训练模块为 `A_phi`。优化器发现其他 trainable parameter 必须 fail closed。

### 4.2 同状态 teacher

```text
z_teacher = NavDP_RGBD_Backbone(last_8_RGB, current_metric_depth)
z_student = A_phi(LingBot(last_RGB_prefix), scale_quality)
```

teacher/student 必须使用同一 current frame、同一八帧 FIFO 和同一预处理。metric depth
仅用于 teacher forward 和标签审计，不进入 student input。

### 4.3 Loss

```text
L = 1.0 L_token + 1.0 L_denoise + 0.25 L_critic + 0.25 L_rank
```

- `L_token`：LayerNorm 后的 Smooth-L1 加 token cosine；
- `L_denoise`：同一个 goal、clean action、DDPM timestep 和 noise 下，冻结 decoder 对
  teacher/student latent 的 epsilon prediction 一致性；
- `L_critic`：同一 label/augment trajectory 的 frozen critic score 一致性；
- `L_rank`：同一候选集的 pairwise critic ordering 一致性。

不把 SR、Novel/Revisit 标签或 CEC accept 当训练标签。goal encoder/decoder 冻结，从
结构上保留原 NavDP 的 goal sensitivity。

## 5. 长程分支与控制接口

CEC 的 proposal/witness/interface 不变：

```text
history + ImageGoal
 -> DINO top-8
 -> local geometry + LingBot historical depth + PnP
 -> atomic certificate
    reject: pi_mono(ImageGoal)
    accept: pi_mono(ImageGoal, 2.5m * unit_bearing)
```

历史分支仍丢弃单目 metric translation。短程 adapter 可以使用尺度质量来恢复局部避障
表征，但长程 CEC 不得因此重新输出 metric waypoint。

在纯单目版本中，`exact fallback` 的含义是回退同一个 `pi_mono` 请求；它不再等于
RGB-D teacher 的物理轨迹。RGB-D CEC v1 的既有 exact-fallback 结论保持不变，不能与
新版本混写。

### 5.1 在线实现不允许复制地图

现有 `NavDP/baselines/memnav/policy_agent.py::add_frame` 已经与离线
`precompute_lingbot_features.extract_trajectory` 使用同一 capture contract：首 8 帧 scale
block、随后 causal keyframes、`scale_k/v`、`anchor_k/v`、camera poses 与 flow-gated sparse
indices。正式单目实现必须复用这一个 live state：

- dense readout 读取当前 `_last_tokens/_last_agg/_psi` 和最近 8 个 special-token records；
- raw-depth 候选复用当前 frozen depth head，不另跑第二个 depth model；
- adapter 候选复用同一 current depth feature 与 scale-quality receipt；
- CEC 继续通过 `_live_cache()` 读取同一份 sparse history；
- 任一 plan-time replay 必须 snapshot/restore live KV，不能重复写当前 RGB/FIFO。

因此 Gate D 前只需增加一个 typed observation-latent API；不得启动第二个 LingBot stream，
也不得把 MemNav learned decoder 重新引入控制路径。

## 6. 三个先验 baseline

在训练 adapter 前固定比较：

1. `metric_teacher`：原生 NavDP + simulator metric depth；
2. `raw_lingbot_depth`：raw depth 按当前 scale 直接转换，显式记录 invalid/clamp；
3. `zero_depth`：NavDP depth 全零，测量 RGB 分支本身的下限；
4. `latent_adapter`：正式 student。

若 raw depth 已与 teacher 功能等价，复杂 adapter 没有必要；若 zero-depth 已接近 teacher，
论文也必须诚实说明 NavDP 对 depth 的边际依赖有限。

## 7. 今晚分阶段判据

### Gate A：接口与梯度

- output 恰为 `[B,128,384]`；
- NaN/Inf 为零；
- teacher query 初始化可复算；
- backward 后只有 adapter 有非零梯度；
- checkpoint 保存只含 adapter/config/hash receipt。

### Gate B：小样本过拟合

在已消费本地数据上固定 16 个 frame states：

- 200--500 steps 内 total loss 明显下降；
- token loss、denoise loss 不能一升一降地互相掩盖；
- 同一输入重复 forward bitwise/deterministic tolerance 一致；
- 不能读取 Final14/HM3D held-out outcomes。

Gate B 失败则停止，不提交长训。

### Gate C：scene-grouped 离线泛化

PT1 只按 scene 分 train/validation，不按随机 frame 分割。报告：

- normalized token cosine/MSE；
- epsilon prediction cosine/MSE，按 DDPM timestep；
- critic Spearman、top-1/top-2 agreement；
- candidate endpoint/heading distribution disagreement；
- early-prefix 与 mature-stream 分层；
- scale-valid/invalid/clamped 分层；
- 每条 scale receipt 必须声明 `causal_first_prefix_rgb_only_v1` 且
  `whole_episode_ground_cache_consumed=false`。

在读取 Gate C 结果前冻结以下授权阈值。相对 `zero_depth`，候选路径必须同时满足：

- token cosine error 不超过 `0.80x`；
- epsilon MSE 不超过 `0.90x`；
- critic Spearman 和 top-1 agreement 均不恶化超过 `0.05`；
- 至少一个 critic 指标绝对改善 `0.02`。

若 raw-depth 与 adapter 都通过，除非 adapter 的 epsilon MSE 至少比 raw-depth 再低 10%，
且 critic Spearman 不差，否则选择更简单的 raw-depth 接口。有效决策至少需要 4 个
validation scenes、32 个 states；更小运行只能叫 smoke。

通过标准不以单一 latent cosine 决定；必须同时看到 functional denoiser 与 critic 改善。
训练脚本生成的 Gate C 还必须由不导入 trainer 的独立 verifier 复算 scene split、样本数、
阈值和 raw/adapter tie-break；`verified=true` 不自动表示 `authorized=true`。

### Gate D：已消费闭环

先在已消费场景做严格配对 Novel base test，唯一变量为 observation adapter。之后才测试
CEC Revisit 是否仍能兑现 bearing。不得在该结果上调阈值后把相同场景称为确认集。

Gate D 的 `first40_scale_receipt` 必须由各 arm 自己实际执行的 causal RGB prefix 产生；
禁止把 PT1 teacher/expert 的前 40 帧、pose 或 scale receipt 注入 student rollout。各 arm
必须另报 frame-40 survival、scale-valid/clamped 比例和 bootstrap 前 40 帧的运动量。正式
配对比较仍共享 seed、episode 与 Goal，不共享会受 observation arm 影响的在线轨迹。

## 8. HPC 长训授权

只有 Gate A/B 全部通过才允许打包 6--8 小时训练。长训必须：

- 使用 PT1 Train scenes；
- scene-grouped fixed validation；
- checkpoint 保存 adapter-only；
- 记录 official NavDP SHA-256 `3bb3ad4a...011947`、LingBot weight hash、数据 manifest、
  git diff/bundle hash；
- W&B 或不可变 JSON receipt 分别记录 token/denoise/critic/rank 和 scale/prefix strata；
  若共享环境的 W&B import/网络路径成为运行瓶颈，先写本地完整 history，训练后再同步，
  不得让观测工具改变科学任务；
- 不自动启动闭环、不读取 development/final/blind。

## 9. 成功与停止条件

成功不是“loss 能降”，而是：

1. RGB-only student 在 scene-disjoint Novel 闭环对 RGB-D teacher 达到预注册
   non-inferiority；
2. CEC Revisit 增益在 `pi_mono` 上仍存在；
3. unsupported query 回退 `pi_mono` 不产生额外 memory interference；
4. 实际部署延迟、显存和启动前缀可接受。

任一项失败，CEC 论文继续保持“monocular episodic relocalization sidecar for a frozen
RGB-D navigator”的诚实定位；不得通过改写叙事宣称全系统单目。

## 10. 当前证据状态

- 已支持：CEC bearing/abstention、frozen NavDP controller、LingBot 单目历史几何；
- 未支持：LingBot geometry 能替代 NavDP metric-depth latent；
- 旧 MemNav `4/10` 不能回答该问题，因为它不是单变量 depth/latent replacement；
- 本协议定义的是一个新的 prospective student，不修改任何已打开的 Final14 结果。
