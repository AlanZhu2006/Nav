# 单目双时间尺度导航状态 — 2026-08-19

> 更新提示：本文写于 CEC+mono 正式任务尚未完成时。20-scene 完整结果及 2026-08-18--19
> 的统一项目总账现以 `STATUS_20260819_TWO_DAY_PROGRESS_FULL.md` 为准。

本文件 supersede `MONOCULAR_DUAL_TIMESCALE_EXPERT_STATUS_20260818.md`。当前正式主线不是
再叠加一个 controller，而是让同一个 frozen LingBot 表征产生两个不同时间尺度的、可审计
读出，最后仍由 frozen NavDP 唯一生成动作。

论文主线、代码入口、证据边界和禁止口径已统一整理到
`ARCHITECTURE_20260819_PAPER_SOURCE_OF_TRUTH.md`。尤其要区分：已确认的 CEC 主结果保留
NavDP 原生 metric RGB-D；本文件的 raw-mono 路线是独立部署扩展。

## 1. 当前架构

```text
causal monocular RGB stream
          |
          v
 one frozen LingBot-Map stream
   |                         |
   | dense short readout     | sparse long readout
   |                         |
 raw relative depth          DINO temporal top-8
 + first-40 RGB/pose scale   + SuperPoint/LightGlue
   |                         + LingBot historical depth/PnP
   |                         + atomic geometry certificate
   |                                 |
 unchanged NavDP RGB-D encoder       +-- pass: scale-free bearing
   |                                 +-- reject: abstain
   +-------------------------+---------------+
                             |
          frozen NavDP goal encoder + diffusion decoder + critic
                             |
                         trajectory
```

概括为：**one causal stream, two time scales, one frozen policy**。

- 短程专家不规划，只给 NavDP 提供其原生 RGB-D encoder 所需的 depth；
- 长程 CEC 不规划，只在 Revisit 可自认证时提供固定半径的 scale-free bearing；
- NavDP checkpoint、goal encoder、diffusion decoder、critic 与执行器均冻结；
- 部署传感器只有 RGB 与已知相机安装高度；Habitat metric depth 在 raw arm 中被显式忽略；
- CEC 的阈值与生命周期保持冻结；Gate D 工程授权后，单目 controller 上的 CEC off/on
  组合复测已经启动。

## 2. Gate C 已完成：为什么选 raw depth，而不是 Adapter

正式人口：40 scenes、160 episodes、640 selected samples，其中 639 valid，1 个预先固定的
all-zero invalid teacher sample；32 train scenes / 511 samples，8 validation scenes / 128
samples，scene overlap 为 0。formal job `15957726`，独立 audit `15957727`，verifier
`verified=true`、`authorized=true`。

| validation 指标 | zero depth | raw LingBot depth | 6.02M Geometry Token Adapter |
|---|---:|---:|---:|
| RGB-D token cosine error ↓ | 0.3024 | **0.1812** | 0.3871 |
| diffusion epsilon MSE ↓ | 0.01152 | **0.00591** | 0.01553 |
| critic Spearman ↑ | 0.6250 | **0.7672** | 0.5328 |
| critic top-1 agreement ↑ | 0.6016 | **0.7266** | 0.5391 |
| critic MSE ↓ | 0.2010 | **0.06918** | 0.4327 |

post-hoc action disagreement 同样一致：raw 的 selected endpoint L2 为 `0.569 m`、heading
error `23.14 deg`；zero 为 `0.948 m / 39.32 deg`；Adapter 为
`1.016 m / 45.23 deg`。

因此冻结结论是：

1. raw LingBot depth 通过 Gate C；
2. 学习型 Adapter 不只是“没有明显增益”，而是显著破坏了 NavDP 已有的 depth interface；
3. 停止 Adapter 长训，采用更小、更可解释的 raw-depth 接口；
4. Gate C 仍是策略可消费性指标，不是闭环 SR。

正式结果根目录：
`/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_geometry_20260818/formal_v5_causal_first40_attrition1`。

## 3. 已实现的因果在线单目 depth contract

核心实现：

- `MemNavData/monocular_depth_runtime.py`；
- `NavDP/baselines/memnav/policy_agent.py`；
- `NavDP/baselines/memnav/memnav_server.py`；
- `NavDP/baselines/navdp/navdp_server.py`；
- `MemNavData/eval_2leg_habitat.py`。

冻结行为：

1. observation frame 0..39 向 NavDP 输出逐像素全零 depth；
2. 收到恰好 40 张 causal RGB 后，只 replay 0..39 一次并冻结 scale receipt；
3. frame index >=40 才输出 `LingBot relative depth * frozen scale_hat`；
4. scale invalid 时 fail closed 为 zero，禁止 pooled/teacher/oracle fallback；
5. depth PNG 与当前 JPEG SHA-256 绑定，stale query 返回错误；
6. `monocular_sidecar` 忽略上传的 simulator depth，并报告
   `metric_depth_sensor_consumed=false`；
7. 同一 RGB stream 同时保留给未来 CEC，不启动第二份 LingBot map。

真实 41-frame prefix smoke 和 NavDP wire smoke 已通过。专用三臂 Habitat smoke
`.diagnostics/mdtec_gate_d_habitat_smoke_20260819/specialized_v3` 进一步确认：metric / zero /
raw 能在同一 server pair 中切换；raw 的前 5 个 plan 为 zero、后 5 个 plan 为 active
LingBot depth，scale valid，metric sensor 未消费。该 80-step smoke 不进入任何 SR 结论。

## 4. Gate D 已完成：raw mono 有用，但尚未证明等价于 metric teacher

Gate D 回答：**在真实 Novel Goal-A rollout 中，因果单目 raw depth 是否给 frozen NavDP
提供可消费的短程几何，而不是只在 latent 指标上接近 teacher？**

冻结人口为已消费 MP3D 20 scenes / 40 episodes，三臂同机、同进程、顺序旋转：

| arm | SR | mean SPL | mean path |
|---|---:|---:|---:|
| metric teacher | 30/40 = **75.0%** | 0.7327 | 7.71 m |
| raw LingBot first-40 | 27/40 = **67.5%** | 0.6307 | 9.14 m |
| zero depth | 23/40 = **57.5%** | 0.5455 | 9.90 m |

严格配对：

- raw vs metric：`+2/-5`，风险差 `-7.5 pp`，McNemar `p=0.4531`，scene-cluster
  95% CI `[-20,+5] pp`；
- raw vs zero：`+6/-2`，风险差 `+10 pp`，`p=0.2891`，CI `[-2.5,+22.5] pp`；
- zero vs metric：`+2/-9`，风险差 `-17.5 pp`，`p=0.0654`，CI `[-32.5,-2.5] pp`。

部署审计全部通过：raw 40/40 不消费 simulator metric sensor，40/40 到达 frame 40，40/40
得到有效且实际被消费的冻结 scale，2/40 scale 被预定边界 clamp；独立 verifier
`verified=true`。因此预注册的**工程授权通过**：`continue_to_cec_on_monocular`。

但 10 pp 论文 non-inferiority **没有通过**，因为 raw-vs-metric cluster CI 下界为 -20 pp。
正确口径不是“单目已等价 RGB-D”，而是：raw mono 相比 zero 的闭环点估计为 `+10 pp`
（本次未显著），并与 Gate C 的策略可消费性证据方向一致，足以按冻结工程门进入 CEC 组合；
40 条不足以排除相对 zero 的零效应，也不足以排除相对 teacher 的较大性能损失。

正式结果：
`/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_raw_depth_gate_d_20260819/formal_20260818T171542Z_b8d2ffb5/POSTHOC`。
H200 task 11/16 的运行时事件及 A100/H100 原样修复记录见
`MDTEC_RAW_DEPTH_GATE_D_ATTEMPT3_H200_RUNTIME_INCIDENT_20260819.json`；最终 20 scenes / 120 arm
records 完整，不含 outcome 后换卡或换配置。

## 5. CEC + mono full composition 正在运行

Gate D 通过工程门后，条件冻结的组合实验已正式启动。它不再比较 depth，而只问：

> 在完全相同的 raw-mono NavDP controller 和完全相同的 causal Goal-A history 上，打开 CEC
> 的 sparse long-horizon readout，是否相对逐动作 native fallback 改善 Revisit Goal-B？

冻结人口：Fresh160 immutable manifest 的每场景前 2 条，共 20 scenes / 40 episodes。两臂：

1. `raw_native`：`monocular_sidecar + native ImageGoal`；RGB 继续写入同一个 LingBot stream，
   但不允许 memory takeover；
2. `raw_cec`：同一个 `monocular_sidecar`，仅打开已经冻结的 top-8 / LightGlue / PnP /
   atomic certificate / fixed-2.5m bearing；reject 必须逐动作复现 native。

每条 episode 的 Goal-A raw-mono rollout 只运行一次，再由 SHA 绑定的 trace 向两臂重放；两臂
共享同一 loaded LingBot/NavDP server pair，不读取 Novel/Revisit role 标签。主要报告 Goal-A
成功后的 B，同时保留 40 条 intent-to-treat。该人口已被消费，因此是**组件兼容性与组合性
实验**，不是新的 held-out 论文确认集。

当前正式任务：

- GPU array `16009201`，20 scene tasks，A100，1 小时/task；
- primary summary/verifier `16009253`，依赖整个 array `afterok`；
- immutable source：
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mdtec_cec_composition_624f9fa9d06ffed6`；
- run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_cec_composition_20260819/formal_20260819T055600Z_624f9fa9`。

截至 2026-08-19 03:02 EDT：2/20 scenes 完成，3 个 scene tasks 正在 A100 上运行，其余因
`QOSGrpGRES` 等待；单 scene 当前约 8--10 分钟。正式 SR 在 20/20 和 verifier 结束前不读取。

调度审计发现重复 summary `16009280` 与 `16009253` 完全相同并会写同一 POSTHOC，已取消
后提交的重复项，避免并发覆写。原始 evaluator 实际保留了每臂 `metric.csv/final_dist_B`，但
初版 independent verifier 只复算 compact CSV 的 success flag。该论文级缺口已在本地修复并
通过 61 项测试；只读 distance verifier `16010883` 已挂在 `16009253` 之后，将从原始最终距离
按 `<1 m` 独立重算，不修改 rollout 或 primary summary，也无需重跑 GPU。

## 6. 这组结果对架构的真正含义

当前最稳的架构结论是 **one map, two readouts, one frozen policy**，而不是“训练出一个新的
大 controller”：

- Gate C 说明 NavDP 的 depth interface 已经是高度校准的；6.02M Adapter 试图重写这个
  interface，反而把表征推离 frozen decoder 熟悉的流形。简单的相对深度 + 一次因果尺度冻结
  保留了空间结构，因此比学习 Adapter 更好；
- Gate D 的短程单目几何点估计不是装饰：raw 为 67.5%，zero 为 57.5%，但该差异在 N=40
  尚未显著；metric teacher 仍有 7.5 pp 点估计优势。短程 readout 应被定位为可部署桥接层，
  不应抢占论文主贡献；
- CEC 仍是方法主角：它处理几百帧尺度的内容寻址与开放集授权，只在证据足够时把 scale-free
  bearing 交给同一 frozen NavDP。短程 depth 与长程 certificate 的接口不同、时间尺度不同，
  强行统一成一个 token adapter 既没有证据，也会丢掉 abstention 的可审计性。

组合实验有三种预先可解释的结果：

1. `raw_cec > raw_native` 且 reject audit 通过：证明单目短程控制与 certified long memory
   可以无冲突组合，形成完整部署架构；
2. 两者持平：说明 raw controller 可工作，但该 40 条已消费 Revisit population 上 CEC 没有
   新增闭环价值，不能用工程组合替代既有 Final14/HM3D CEC 证据；
3. `raw_cec < raw_native`：优先诊断 first-40 scale 生命周期、shared trace replay 与 CEC
   接管后的状态分布失配，不允许调 threshold 追结果。

目前已经成立：Gate C 选出了 raw interface；Gate D 给出完整 N=40 闭环和工程授权；完整
rollout 中不消费 metric depth；CEC+mono 的正式配对正在运行且审计链已补齐。

尚未成立：raw mono 与 metric teacher 的统计等价；CEC+mono 的组合净增益；该组合在新的
held-out population 或真机上保持同样效果。论文叙事应把“单目双时间尺度接口”作为系统化
机制，把 CEC 的 role-free Revisit utility/safety 作为主要实证贡献，避免把一个未通过
non-inferiority 的短程替代包装成核心突破。
