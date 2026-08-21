# X-NavDP Revisit Controller 冻结协议

日期：2026-08-09

状态：**实现与 smoke 获授权；任何闭环效果声明均未授权。**

## 1. 唯一问题

可靠 geometry memory 已经能在 Revisit-B 上产生当前机器人坐标系中的 metric
`PointGoal = [forward, left]`。现役系统把它与原 goal image 一起送入冻结 NavDP 的
mixed ImageGoal/PointGoal decoder。X-NavDP 最新公开结果则专门 post-train 了
PointGoal actor、twin-Q 与 embodiment conditioning。

本实验只回答：

> 在 memory 检索、几何验证、metric pose、fallback 与物理预算完全相同的条件下，
> X-NavDP PointGoal controller 是否比现役 mixed controller 更可靠地兑现已经验证的
> revisit 方向？

它不测试 Novel ImageGoal 方向来源，也不允许把 X-NavDP 的 PointGoal 结果写成
ImageGoal post-training 结果。

## 2. 为什么现在才允许替换 controller

现有 `4/40 -> 19/40` 实验固定了原 NavDP controller，因而可以把增益归因给 memory。
该因果问题已经回答后，controller 才能作为正交因素进入实验。此前 N=9 的 X-NavDP
方向探针来自 consumed Novel-A plan-0 state，不是 geometry-activated revisit
closed loop，只能验证源码、checkpoint 与动作模态，不能决定本实验结果。

## 3. 冻结四臂

所有臂共享 Goal-A、episode seed、memory 数据、geometry router、成功半径和物理预算：

1. `native_image`：原 NavDP ImageGoal；
2. `memory_mixed`：geometry memory + 原 NavDP ImageGoal/PointGoal mixed decoder；
3. `memory_base_point`：geometry memory + 原 NavDP pure PointGoal decoder；
4. `memory_xnavdp_point`：geometry memory + 官方 X-NavDP post-trained PointGoal actor、
   twin-Q 与 wheeled embodiment conditioning。

臂 3 防止把“去掉 image token / 攟成 pure PointGoal”误归因给 X-NavDP post-training。
正式比较的两个问题预先固定为：

- 部署比较：`memory_xnavdp_point` vs `memory_mixed`；
- post-training 归因：`memory_xnavdp_point` vs `memory_base_point`。

X-NavDP 的 Q 只在同一个已给定 PointGoal 的 trajectory samples 内选轨迹，禁止跨
memory anchor 或跨 PointGoal 把 Q 当作方向来源。

## 4. 不可破坏的在线契约

- 上游源码 commit 固定为
  `878740a2011856d0e3782dd6ccd880fd2eccd70f`；
- commit-pinned checkout 的 tracked worktree 必须为空，禁止用相同 HEAD 掩盖源码修改；
- post-trained checkpoint SHA256 固定为
  `267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24`；
- 启动时必须复核 official eval model 对 checkpoint 的加载覆盖：model state 1329 个 tensor
  全部命中（missing=0、shape mismatch=0）；checkpoint 额外的 357/1686 个 tensor 来自该
  PointGoal eval 类未构造的 ImageGoal/pixel/旧 critic 模块。覆盖审计不符时 fail closed，
  不能因官方 `strict=False` 静默使用随机初始化参数；
- 原 NavDP checkpoint SHA256 固定为
  `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947`；
- Goal-A 永远由原 NavDP ImageGoal 控制，X-NavDP 只 shadow 写入相同的 decision-frame
  history，不允许在 A 段影响轨迹；
- X-NavDP 每个 planning request 恰好 append 当前 RGB 一次；fallback/native request
  通过 read-only replay endpoint 同步一次，禁止双写；
- evaluator 在每个 episode 末核对 `history_frame_count == Goal-A plans + Goal-B plans`，
  单次 `+1` 正确但累计计数不符时仍判整臂 invalid；
- episode reset 同时清空原 NavDP FIFO、X-NavDP FIFO/RTC 与 memory router latch；
- 每个 diffusion request 接收并回显冻结 seed；
- Habitat 地面坐标映射为 X-NavDP 世界坐标 `[x_hab, -z_hab, 0]`，yaw 映射为
  `theta = pi/2 + psi_hab`，使 X-NavDP local `[x,y]` 严格等于 `[forward,left]`；
- X-NavDP response 缺失、NaN、shape 错误、源码/checkpoint 身份不符时整个正式臂
  invalid，不能静默记作 native；
- benchmark 仍以距离成功和完整物理执行计分，不用 heading fidelity 代替 SR/SPL。

## 5. 分阶段授权

### S0：工程 smoke，不产生效果结论

- fake-policy 契约测试：reset、history 单写、seed 回显、pose axis、response shape；
- 1 个 train scene / 1 个 episode：四臂均能完成 reset 与至少一个 planning request；
- Goal-A trajectory hash 在三个 memory-controller 臂间完全一致；
- source/checkpoint/protocol SHA 写入 receipt。

### G1：已耗尽 20-scene 池上的 controller gate

这一步只用于发现 controller 是否值得送往未见池，不是论文确认：

- 20 scenes x 2 episodes，逐 episode、同机、同进程族配对；
- 主指标为 conditional Revisit-B SR；同时报告 joint SR、SPL、final distance、
  gain/loss、exact McNemar、scene-cluster interval；
- 单独报告 common-activation 与 router-divergence，但不得只挑 activated successes；
- 任一 controller arm 改变 Goal-A trajectory hash，整次 gate invalid。

只有 `memory_xnavdp_point` 对 `memory_mixed` 净 gain > 0，且无安全/invalid-response
回归，才允许进入 G2。安全门在看 G1 结果前具体化为：X history/response invalid 为零，
checkpoint coverage audit invalid 为零，且不存在 `X=stuck、mixed!=stuck` 的新增配对失败；
navmesh 拒绝的 blocked-step 总数同时完整报告，但不再事后发明阈值。这里不要求 G1
显著，因为它是 consumed train-scene gate；也不允许按 bearing 区间事后选择“rear-only”
阈值。

### G2：scene-disjoint confirmation

G1 通过后，才从冻结 535 条 2-leg 池按 scene hash 预选固定样本量；样本量必须在看
结果前由保守的 paired power calculation 决定。G2 之前不得读取所选场景结果，不得
使用 development/blind 调参。

## 6. 结果分叉

- X-NavDP 全局 controller 通过 G2：作为 verified-memory 后的默认 PointGoal executor；
- 全局不通过但预注册诊断显示只新增后向模态：重新开一个独立 rear-recovery safety /
  progress 协议，不能复用 G1 选择角度阈值；
- 无净增益：保留现役 mixed controller，结论为 X-NavDP 的公开 PointGoal post-training
  不迁移到本 revisit/Habitat 控制接口。

## 7. S0 本机工程回执（不构成效果证据）

官方 checkpoint 真实推理已经通过：同一 reset/history/PointGoal/seed 重复两次，trajectory、
8 个 candidates 与 twin-Q 逐元素一致；history 为 `0 -> 1 -> 2`。两个完全独立的新服务
进程也逐元素一致，trajectory/candidates/Q 联合 SHA256 均为
`f7d7977d001e36f6939bb2b5a64bca1ee2457c72c1b2b804a3ebf645f5fa43d0`。随后在
`s8pcmisQ38h/episode_0000`、同一冻结 Goal-A trace
`9e2d5fff5db2917c76074b6362aa71bff001b3ab85ebfe3a6c614861c01adaa3` 上完成三种 memory
controller 的闭环 S0：

| controller | B success | steps | path m | SPL |
|---|---:|---:|---:|---:|
| mixed | 1/1 | 130 | 4.654 | 0.900 |
| base pure PointGoal | 1/1 | 132 | 4.732 | 0.885 |
| X-NavDP PointGoal | 1/1 | 125 | 4.217 | 0.993 |

X arm 的累计 history 为 28/28，RTC robot state 在 15 个 active requests 上启用。该表只证明
归因接口可运行，并提示改善不等同于简单移除 image token；`N=1` 禁止用于选择模型或宣称
增益。
