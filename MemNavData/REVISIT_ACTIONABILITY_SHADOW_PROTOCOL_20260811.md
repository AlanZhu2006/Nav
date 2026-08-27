# REVISIT Actionability Shadow：冻结机制探针

日期：2026-08-11（CST）  
状态：协议冻结；consumed-pool 机制调试；不控制动作、不选阈值、不授权性能声明。

## 问题

已知 Revisit direct 相对 geometry router 得到 `+6/-1`。唯一 loss
`pLe4wQe7qrG/episode_0001` 的 DINO anchor 距事实轨迹最近 anchor 仅 `0.07–0.37 m`，
但 memory-conditioned NavDP rollout 出现短/零轨迹。因此本探针只检验：

> 在 anchor 已提出后，冻结 NavDP 的候选 rollout 是否包含独立于 place matching 的、可部署
> actionability 信号？

## 因果传输

每个 Goal-B replan：

1. `known-Revisit direct` 先按原协议调用 mixed Image/PointGoal NavDP；它是事实动作，并且是
   唯一允许向 NavDP FIFO 追加当前 observation 的请求；
2. 在完全相同 FIFO 上，使用相同 diffusion seed 调用只读 `imagegoal_resample`，得到 native
   ImageGoal counterfactual；
3. endpoint 前后必须返回逐项相同的 FIFO content fingerprints、
   `memory_mutated=false` 和相同 seed echo，否则整条运行无效；
4. shadow 返回永不替换事实 trajectory，任何统计都不得回写 controller。

不使用 Habitat goal、geodesic、oracle candidate selector、重新采样多 seed 或 learned head。

## 冻结目标 episode

只运行上一消融的 7 条 discordant pair：

- direct gains：`yqstnuAEVhm/episode_0001`、`uNb9QFRL6hY/episode_0000`、
  `ac26ZMwG7aT/episode_0000`、`qoiz87JEwZ2/episode_0000`、
  `i5noydFURQK/episode_0000`、`gZ6f7yhEvPG/episode_0000`；
- direct loss：`pLe4wQe7qrG/episode_0001`。

它们是明确选择过的机制样本，不是统计分母；任何分离度都只能产生下一协议，不能声称泛化。

## 只记录的字段

- memory/native candidate endpoint mean/std、全零比例；
- server-selected endpoint 长度与方向；
- candidate heading resultant、最大分离角、pairwise diversity；
- critic max 和既有 STOP evidence；
- memory/native endpoint ratio；
- memory endpoint / LingBot point-goal distance ratio；
- FIFO 指纹和 seed 收据。

本协议不定义 actionability cutoff，不根据 `pLe` 选择阈值，也不形成在线 arbiter。

## 下一门

只有在 shadow 非干预审计全部通过、`pLe` 的退化可重复且六条 gain 不呈现相同系统性退化
后，才允许另写预注册协议定义 scale-free contract，并在完整 consumed 20-scene pool 跑
`geometry / direct / safe residual` 三臂。否则停止该 actionability 方向，回到 pose/action
接口归因。

