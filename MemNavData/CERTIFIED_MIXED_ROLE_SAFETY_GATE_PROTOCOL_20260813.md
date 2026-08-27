# Certified mixed-role safety gate（冻结协议，2026-08-13）

## 1. 为什么还需要这一步

现有正式 `fresh160` 已证明：在**已知目标是 Revisit** 的条件下，certified residual 的
`B|A=112/120`，高于 native `27/120`、旧 geometry `91/120`，但相对 raw-DINO direct
仅为 `+9/-3, p=0.146`。现有 `fresh20` double-Revisit 又证明系统能保留并调用两个在线
记忆，但该数据的 37 个独立 certificate 查询全部是正例，不能测 no-match specificity。

strict-v4 的 gTV 单例已经展示 `A/B/C` 自动激活为 `0/0/11`，且 Novel A/B 与同进程
reference 的 rollout 和 memory trace 完全相同；不过 `N=1` 只能作为接口集成证据。

因此当前最小缺口不是再重复五个 SR arm，而是回答一个更窄、可证伪的问题：

> 同一套 role-free certified 系统在自然的 `initial ImageGoal -> Novel -> Revisit` 流中，
> 能否在 Novel A/B fail closed，并在真正 Revisit C 上接管？

## 2. 数据边界

- 仅使用已消费的四条 strict-v4 smoke：
  `e9zR4mvMWw7`、`gxdoqLR6rwA`、`dhjEzFoUFzH`、`gTV8FGcVJC9`，各
  `episode_0000`。
- 四条均满足 `multileg_v4_role_paired_20260812`：
  `initial_imagegoal -> novel -> revisit`。
- 这是实现/因果安全门，不是 SR 或 specificity 估计。
- 不读取 `blind16`。scene-role receipt 已表明当前剩余未消费 3-leg 场景全部属于 blind；
  在没有一次性确认授权前，实验到此边界停止。

## 3. 两臂与固定控制变量

两个 arm 必须在同一对已加载 server、同一 scene/episode/seed 中成对运行，arm 顺序按场景
平衡；NavDP 使用 deterministic plan seeds、server selector、每腿 600 steps、每次执行 8 帧。

1. `known_c_reference`
   - `hybrid_pose + phase`；
   - A/B 为 native NavDP，但每个决策帧仍写入相同的因果长记忆；
   - 仅 benchmark 已知的 Revisit C 使用 `navdp_mix + legacy_metric`。
2. `certified`
   - `hybrid_pose + certified_relocalization`；
   - A/B/C 均为 `navdp_auto`，不得读取 causal-role 标签；
   - certificate accept 才允许 `verified_bearing_v1 + fixed 2.5 m + navdp_mixed`；
   - reject 必须在当步调用相同 native ImageGoal controller。

reference 不是论文方法，也不用于声称 known-role 优势；它只提供检查 exact fallback 所需的
同进程原生 A/B 反事实。

## 4. 预注册通过条件

### 4.1 Novel 安全（primary）

对所有实际执行的 A/B：

- `router_active_plans_A = router_active_plans_B = 0`；
- 不得出现 `certified_relocalization_accepted=True`；
- 不得出现 `revisit_adapter_takeover=True`；
- certified 与 reference 的 rollout trace、memory trace 逐项相同；
- success、steps、path length、final distance 逐项相同；
- 不得出现 certificate endpoint/runtime failure。

这里要求的是**物理反事实等价**，不是仅比较最终 success bit。

### 4.2 Revisit 正控（secondary）

在至少一条 reference 已成功完成 A/B、因而 C 可评的 episode 上：

- certified C 至少有一次独立 uncached certificate；
- certificate 必须 accept，并产生 `revisit_adapter_takeover=True`；
- 至少一条这样的 positive-control episode 完成 C。

### 4.3 审计完整性

- 两臂的 scene、episode、seed、checkpoint、代码和输入 receipt 一致；
- 数据契约通过；
- server/评测进程正常退出；
- 输出中不含 NaN 或缺失的关键字段。

## 5. 决策规则

- 任一 Novel false takeover、prefix 不一致或 runtime failure：安全门失败，停止扩样并归因。
- 四条全过：只能写成“strict-v4 mixed-role implementation/safety smoke passed”；不能报告
  specificity、SR 增益或论文确认。
- 之后若需要统计结论，应先在**非 blind、scene-disjoint** 的自然 mixed-role 流扩样；当前
  inventory 没有这样的保留池，所以不得自动升级到 blind/HPC 正式确认。

