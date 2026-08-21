# Revisit actionability shadow 结果

日期：2026-08-11（CST）  
状态：7 条已选择 discordant episode 的机制探针完成；**不授权阈值、在线 arbiter、blind 或论文性能声明**。

## 一句话结论

唯一 direct loss `pLe/0001` 确实出现了与 6 条 direct gain 不同的、持续四次的
**平移输出抑制**。但进一步读代码后，不能把它直接叫作“NavDP 没有动作能力”：冻结 NavDP
会把后向 PointGoal 的 forward 分量截为 0，并把短候选的 `x,y` 清零；当前 Habitat
pure-pursuit 又完全忽略轨迹第 3 维。当前最强归因因此是
**memory pose 超出 PointGoal controller 的前半平面支持域，并在执行接口处丢失转向语义**。

这否定了“马上训练第二个 actionability expert”的必要性。下一步应先测试一个由代码契约直接
推出的零参数 residual：后向 PointGoal 逐计划回退 native，进入前半平面后再交给 mixed
NavDP。

## 1. 探针与非干预审计

每个 Goal-B replan 都先产生事实 memory-conditioned mixed-NavDP rollout，再用相同 FIFO、
相同 diffusion seed 只读重采样 native ImageGoal rollout。shadow 从不控制动作，也不追加
observation。

| 审计项 | 结果 |
|---|---:|
| 目标 episode | 7 |
| 事实 replan | 118 |
| 可配对 shadow replan | 113 |
| 明确不可配对 | 5，均为 `memory_pointgoal_unavailable` |
| FIFO content fingerprint | 113/113 一致 |
| diffusion seed | 113/113 一致 |
| 事实轨迹与旧 direct 参考 | 7/7 完全一致 |

“事实轨迹完全一致”包括 episode seed、Goal-A trace hash、A/B 成败、A/B 步数、路径长度、终点
距离和终止原因；因此 shadow 没有改变闭环结果。

## 2. 描述性结果

窗口统一定义为每条 episode 的前 4 个 **memory-active** replan；`i5/gZ` 开头分别有 2/3
个无 PointGoal 的 native replan，因此不是 episode 的绝对前四步。

| scene | 旧配对结果 | paired/total plans | memory/native endpoint ratio | memory 全平移零 | endpoint/PointGoal 序列 | 全段 endpoint ratio |
|---|---|---:|---:|---:|---|---:|
| `pLe` | direct loss | 33/33 | **0.095** | **2/4** | **0.000, 0.127, 0.000, 0.092** | 1.027 |
| `yq` | direct gain | 9/9 | 1.065 | 0/4 | 0.807, 0.969, 0.941, 1.000 | 0.789 |
| `uNb` | direct gain | 6/6 | 0.491 | 0/4 | 1.074, 1.073, 1.035, 1.050 | 0.400 |
| `ac26` | direct gain | 36/36 | 0.930 | 0/4 | 0.396, 0.755, 0.883, 0.879 | 0.894 |
| `qoiz` | direct gain | 6/6 | 0.414 | 0/4 | 0.052, 0.480, 1.022, 1.053 | 0.440 |
| `i5` | direct gain | 12/14 | 0.430 | 1/4 | 0.000, 0.022, 0.638, 1.031 | 0.670 |
| `gZ` | direct gain | 11/14 | 0.761 | 0/4 | 0.231, 0.889, 1.124, 1.060 | 0.762 |

只看“第一次短轨迹”不能分离成功与失败：`qoiz/i5` 也会短暂抑制，但随后恢复。`pLe` 的差异
是四次连续不恢复；它的前四次 memory PointGoal 方位约为
`+175°, +160°, +176°, -166°`，始终处于后半平面。六条 gain 的方位则随执行逐步进入前半
平面。全 episode 平均同样无辨识力：`pLe` 的全段 endpoint ratio 回到 `1.027`。

因此本探针只支持“存在早期、持续的平移抑制事件”，不支持任何单次 endpoint cutoff，也不
支持用全段平均作为 gate。

## 3. 代码级反证：这还不是 action failure

三个现有代码事实改变了归因：

1. `NavDP/baselines/navdp/policy_agent.py::process_pointgoal` 执行
   `clip_goals[:,0] = clip(..., 0, 10)`；后向 PointGoal 的 forward 信息在进入网络前被删除。
2. `NavDP/baselines/navdp/policy_network.py` 对 endpoint 平移 `<0.5 m` 的候选执行
   `[x,y,theta] *= [0,0,1]`。所以本报告中的“全平移零”不等于三维 trajectory 全零，更不
   能直接解释成网络 abstain。
3. `MemNavData/eval_2leg_habitat.py::waypoints_to_world` 只使用 `x,y`，当前 pure-pursuit
   不消费第 3 维。对全平移零路径，它也没有显式的 rotate-in-place 契约。

`pLe` 的可检验因果链因此是：

```text
正确附近的 memory anchor
  -> 后向 relative PointGoal
  -> forward 分量被 NavDP 输入预处理截断
  -> mixed decoder 连续输出短/转向型候选
  -> 当前执行器只消费 x,y，未兑现转向语义
  -> memory bearing 长期停留在 ±180° 附近
  -> native success 被 direct memory 接管破坏
```

这是强机制假设，但最后一箭仍需闭环反事实确认；不能仅凭 1 条 loss 写成结论。

## 4. 对“残差 expert”的修正

当前不需要新增一个 learned action expert。更小、更优雅的分解是：

```text
known Revisit -> temporal DINO pose proposal
                  |
                  +-- PointGoal 在 NavDP 支持域（forward >= 0）
                  |      -> frozen mixed ImageGoal+PointGoal NavDP
                  |
                  +-- PointGoal 在支持域外（forward < 0）
                         -> 当前 replan 使用 frozen native ImageGoal NavDP
                         -> 下一 replan 重新判断，不永久锁存
```

RANSAC 仍是高精度 place-support evidence：pass 可以增加可信度，reject/insufficient 仍然不是
task negative。rollout shadow 的角色只应是诊断 controller/interface 是否兑现 proposal，
而不是在这 7 条上选阈值后成为第三个模型。

## 5. 下一实验门

1. 实现 `navdp_front_support_v1`：只增加 `forward < 0 -> native` 的逐计划 fail-closed
   条件；不训练、不选数值阈值、不改 native/mixed NavDP 权重与 FIFO。
2. 先复跑 `pLe + 6 gains` 做传输检查；通过后直接跑完整 consumed 20 scenes / 40 episodes，
   比较 `geometry / direct / front-support residual`。
3. 通过条件仍是配对 `gain > 0, loss = 0`，并单列 conditional B、joint、SPL 和 scene-cluster
   CI；7 条机制样本不作为统计分母。
4. 若 `pLe` 仍被破坏，才实现显式 rotate-first executor，并以 trajectory 第 3 维/PointGoal
   bearing 的独立契约验证；若 support residual 已消除回退，则不增加该复杂度。

## 6. 产物

- 自动报告：`.diagnostics/revisit_action_shadow_20260811/report.json`
- 报告哈希：`.diagnostics/revisit_action_shadow_20260811/report.json.sha256`
- 冻结协议：`MemNavData/REVISIT_ACTIONABILITY_SHADOW_PROTOCOL_20260811.md`
- shadow 诊断：`MemNavData/revisit_action_shadow.py`
- 汇总器：`MemNavData/summarize_revisit_action_shadow.py`

边界：该集合来自 consumed pool 的 post-hoc discordant pair；本结果可以生成下一冻结协议，
不能独立形成泛化或性能声明。

## 7. 后续 T0 闭环反事实（已完成）

`navdp_front_support_v1` 已在同机、同一对长期存活 server process 上复跑 `pLe` 两条：

| arm | `0000` | `0001` | joint |
|---|---:|---:|---:|
| direct | fail，终距 1.700 m | fail，终距 3.407 m | 0/2 |
| front-support | fail，终距 1.624 m | **success，终距 0.969 m** | **1/2** |

配对为 `+1/-0`。support 两条共执行 11 次后向 native fallback，并分别在 step 48/40 重新
进入 mixed controller；关键 `0001` 用 132 steps 成功，几乎复现 trace-source geometry/native
的 131 steps、终距 0.985 m。这验证了支持域规则确实能切断本轮唯一已知 harm 的因果链。

但 T0 仍是 post-hoc 的 1 scene / 2 episodes，不是性能证据。正式下一门是使用完全冻结代码
运行 20 scenes / 40 episodes 的 direct vs front-support 配对；在完整分母出来前，不把
`+1/-0` 写成方法提升。

T0 自动报告：`.diagnostics/revisit_front_support_t0_20260811/report.json`；因协议措辞在 T0 后
修正为准确记录 geometry trace source，T0 实际执行时的全部源码已封存于
`.diagnostics/revisit_front_support_t0_20260811/source_bundle.tar`，其 SHA256 单独保存。

## 8. T1 已推翻 front-support 泛化

正式 T1 前 4 个完整场景已经得到 `+0/-4`：direct 在 6 条 A-success 上 B 为 6/6，
front-support 只有 2/6。按 `loss=0` 安全门提前停止，其余 16 scenes 不再运行。

这说明 T0 的 `pLe` 救援是 case-specific；behind mixed 本身通常正是完成 U-turn 所需的 memory
guidance。完整负结果与逐例归因见 `REVISIT_FRONT_SUPPORT_RESULT_20260811.md`。因此本文件第
5 节提出的 front-support 下一门已经关闭，不应恢复或在 consumed pool 上缩窄角度阈值。
