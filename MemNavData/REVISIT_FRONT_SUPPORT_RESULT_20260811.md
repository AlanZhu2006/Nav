# Revisit front-support residual：提前停止结果

日期：2026-08-11（CST）  
状态：T0 通过；T1 在预注册安全门首次不可逆失败时提前停止；分支 **rejected**。

## 一句话结论

`forward < 0 -> native` 在 post-hoc `pLe` 上救回一条，但不能泛化。T1 前 4 个完整场景已经
出现 `+0/-4`：它破坏了 4 条 direct 原本成功的 Revisit。因此“NavDP PointGoal 会裁剪后向
forward 分量”是代码事实，却不能推出“后向时应该回退 native”。mixed decoder 仍能利用
lateral 分量和 goal image 完成 U-turn；硬 fallback 恰好删除了最有用的 memory guidance。

按冻结协议要求 `loss=0`，第 4 条 loss 出现后后续场景已不可能通过，任务随即停止，未浪费
剩余数小时，也没有在中途改阈值挽救。

## 1. T0 为什么看起来成功

单场景 `pLe` 的同进程 T0：

| arm | `0000` | `0001` | joint |
|---|---|---|---:|
| direct | fail | fail | 0/2 |
| front-support | fail | success | 1/2 |

关键 `0001` 从终距 `3.407 m` 变为 `0.969 m`，配对 `+1/-0`。这证明 native fallback 可以
切断这一个 harm 的因果链，但没有证明 behind-bearing 是可泛化的 gating feature。

## 2. T1 partial-stop 结果

完成分母：4 scenes、8 episodes；其中共享 Novel A success 为 6 条。

| 指标 | direct | front-support |
|---|---:|---:|
| Joint | 6/8 = 75% | 2/8 = 25% |
| B \| A success | **6/6 = 100%** | 2/6 = 33.3% |
| 配对 | — | **+0/-4** |
| exact McNemar | — | p=0.125（提前停止的小分母） |

四条 direct-only harm：

| scene / episode | direct final | support final | direct steps | support steps |
|---|---:|---:|---:|---:|
| `s8pcmisQ38h/0000` | success，0.980 m | fail，5.787 m | 101 | 257 |
| `rqfALeAoiTq/0000` | success，0.998 m | fail，7.972 m | 115 | 330 |
| `rqfALeAoiTq/0001` | success，0.980 m | fail，4.940 m | 80 | 219 |
| `zsNo4HB9uLZ/0000` | success，0.991 m | fail，3.827 m | 40 | 271 |

support 的 213 个 Goal-B plans 中：

- `131` 次 `pointgoal_behind_navdp_support` native fallback；
- `81` 次进入 mixed；
- `1` 次 PointGoal unavailable；
- 6/8 episode 实际触发 behind fallback。

这不是偶发一条回退，而是规则系统性过度拦截。

## 3. 失败机制

四条被破坏的 direct-success 均展示相同结构：direct 接受 behind memory 后，bearing 随 mixed
执行逐渐进入前半平面；support 改走 native 后，bearing 长期留在后半平面，最终远离目标。

示例：

- `s8/0000` direct：`-161,-142,-109,-85,-57,...`，101 steps 成功；support 前 8 次仍在
  `-161..±180°` 附近，257 steps 后终距 5.787 m。
- `rqf/0000` direct 前六次虽多为平移零候选，但之后 bearing 从约 `-169°` 降到
  `-149,-129,-103,-76,...` 并成功；support 的 native fallback 反而连续约 12 plans 停在
  `-162..-174°`。
- `zs/0000` direct 只用 5 plans 将 `-126°` 降到 `-66°` 并成功；support 逐渐退化到
  `-150°`。

所以：

1. PointGoal forward 裁剪只丢失后向幅值，不会删除 lateral 转向符号；
2. mixed decoder 还同时看到 goal image；
3. 当前执行链虽然不优雅，却在多数 behind-Revisit 上能够形成有效 U-turn；
4. `pLe` 是 mixed U-turn 的特殊失败，不代表后半平面整体是 unsupported domain。

## 4. 其他简单 gate 也未通过离线反证

在旧 direct 的全部 29 条 A-success episode 上，26 条 B success、3 条 B fail。前四个
memory-active plans 的 bearing-error change 并不能分离：

- 成功 `e9/0000`：`-12.4°`（误差反而增加）；
- 成功 `rqf/0000`：`-9.9°`；
- 失败 `pLe/0000`：`+9.2°`；
- 失败 `pLe/0001`：`+9.1°`。

此前 shadow 还已排除单次 endpoint、全段 endpoint mean 和 critic。当前没有证据支持把
behind、短轨迹、早期 bearing slope 或 critic 单独做成 safety expert。

## 5. 冻结结论与下一步

```text
navdp_front_support_v1 = rejected
do_not_resume_remaining_16_scenes
do_not_tune_a_narrower_angle_gate_on_this_pool
```

项目当前仍应保留：

- geometry router：唯一显著、可部署的闭环方法结果；
- known-Revisit direct：`26/29` 的高潜力消融，但有一条真实 native-success harm，尚不能替换
  geometry baseline；
- actionability shadow：机制诊断工具，不是已验证的 arbiter。

若继续处理这一条 harm，必须另立独立协议研究 **显式 rotate-first 执行契约**，因为它保留
memory bearing，而不是像 native fallback 一样删除方向信息。在建立执行语义和成功-control
反证前，不应立即叠加该机制或启动训练。

## 6. 审计产物

- partial-stop 报告：`.diagnostics/revisit_front_support_full_20260811/partial_stop_report.json`
- 报告 SHA：`.diagnostics/revisit_front_support_full_20260811/partial_stop_report.json.sha256`
- 冻结源码包：`.diagnostics/revisit_front_support_full_20260811/source_bundle.partial_stop.tar`
- 源码包 SHA：`.diagnostics/revisit_front_support_full_20260811/source_bundle.partial_stop.tar.sha256`
- T0 报告：`.diagnostics/revisit_front_support_t0_20260811/report.json`

两个实验服务均已退出，GPU 已释放。
