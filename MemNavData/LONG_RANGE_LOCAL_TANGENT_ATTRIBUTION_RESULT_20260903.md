# 长程 Revisit：Local-Tangent 闭环机制归因结果

日期：2026-09-03（Asia/Shanghai）  
性质：**已消费单失败机制实验；不具备论文结果资格。**

## 问题与冻结设计

Canonical CEC 已能在该 episode 的每个决策点认证同一历史目标，但固定 2.5 m 的
endpoint/path chord 仍然失败。为区分路线表征与 frozen NavDP conditioning，结果揭封前固定
三个同机同进程臂：

1. `oracle_chord_mixed_realized`：当前 Habitat shortest path 上 2.5 m lookahead chord；
2. `oracle_tangent_mixed_realized`：第一个平面 baseline 至少 0.30 m 的可行局部切向；
3. `oracle_tangent_then_native_realized`：同一切向仅在 heading residual 大于 20° 时注入，
   对齐后用同 observation、seed 与 FIFO 的 native read-only resample。

三臂都先完成真实的 DINO proposal、LightGlue/PnP witness 与 strict certificate；只替换
certificate 之后的 privileged direction readout。第三臂只用于诊断持续 mixed conditioning，
20° 不是候选方法阈值。

身份与完整性：

- history index 32，`5jp3fCRSRjc/episode_table3_survey_080`；
- 初始 geodesic `33.791 m`；起点/目标高度差 `3.200 m`；
- protocol SHA-256：
  `3e48de6deeff53da7ed4e7732efbee4ecf3e02691480e35021ea2e29abc4a3ba`；
- pre-unseal analyzer SHA-256：
  `5d0970230bc9c37d80c070fd74e643e50828346fcdb1fdc2bfc6228127467991`；
- Slurm `16822227_32`，`COMPLETED`，`00:37:12`，exit `0:0`；
- completion SHA-256：
  `2a147b8c1d82a55e0881ff9b4a5a718a82eff950238f3de5f9e33b071f4e2cc1`；
- frozen-analysis output SHA-256：
  `cf5c47944fca307df99008d6a8584335a7005f8a6c656e1f6f608a206a87e9fd`。

## 结果

| arm | success | final 3-D / planar / vertical | min geodesic | realized path | critic < -0.5 | stationary |
|---|---:|---:|---:|---:|---:|---:|
| 2.5 m chord + mixed | 0 | 12.773 / 12.366 / 3.200 m | 25.157 m | 25.720 m | 231/296 = 78.04% | 25/2365 = 1.06% |
| first tangent + mixed | 1 | 0.990 / 0.990 / 0.000 m | 1.166 m | 33.089 m | 76/129 = 58.91% | 0/1030 |
| first tangent then native | 1 | 0.977 / 0.977 / 0.000 m | 1.164 m | 35.142 m | 84/142 = 59.15% | 0/1132 |

continuous tangent 相对 chord：

- final 3-D distance `-11.784 m`；
- minimum planned geodesic `-23.991 m`；
- critic 低分比例 `-19.13 pp`；
- stationary transition 比例 `-1.06 pp`；
- 从 `y=3.209 m` 到达 `y=0.009 m`，不是跨楼板的平面假成功。

selective 相对 continuous tangent：

- final 3-D distance只差 `-0.013 m`；
- minimum geodesic只差 `-0.002 m`；
- critic 低分比例为 `+0.24 pp`；
- 两者都无 stationary transition。

## 结论边界

这条受控反事实支持的因果解释是：在该失败中，2.5 m arc-ahead chord 抹掉了拐角处的
第一个可行方向；真正的 local tangent 可以被同一个 frozen mixed NavDP 执行，并完成跨层
路线。它不支持“增大 PointGoal 半径”，也不支持“对齐后切回 native”或 20° 门控。

它尚未证明：

- 单目历史可以在未读 episode 上稳定恢复同样的 tangent；
- route-tangent 在总体 SR 上显著超过 canonical endpoint-bearing CEC；
- 30--50 m same-floor 泛化已经成立；用正确的 online-A endpoint 到 Revisit-goal 高差重算后，
  现有未消费池在这一 stratum 为 0 条，不能做诚实确认。

因此唯一被授权的下一步是：实现一个连续、无距离分支、无 stuck trigger、无 endpoint/native
fallback 的 causal monocular route-tangent readout，并在预冻结的未消费 20--30 m same-floor
pool 上做 paired closed-loop confirmation。初始 certificate reject 的 exact native 行为仍属于
CEC 的开放集合约，不是 route-tangent 内部 fallback。

本地完整分析：

```text
.diagnostics/longrange_local_tangent_attribution_20260903/analysis.json
```
