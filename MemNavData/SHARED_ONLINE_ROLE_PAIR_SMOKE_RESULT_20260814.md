# Role-free online-memory paired smoke：闭环结果（2026-08-14）

状态：单个 consumed scene、一个 matched Novel/Revisit pair、三臂、每查询最多 120 steps 的
integration smoke 完成并独立审计通过。**不是 SR、泛化或方法优越性结果。**

## 一句话结论

新的评测接口按预期工作：在相同 frozen online-A 历史后，raw-direct 对 Novel/Revisit 都无条件
接管；certified 对 Novel 完全 abstain 且 120 帧物理 rollout 与 native 逐帧一致，同时对 Revisit
15/15 plans 接受并接管。role 未进入任何 policy 请求，online-A replay 哈希全过且没有 diffusion
采样或 runtime failure。

## 1. 冻结设置

- scene/episode：`gxdoqLR6rwA/episode_0000`（已消费内部场景）；
- query pair：同一 online-A 终点，Novel/Revisit distance error `0.0047 m`、initial path bearing
  error `0.84°`；Novel full-history max covis `0.000`，Revisit eligible max covis `0.750`；
- 三臂：native、raw-DINO direct metric residual、certified scale-free bearing；
- 每查询最多 120 steps、exec horizon 8、deterministic plan seeds；
- 每个 query fresh reset 后 exact online-A replay；MemNav 接收全部 240 帧，NavDP 只接收原始
  30 个 decision frames；replay diffusion samples 为 0；
- CDEC、graph rescue、X-NavDP 和所有 oracle 均关闭。

## 2. 实际结果

| role / arm | success@120 | final distance | takeover plans | certificate accept |
|---|---:|---:|---:|---:|
| Novel / native | 0 | 10.129 m | 0 | 0 |
| Novel / raw-direct | 0 | 3.660 m | 15/15 | n/a |
| Novel / certified | 0 | 10.129 m | 0/15 | 0/15 |
| Revisit / native | 0 | 10.461 m | 0 | 0 |
| Revisit / raw-direct | 0 | 3.698 m | 15/15 | n/a |
| Revisit / certified | 0 | 3.561 m | 15/15 | 15/15 |

120-step budget刻意用于接口 smoke；所有 success 都是 0，因此不能把终距变化写成 SR 提升。

## 3. 因果/安全审计

### Novel

- certified 独立拒绝原因为 `precheck_fundamental_inliers`；
- `router_active=0`、certificate accept `0`、adapter takeover `0`；
- certified 与 native 的完整 executed rollout（online A + query）逐帧相同；
- success、steps、path length、final distance 完全相同；
- 内部长记忆 trace 不要求相同：native 没有 MemNav，而 certified 必须观察历史后才能 abstain。

### Revisit

- certified `15/15` plans 为 `certificate_accepted`，并全部经 fixed-2.5m bearing adapter 接管；
- raw-direct 在 Novel 和 Revisit 上均 `15/15` takeover，证明它没有使用 role 标签，也不是隐藏的
  Revisit-only upper bound；
- certified Revisit rollout 与 native 不同，排除“系统因永远 abstain 而安全”。

### 全局

- 三臂的 scene、episode、pair、query、seed 和 geodesic 完全一致；
- policy runtime fields 不含 `analysis_role`、co-visibility、GT geodesic 或初始 path bearing；
- online-A 所有 RGB 哈希验证通过，replay diffusion samples `0`；
- runtime failure plans `0`；任务结束后两台本机服务自动关闭。

## 4. 它证明与不证明什么

证明到 implementation/causal-smoke 强度：

1. matched role-pair benchmark 能从真实 online-A 历史运行闭环；
2. automatic certificate 在同一接口下能拒绝一个严格 Novel、接受一个强 Revisit；
3. reject 后 exact physical identity 真的成立；
4. raw-direct 是有覆盖、有风险的有效简单对照。

仍未证明：Novel false-takeover rate、Revisit activation recall、任何 SR 增益、scene generalization、
certified 优于 raw-direct。`N=1 pair` 对这些问题没有统计能力。

## 5. 正式扩样前必须补的一项公平对照

当前 raw-direct 使用 legacy metric residual，而 certified 使用 fixed-2.5m scale-free bearing；因此两者
比较同时改变了授权规则、定位算法和 controller input scale。正式 one-shot 之前应增加或至少在
consumed pool 验证一个 `raw-fixed-bearing` arm：raw-DINO top-1 仍无证书、仍对所有 query 接管，
但只保留方向并投影到相同 2.5 m，再调用同一个 frozen mixed controller。这样可以区分：

- fixed bearing 接口本身的效果；
- geometry certificate/abstention 带来的 risk–coverage 效果。

若不补这一臂，certified vs raw-direct 仍可作为“完整方法 vs 强简单 baseline”，但不能写成纯
certificate ablation。

## 6. 产物

- protocol：`MemNavData/SHARED_ONLINE_ROLE_PAIR_PROTOCOL_20260814.md`
- runner：`MemNavData/run_shared_online_role_pair_smoke_local.sh`
- evaluator：`MemNavData/eval_shared_online_role_pairs.py`
- independent auditor：`MemNavData/audit_shared_online_role_pair_smoke.py`
- raw outputs：`.diagnostics/shared_online_role_pair_closed_loop_smoke_20260814/`
- audit SHA256：`eaef4cb49ab02ad9576f67ca004fa2bd3b922dcd66bf78a9203e9930fa79b907`
