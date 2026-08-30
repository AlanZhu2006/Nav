# HM3D Table-1 controller portability formal result（2026-08-29）

## 1. 冻结问题与 population

在同一份 fresh-query、scene-overlapping HM3D population 上，分别估计 CEC 对两个
冻结 ImageGoal controller 的组内 paired effect：

- 28 actual-mono causal histories；
- 21 scene clusters；
- 每条一个 unsupported Novel 和一个 supported Revisit，共 56 queries；
- runtime 不读取 role；
- 同一 controller 内 native/CEC 共享 query、history、checkpoint、seed、600-step
  budget 与 1 m success radius；
- NavDP 与 ViNT 的绝对 SR 不是 paired controller-superiority estimand。

协议：`hm3d_table1_controller_portability_protocol_20260829.json`。

## 2. 正式四行

| Controller | Arm | Novel | Revisit | Overall |
|---|---|---:|---:|---:|
| NavDP | native | 6/28 | 8/28 | 14/56 |
| NavDP | +CEC | 6/28 | 25/28 | 31/56 |
| ViNT | native | 3/28 | 3/28 | 6/56 |
| ViNT | +CEC | 3/28 | 19/28 | 22/56 |

### NavDP primary paired result

- Revisit：`8/28 -> 25/28`，paired `+18/-1`，risk difference
  `+60.71 pp`，two-sided exact McNemar `p=7.629e-5`，scene-cluster bootstrap
  95% CI `[+35.48,+85.19] pp`；
- Novel：`6/28 -> 6/28`，paired `+0/-0`，0 takeover，所有 28 条 exact native；
- Overall：`14/56 -> 31/56`，paired `+18/-1`，risk difference
  `+30.36 pp`，`p=7.629e-5`，scene-cluster CI `[+17.74,+42.59] pp`。

NavDP 次要量：

| Role | Native SPL | CEC SPL | Native mean path | CEC mean path | Native mean steps | CEC mean steps |
|---|---:|---:|---:|---:|---:|---:|
| Novel | .1052 | .1052 | 12.83 m | 12.83 m | 395.5 | 395.5 |
| Revisit | .1251 | .7138 | 11.14 m | 4.06 m | 340.7 | 130.9 |
| Overall | .1151 | .4095 | 11.98 m | 8.45 m | 368.1 | 263.2 |

### ViNT paired result（先前独立封存）

- Revisit：`3/28 -> 19/28`，paired `+16/-0`，`p=3.052e-5`；
- Novel：`3/28 -> 3/28`，0 takeover、exact fallback；
- Overall：`6/56 -> 22/56`。

这些数字证明同一 proposal--witness--authority interface 可以被两个不同的冻结
ImageGoal controller 消费；它们不证明 NavDP 优于 ViNT，或反之。

## 3. NavDP 安全与运行审计

- CEC takeover：27/28 Revisit，0/28 Novel；
- fully rejected exact-native：28 Novel + 1 Revisit；
- runtime failure plans：0；
- metric-depth sensor reads：0；
- raw query rows：每臂 56；
- raw metric rows：112；
- raw monocular plan receipts：4,462；
- arm order：`native/CEC` 14 histories，`CEC/native` 14 histories；
- success 由 independent verifier 从 raw final distance 重新计算。

因此旧 repair 链的 `14/56 vs 14/56` 与 2,657 个 endpoint runtime failures 仍只属于
fail-closed infrastructure incident，不能与本结果混用。本次修复后所有 runtime
failures 为 0，并重跑了完整双臂。

## 4. Provenance

- NavDP run root：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_controller_portability_20260829/navdp_server_namespace_repair_20260829T054615Z_51ee9a4c`；
- smoke / formal / aggregate / verifier / seal：
  `16544384 / 16544385 / 16544386 / 16544387 / 16544388`；
- NavDP summary SHA-256：
  `e0d1247d2b589fbf99ccb24fa7044c9539126e4f73a4b37432107b7be4b76dae`；
- NavDP independent verifier SHA-256：
  `0e8bf294908d5e8a89400043eee35f4c58bc66b33fb3c5982a4932f2e6a4f33f`；
- ViNT independent verifier SHA-256：
  `89a5e85f7899e283ce8924fd1777e98b061ceb6577641cb6811b4c5f6b02bf32`；
- joint controller receipt SHA-256：
  `de0d755fc8498ab799e742ad761385b48e4a1c4acbadc98834fed7d352b72c18`。

Joint receipt 明确记录：`cross_controller_absolute_sr_is_not_a_paired_estimand=true`。
