# HM3D actual-online Full-Mono mixed-role 正式结果

日期：2026-08-20（Asia/Shanghai）  
状态：**完整完成；independent verifier 通过**

## 1. 冻结问题

这项实验检验：当历史生成、历史重放和后续 Novel/Revisit 查询都只使用同一条因果 RGB
流及冻结的单目 depth sidecar 时，CEC 是否仍能给冻结 NavDP 带来 Revisit utility，并在
没有 role 标签的情况下对 unsupported Novel 精确回退。

它是 full-monocular sensor/control integration 实验，不是 fresh-scene generalization：九个
HM3D source scenes 在此前 metric-controller HM3D 实验中已经出现过。协议冻结于读取本次
Goal-A 和 query outcome 之前；CEC、raw memory、NavDP、阈值、seed、step budget 和成功
判据均未按本次结果调整。

冻结协议：

- `MemNavData/HM3D_FULLMONO_MIXED_ROLE_PROTOCOL_20260820.md`
- `MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json`

## 2. 远端工件与身份

Run root：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_mixed_role_20260820/formal_20260819T182657Z_0e587874
```

Task bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_mixed_role_0e587874d5b89531
```

关键 SHA-256：

| 工件 | SHA-256 |
|---|---|
| task receipt | `0e587874d5b8953176a49a77345f6bde8733c2d31930ab3547d8c238a857b372` |
| sealed parent manifest | `62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89` |
| constructed benchmark manifest | `b70d53557d5fb656badf1311d19b3f643583a1ab5fad782b542b66e4647c5948` |
| population receipt | `7a2acbbf86d96f83f4cbda6ec61b5b3a6f84f07d787bfbcec40927acdd1f974b` |
| final summary | `971cabd0f9c64f86ec79f4ab1b9ad2b823a796f23b7056663c32aa4af840f007` |
| independent verification | `312d7b1268c3b2d2cf01ea4795593786e644fc025e261cf4cd683aee6a68cf38` |

提交链：Goal-A `16032636` -> construction `16032656` -> population seal
`16032668` -> query smoke `16032673` -> formal query `16032690` -> summary
`16032711` -> independent verification `16032721`。

## 3. Population 与可观测性

- frozen source Goal-A：36 条，9 scenes；
- actual mono Goal-A 成功：24/36；
- materialized actual-online histories：24；
- 同时满足 Natural Novel 与 standard Revisit 构造合约：8 histories，7 scenes；
- 每个 history 生成 1 个 Novel 和 1 个 Revisit 查询；
- 每个 arm 共 16 条查询；
- runtime 不读取 Novel/Revisit role；
- 所有 arm 独立 reset，并重放同一条 actual mono Goal-A RGB history。

构造支持范围：Novel max online-A co-visibility 为 `0--0.0432`；Revisit 为
`0.7177--0.7656`。冻结目标原为 18 histories / 8 scenes，因此本次构造分母明确
underpowered。

## 4. 闭环结果

| arm | Novel | Revisit | role-balanced total |
|---|---:|---:|---:|
| mono native | 2/8 | 0/8 | 2/16 = 12.50% |
| mono raw fixed | 3/8 | 8/8 | 11/16 = 68.75% |
| mono CEC | 2/8 | 7/8 | 9/16 = 56.25% |

CEC 对 mono native：

- 全部：`+7/-0`，risk difference `+43.75 pp`，exact McNemar
  `p=0.015625`，scene-cluster 95% CI `[+35,+50] pp`；
- Revisit：`+7/-0`，`+87.5 pp`，`p=0.015625`，cluster CI
  `[+70,+100] pp`；
- Novel：`+0/-0`，轨迹与 native 相同。

CEC 对 mono raw fixed：

- 全部：`+1/-3`，`-12.5 pp`，`p=0.625`，cluster CI
  `[-28.57,0] pp`；
- Revisit：`+0/-1`，`-12.5 pp`，`p=1.0`；
- Novel：`+1/-2`，`p=1.0`。

因此，这一小样本不支持“CEC 比 always-on raw memory 的 SR 更高”。Raw 在 8 条
Revisit 上达到 8/8，而 CEC 为 7/8；差异不显著，但方向也不能被隐藏。

## 5. Authorization、fallback 与单目审计

- CEC accept：Revisit `8/8`，Novel `0/8`；
- Novel memory takeover：`0/8`；
- Novel fully rejected exact-native：`8/8`；
- CEC runtime failure plans：0；
- Goal-A simulator metric-depth reads：0；
- query simulator metric-depth reads：0；
- independent verifier 检查 Goal-A 单目 receipts：1,231 个；
- independent verifier 检查 query 单目 receipts：1,809 个；
- verifier 从 48 条 raw final-distance records 独立复算所有成功数和 paired contrast，
  `verified=true`。

## 6. 唯一准确的结论

本结果建立了以下新事实：

1. actual mono Goal-A rollout 可以产生可用的因果 RGB history；
2. 在 history、replay 和 query controller 均禁止 simulator metric depth 时，CEC 仍能
   带来显著 Revisit utility；
3. 在本次 8 个 Natural Novel 上，certificate 全部 abstain，并逐动作复现 mono native；
4. “one causal RGB stream, two time scales, one frozen policy”已经从模块实验升级为完整
   闭环 integration result。

本结果没有建立：

- fresh HM3D scene generalization；
- mono controller 相对 metric RGB-D 的 non-inferiority；
- CEC 相对 raw fixed 的成功率优势；
- certificate 的形式安全保证或零误激活保证；
- Novel ImageGoal 能力被解决。

对论文最重要的边界是：CEC 的已证价值是 **proof-before-control、open-set
authorization 和 exact fallback**；不是在每个高支持 Revisit 上必然优于 raw retrieval。
下一项正式实验必须使用此前完全未读 outcome 的 HM3D scenes，扩大 mixed-role 分母，
同时估计 Revisit utility、Novel interference 和 CEC-vs-raw 风险—覆盖关系。
