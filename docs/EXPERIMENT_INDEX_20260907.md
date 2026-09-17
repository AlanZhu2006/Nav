# 实验与证据索引（2026-09-07）

数字来自已有独立复算及冻结报告；本次 Git 整理本身没有重新运行这些 benchmark。
每行各有 population，不能相互替换分母或把重复查询当作新的独立样本。

## 1. 当前论文主体

| 对应部分 | 已完成结果 | 原始来源/详细报告 | 边界 |
|---|---|---|---|
| Table I：HM3D/NavDP | Revisit 8/28→25/28，+18/−1，p=7.63e−5；Novel 6/28不变 | [HM3D controller paired](../MemNavData/HM3D_TABLE1_CONTROLLER_PORTABILITY_RESULT_20260829.md) | 同一历史下的控制器内配对 |
| Table I：HM3D/ViNT | Revisit 3/28→19/28，+16/−0，p=3.05e−5；Novel 3/28不变 | 同上 | ViNT 接收认证 anchor 图，不与 NavDP 做绝对排名 |
| Table I：MP3D/NavDP | Revisit 9/42→37/42，+29/−1，p=5.77e−8；Novel 17/42不变 | [MP3D controller paired](../MemNavData/MP3D_TABLE1_CONTROLLER_PORTABILITY_RESULT_20260829.md) | 42 histories /25 scenes |
| Table I：MP3D/ViNT | Revisit 2/42→24/42，+22/−0，p=4.77e−7；Novel 9/42不变 | 同上 | 共享 NavDP 产生的 A history，不是 ViNT 自主 A |
| 完整单目 | Native 17/56→CEC 32/56，+16/−1，p=0.000274658；raw 28/56 | [完整审计 §4.2](../MemNavData/OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md) | actual mono A + mono query；不同于 Table-I HM3D population |
| Table II：连续第三目标 | Revisit-C 8/20→17/20，+10/−1，p=0.01171875；Novel-C 4/20不变 | [主账 §4.3](../MemNavData/OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md)、[SPL 修正](../MemNavData/PAPER_SPL_CORRECTION_20260906.md) | conditional on A+B；不是三段 joint 17/20 |
| Table III：深度消融 | Metric/zero/mono native 11/4/10，metric/mono CEC 26/28，均分母42 | [保留 SPL](../MemNavData/PAPER_SPL_RETAINED_20260907.md) | 21 histories /10 scenes，共享 metric A；目前保留 SPL bounds |
| Table IV：授权归因 | Final14 mono raw/finite-PnP/CEC 23/25/28，分母42；Revisit均20/21 | [主账 §4.4](../MemNavData/OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md) | 对 raw +5/−0，p=.0625；对 witness +4/−1，p=.375 |
| 新 HM3D 授权消融 | Native/raw/witness/CEC 15/35/30/32，分母56 | [四臂完整结果](../MemNavData/HM3D_AUTHORITY_FOUR_ARM_RESULT_20260906.md) | Retrospective；CEC vs raw +5/−8，p=.581；不能称 strict 总 SR 最高 |

Novel outcome 相同不自动意味着每条轨迹相同；exact-native 主张需具体看拒绝和执行 trace。
本轮不把 raw 的误授权等同于必然导航失败。

## 2. 正在补齐的计分实验

[Table III 精确 SPL 重跑](../MemNavData/FINAL14_TABLE3_EXACT_SPL_REPLAY_PROTOCOL_20260907.md)：
原 21 histories、42 queries、五臂 210 rollouts。补存完整终点、积分实际执行路径，
保留旧指令计数作对照。不是新增训练，不改变控制方法，不重新采集 A。

提交 Job ID：[17057430/31/32/33 收据](../MemNavData/FINAL14_TABLE3_EXACT_SPL_REPLAY_SUBMISSION_20260907.json)。
只有完整原始重算通过后才更新该表的新 SR+SPL。

## 3. 长程研究：保留负结果，不升格为主方法

| 实验 | 观察 | 能说明/不能说明 |
|---|---|---|
| 三桶长度，48 controlled histories | Revisit native→CEC：2/16→4/16、1/16→2/16、2/16→0/16 | 不是 actual-online A；距离和跨楼层结构存在混杂 |
| Same-floor 20–30 m，23 histories /8 scenes | Native 0/23、endpoint 0/23、route tangent 4/23；p=.125 | 新历史、复用场景；未满足预注册确认门 |
| Rear alignment / U-turn，9 consumed histories | 3/9→4/9，+2/−1，p=1；4条 geometry stop | 没有通用解决长程漂移的证据 |
| 短中程 metric-distance，28 histories | Fixed 25/28、metric 24/28，+1/−2，p=1 | 没证明尺度距离一定差；也不能外推至30 m |
| 长程 oracle 四臂，16 histories | 正式批次未完成，0个完整四臂块 | 失败日志有 quota error；不能宣称所有失败是 NavDP 上限或 LingBot 漂移 |

详细来源：[总审计 §6](../MemNavData/PROJECT_REAUDIT_20260906.md)、
[same-floor result](../MemNavData/HM3D_LONGRANGE_ROUTE_TANGENT_FORMAL_RESULT_20260903.md)、
[最新未完成归因状态 §7](../MemNavData/OVERNIGHT_PROJECT_AUDIT_AND_REFACTOR_20260906.md)。
单独的旧状态文档保留原时间快照，不代表任务仍在队列中。
已退役的SE2/odometry-proxy分支保留历史源码，不是可部署单目方法；其旧协议sidecar
与后来的撤回说明不同，见[归档哈希说明](../MemNavData/GIT_MAIN_RELEASE_VALIDATION_20260907.md)。

## 4. 学习路线与已有机制

- [CDEC](../MemNavData/CDEC_LEARNED_EPISODIC_DIRECTION_RESULT_20260813.md)：排序的离线改善没有兑现为更强的 certified-actionable 结果。
- [Pi3X 连续状态闭环](../MemNavData/PI3X_SHARED_STATE_CLOSEDLOOP_RESULT_20260906.md)：四个已消费 query 的三臂均4/4；轨迹改善集中在少量场景，不能称新 SR 增益。
- [固定 anchor 关系学习 V0](../MemNavData/ANCHOR_RELATION_LEARNING_RESULT_20260906.md)：103/20 pairs、32/8 scenes；验证平均位置误差1.3054 m；未达到几何参考。
- [几何条件关系学习](../MemNavData/ANCHOR_RELATION_GEOMETRY_PROGRESS_20260906.md)：历史 RGB/XYZ 提取分支；9月7日复核自动链停在 `verify_overfit8` 的数值一致性断言，不能把后续32/8三种子验证写成已完成。
- Novel oracle bearing 28/40→40/40 是特权上界；Active-glance 和 X-NavDP 对照没有产生可替代当前主线的新方法结果。早期里程碑见 [旧完整总账](../MemNavData/STATUS_20260825_GIT_RELEASE.md)。

## 5. 尚未成立的外推

完整自主 5-leg 确认、公开 GOAT 完整跑分、正式真机闭环、通用长距离到达和 learned
替代 certificate 仍不是已完成证据。本轮不启动这些工作，不用工程测试通过代替导航 SR。
