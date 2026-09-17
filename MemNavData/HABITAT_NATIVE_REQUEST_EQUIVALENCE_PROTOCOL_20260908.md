# 最小修复的非接管分支收尾检查

2026-09-08。工程接口验证，非新 SR 实验。已在 `front_goal_bridge_v1` 完整核验后执行，
4/4 短测完成，独立核验与全部视频导出通过。下方保留执行前冻结的设计。

后向适配用于已经发出的 PointGoal，不能在 CEC 不接管时改变原生 ImageGoal 请求。
此前 CPU 测试覆盖了该条件；还需在真实服务上验证一次完整输入/历史链。

## 固定短测

- 使用既有四历史清单的前两条（gxdoq、pLe），取各自已经冻结的 Natural Novel 查询。
- 每历史两臂：native sidecar / CEC；均为 corrected RGB、source-raster mono、bounded standard。
- 共用同一低层前向域适配，开启但不得凭空生成 PointGoal；所有模型收到的 odometry 字段均移除。
- 模型不读 Novel/Revisit 标签；标签只由 evaluator 用来选择原有 query。
- 每臂 64 动作预算；转向预算没有额外增加。不会重构数据、选新 query 或改阈值。
- 不加 raw：本短测只验证 CEC 不接管时的执行等价性，不做方法 SR 排名。

## 判定

1. 先报告 CEC 实际是否接受，不能为了通过测试强制拒绝。
2. 若全过程未接管，要求与同机同服务 native 的每个规划输入、输出、动作和位姿一致；
   对共享 A、depth、FIFO、frame index 和最后落点同时对账。
3. 若发生接管，保留真实收据：这是证书接受事件，不能伪称完成了 exact-native 验证，
   也不自动归咎于执行接口。短测不单凭初始构造角色断言所有未来支持均为误识别。
4. 保存完整计划、动作、位姿读数及全部视频；无新增模型推理外的控制来源。

64 动作没有能力证明完整 Novel 成功率或长期安全性；它只关闭这次新增适配的真实请求链检查。
四历史 Revisit 对照也不替代后续修复版 full-mono mixed-role 主评测。

## 完成结果

原始目录：

```
/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_minimal_repair_20260908/native_request_smoke_v1/
```

| 原历史 | CEC 接管计划 | native / CEC 动作数 | native / CEC 实际路程 m | 前缀核验 |
|---|---:|---:|---:|---|
| gxdoqLR6rwA | 0 | 64 / 64 | 2.403223 / 2.403223 | 完全一致 |
| pLe4wQe7qrG | 0 | 64 / 64 | 2.339824 / 2.339824 | 完全一致 |

- 不曾强制 CEC 拒绝；两条查询实际均未接管，也没有触发新增转向动作。
- 每历史两臂的共享 A、初始 RGB、每次规划的当前/目标编码输入、完整候选轨迹、
  全部 critic 值及逐动作位置/朝向一致；独立 verifier 使用逐数组精确比较。
- 额外逐对读取 16 个规划时刻的深度收据：原始深度 PNG、帧号、输入图 SHA、尺度收据，
  以及裁剪前/后深度张量 SHA 和形状均完全相同。没有以内容相近替代相同帧。
- 32 个保存的深度数组均通过独立转换复算；4 个第一视角视频全部导出。
- `summary.json` 为 `completed=true, changed_source_files=[]`；
  `independent_verification.json` 为 `verified=true`，两条 pair 均为
  `exact_native_prefix_verified=true`。

这只证明这两条真实非接管请求链的 64 动作前缀等价。两臂在该预算内均未到达，
**不将其记作一项完整 Novel SR 的零结果，也不声称已证明长期零干扰。**
修复版全流程 mixed-role 配对仍是正式重跑的第一优先级。

私有任务的模型服务已由 runner 正常退出；没有停止用户驻留的 18888/8888 服务。
