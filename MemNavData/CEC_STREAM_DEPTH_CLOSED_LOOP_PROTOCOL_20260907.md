# 历史在线深度复用：本机小闭环协议

日期：2026-09-07。用户同意本机继续；不提交 HPC、不训练、不更改正式默认值、论文或真机。

## 问题与范围

上一轮 4 历史／19 张非重复目标图的组件测试没有授权改判，但没有执行导航。
本轮只回答：**将初始 CEC 的 canonical replay depth 换成观察该帧时保存的
online depth，是否在真实查询 rollout 中破坏授权、方向与控制衔接？**

这不是长程方案、正式泛化结果，也不是 learned relocalizer。成功不意味着可以
挪用旧 CEC 的正式 SR，更不意味着已经解决漂移、障碍绕行或真机自主停车。

## 数据、两臂与执行

- 固定 `shared_online_role_pair_natural_heading_v1_smoke_20260814` 全部 4 个已消费历史：
  gxdoqLR6rwA/0000、pLe4wQe7qrG/0000、yqstnuAEVhm/0001、mJXqzFtmKg4/0001。
- 每条历史一个自然 Novel、一个不同视角 Revisit，8 查询、两臂，共 **16 次导航**。
- 仅使用这一自然方向构造，不混入 heading-controlled 或 hard-support 查询。
- 原始 A 来自实际 metric-depth NavDP；这里只读取其因果 RGB 与决策帧顺序。
  **本轮不是 actual-mono-A 的端到端试验。**
- canonical：现有按需重放深度，在线历史深度 writer 关闭。
- route_sparse：按观察时序为全部可检索历史帧保存在线 depth/confidence（stride=1），
  查询只读取已保存深度，不在缺失时偷偷改回 canonical。
- 两臂 query 观察深度均为同一 frozen LingBot mono sidecar + first40 相机高度标定；
  DINO top-8、SP/LightGlue、PnP、certificate、固定 2.5 m residual、frozen NavDP 不变。
- 600 步、exec_horizon=8、位置距离 <1 m 成功，与现有 Final14 query budget 保持一致。
  不进行 terminal U-turn、视觉 refinement、graph rescue、额外 alignment。
- role、GT pose、共视度仅用于构造／仿真执行与评分，不传给检索和 controller。
- 从实际 A 终点启动并渲染第一张新 RGB，不把末动作前图像冒充动作后观察。

## 配对与记录

4 个历史交替 canonical-first / online-first；两臂模型在同一 GPU、同一常驻模型
进程中运行，每个查询独立 reset、恢复同一个 A 和 NavDP FIFO，使用相同 seed。
初次 CEC 调用前核对主流状态摘要；写入缓存不应改变当前因果几何状态。

保留原 evaluator 的轨迹、计划和收据；另存每条 query 最终动作后位置、实际路径
长度、SPL、首次 certificate 时间、完整 rollout 时间、depth/conf CPU 缓存大小。
SPL 由逐动作实际位置及终点积分，不使用控制命令长度，也不遗漏最后一步。

首次状态哈希的测量成本排除在 certificate CUDA-synchronized 计时之外。
`request_wall_ms_including_audit` 和 rollout wall time 包含本轮诊断开销，不直接
宣称真机端到端延迟。模型加载、A history ingest 与 query latency 分开解释。

正常 certificate rejection 仍走原生 ImageGoal；实验错误不算有效 SR。若两臂
均完整拒绝，核对物理轨迹相同；这只证明两臂间一致，不是新加 native 第三臂。

## 实现隔离与结果使用

`run_cec_stream_depth_closed_loop.py` 只在自己启动的子进程里 hook reset/depth
provider 和结果观察；生产 `policy_agent.py`、server 和 evaluator 不修改。
独立端口 21560/21561，与 18888/8888 现有真机服务隔离。结束仅停止自己创建的进程。

先通过 4 场景实际渲染／geodesic preflight，再运行两臂全量小样本。任何失败保存日志，
不调阈值、换成功样本或将失败行删除。观察差异后再定位，不预设 SR 持平或提升。

CPU verifier 从实际终点重新算 success、路径及 SPL，并核对配对字段与 source。
N=4 scenes 太小，结论仅为下一轮更大配对试验提供依据，不作统计等效性声明。
