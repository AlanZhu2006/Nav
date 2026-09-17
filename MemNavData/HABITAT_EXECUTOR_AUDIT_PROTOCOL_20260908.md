# Habitat 执行器本机调查（2026-09-08）

用户要求立即本机调查。只运行独立诊断，不改正式执行器、论文、封存结果或真机服务，不提交 HPC。

## 先回答什么

1. 已保存实际位移是否在原始 NavMesh 上可从前一点到达？是否有侧向投影或超出原始步长的移动？
2. 同一个真实预测轨迹，经原执行器和 Habitat `try_step` 后的落点是否不同？
3. 在相同查询上，仅替换执行器，是否改变闭环结果？

`try_step` 仍使用 GT NavMesh，只是标准的起点到终点碰撞步进；不是 GT-free 导航，也不等价于 IsaacSim 动力学或 Go2 全身避障。

## 已有轨迹检查

先用已下载且绑定原始 NavMesh 的 HM3D eF36g7L6Z9M/005 长距离案例。三个历史结果全部检查，不选择成功臂。
原日志缺完整预测轨迹（只有 SHA），因此该阶段只验证记录位移可达性，不能声称重放原始控制命令。
禁止由侧移次数推断 SR 被提高了多少，也禁止由落点可达推断命令执行等价。

## 新本机配对

- 原始 consumed 四场景 manifest 的前 N histories，默认先第一个 gxdoqLR6rwA/0000。
- 每 history 只测试原始自然 Revisit 查询；运行时不读取 role。不是 Novel safety 实验。
- 2 policies（mono native、mono CEC）× 2 executors（原 snap、标准 try_step），4 条 rollout/history。
- 首 history 顺序：native/snap、native/try_step、CEC/try_step、CEC/snap；下个 history 反向。
- 默认 600 步，8-action horizon，1 m evaluator 到达，2.5 m residual；NavDP、CEC、RGB/mono-depth 合约不变。
- 历史是既有 actual metric-NavDP-A，所有臂读同一 causal RGB；本轮不验证 actual-mono-A 全链路。
- 固定 canonical historical depth，避免同时改变深度复用实现；NavDP query depth 为同一 LingBot mono sidecar。
- 私有端口 21680/21681、独立 buffer；只关闭自己创建的进程。
- 每步记录原始命令、原执行器、try_step、no-sliding 三个落点；no-sliding 在此阶段只作为 shadow。
- 每个规划保存完整选中轨迹和候选；保存末动作后坐标、实际路径和 SPL。
- 诊断所重建的 legacy 结果每步与未修改原函数比较；legacy 臂直接返回原函数的元组。

本轮小样本结果只用于判断执行协议是否实质改变行为，不能宣布正式 SR 等效或替换论文结果。
