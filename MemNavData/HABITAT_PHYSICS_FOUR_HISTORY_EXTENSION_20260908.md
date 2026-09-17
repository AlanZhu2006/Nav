# 四历史物理执行补充对照（2026-09-08）

## 冻结范围

本轮按用户确认补完既定本机四历史池，不改变论文、正式评测器、模型、真机部署或 HPC。
第 0 条 `gxdoqLR6rwA/episode_0000` 已完成并独立复算，保留原结果，不重复运行。

其余顺序固定：

1. `pLe4wQe7qrG/episode_0000`；
2. `yqstnuAEVhm/episode_0001`；
3. `mJXqzFtmKg4/episode_0001`。

每条保持原臂顺序 native/PP → CEC/PP → CEC/MPC → native/MPC。
同一历史内四臂共用私有模型进程，逐臂重置并重放同一 actual-online metric-A 历史；
查询使用单目深度。600 次命令、8-step 重规划、1 m 平面到达、2.5 m residual、
速度幅度、物理圆柱代理、2 s 站立初始化和 5° 失稳标准均不变。

## 本轮仅有的运行改动

单历史 runner 新增 `--history-index`，默认仍为 0。其余历史仍来自同一个 manifest，
不采样新目标。新增顺序批处理入口，复用原 verifier 与第一视角重渲染程序。
除该 runner 的选历史参数外，其余原记录的模型／运动源码和权重须与第 0 条记录一致。

使用独立端口 21680/21681；不连接、重置或关闭 18888/8888 的既有真机服务。
每个历史结束后释放该历史自建的模型子进程，再运行下一条。

## 判读约定

- 主要看同一物理执行下 CEC/native 的配对成败；其次比较同策略的 PP/MPC。
- 保存完整所选轨迹、命令、物理子步、终点与第一视角视频；独立复算实际路径与到达。
- 不将地面接触计为碰撞失败，也不将推理等待当作真实运动时间。
- 代理失稳或基础设施错误保留原始失败，不填为方法的导航失败，不回退旧 NavMesh 执行。
  批处理保留该历史错误并继续后续历史；不从失败中改阈值重试。
- 只有四个已消费历史，不做总体等价／显著增益声明，也不替换论文主表。
- A 不是物理重采集，GT 到达仍由评测器判定；不称完整物理端到端或自主 STOP。

本轮输出根目录：

`.diagnostics/habitat_physics_executor_20260908/remaining_three_v1/`

运行后的完整状态与结果以该目录 `progress.json`、`partial_results.json`、
`summary.json` 及各历史的 `independent_verification.json` 为准。

## 完成原定四臂覆盖的调度补充

第一批结束后，`pLe4wQe7qrG` 和 `mJXqzFtmKg4` 的 native/PP 分别在第 48、270
次命令触发 5° 失稳标准。原 runner 退出整个历史，导致每个历史余下三臂没有运行。
`yqstnuAEVhm` 已完成四臂并独立核验，通过的历史不重跑。

为补齐原定对照，新 runner 仅增加显式的 `--continue-after-invalid-physics`：
每个失稳臂仍立即停止、不赋 SR，但其余臂仍按原顺序 reset/replay 后执行。
这两个历史各自重新运行四臂，以继续保持同一组模型进程；旧失败记录完整保留。
`evaluate()` 函数的 AST 与第一批快照逐项一致，模型、深度、控制和倾斜标准均不变。
任何其他异常仍终止该历史，不回退到其他执行器。

第二批输出：

`.diagnostics/habitat_physics_executor_20260908/complete_arms_v2/`

第二批属于第一次结果之后的覆盖补充，不是新独立样本，也不用于择优替换第一批。
报告须并列保留失稳臂、完整臂和所有重跑状态，并核验重复 native/PP 是否复现原始过程。
