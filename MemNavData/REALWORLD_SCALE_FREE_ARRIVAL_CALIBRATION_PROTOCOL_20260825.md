# 真机 scale-free arrival calibration 协议（2026-08-25）

## 1. 为什么现在不能直接接 STOP

S→Q 真机轨迹中，LightGlue/PnP 在 frames 325--328 连续通过，但单目 PnP metric
translation 最低报到 `0.125 m`，独立物理评估却显示整轮最近也只有 `0.993 m`。该尺度
至少低估 `7.9x`。所以：

- 两视图 proof 可以授权方向；
- metric translation norm 只能写诊断；
- NavDP 零/短轨迹不是到达；
- 未经物理标定的高共视也不能授权 STOP。

最新只读 LightGlue audit 在 431 帧中只有 15 帧通过既有 certificate precheck。最强
frame 326 有 331 matches、299 fundamental inliers、query/reference hull coverage
`0.712/0.398`，normalized median identity flow `0.0613`；但它仍是物理未到达负样本。
这证明视觉残差必须以 proof 为条件，也证明不能从当前一条失败轨迹后调一个 STOP 阈值。

## 2. 新增的 measurement-only 合约

代码：`MemNavData/realworld_visual_convergence_contract.py`。

它只消费尺度无关的图像量：

- frozen two-view precheck；
- fundamental inliers 与双侧 hull coverage；
- normalized identity flow；
- affine near-identity corner residual、rotation 和 scale；
- 连续 frame identity 与同一 goal SHA-256。

它明确不读取：

- 单目或 metric translation norm；
- Novel/Revisit role；
- simulator geodesic；
- NavDP 的零轨迹或 critic stop。

状态机分两步：

```text
单帧 proof-conditioned near-view
  -> request_hold（只请求零平移视觉对齐）
  -> hold 中连续 K 帧重新验证
  -> shadow_stop
```

当前 `runtime_stop_authorized` 永远为 false。即使 shadow 条件通过，也不会获得 ROS/Jetson
电机或 estop 权限；物理 confirmation 是单独的 release gate。

## 3. 物理数据冻结

至少选择 3--4 个外观不同的地点。每个地点先固定 ImageGoal 拍摄位姿，再按预声明网格
采样当前 RGB：

```text
distance: 0, 0.25, 0.50, 1.00 m
yaw:      0, +/-10, +/-20 deg
repeat:   每个位姿至少 3 帧
```

实际 success radius 不写死在代码中，必须在 manifest 中预声明。每个样本绑定：

- `location_id`；
- `sample_id`；
- `distance_m`、`yaw_deg`（卷尺/独立 pose）；
- goal/frame SHA-256；
- `calibration` 或 `confirmation` split。

按地点切分，禁止同一地点同时进入 calibration 和 confirmation。先冻结全部物理标签和
split，再运行 LightGlue audit；不能看完视觉分数后移动样本。

## 4. 阈值与确认纪律

1. 只允许从 calibration locations 冻结 rule JSON；
2. confirmation locations 在 rule SHA-256 冻结前不可读取视觉结果；
3. confirmation 至少应含 60 个 success-radius 外负样本；若 `0/60` 误停，零误停率的
   单侧 95% 上界仍约为 4.9%，因此不能写“形式安全”；
4. 同时报告 positive coverage，避免以永不 STOP 换取零假阳性；
5. 连续 K 帧必须在 terminal hold 中取得，移动中的单帧不能累积；
6. 任何 proof 丢失、frame gap、goal SHA 变化或服务异常都把 streak 清零并回到 replan。

## 5. 晋级顺序

```text
静态有标签采集
-> calibration-only freeze
-> held-out location confirmation
-> disabled shadow replay
-> 系绳低速、人工终止 trial
-> 自动 estop shadow/人工复核
-> 最后才允许 runtime STOP
```

未完成 held-out confirmation 前，不再进行盲目的自主 Novel/Revisit 导航，也不把
`shadow_stop` 写成真机成功。

## 6. 与现有工程的边界

- `Memnav_Realworld/deployment/gpu/audit_visual_convergence.py` 负责生成 measurement rows；
- 本文件对应的纯合约负责验证 rule、时序和物理 population provenance；
- `revisit_local_pose_adapter.py` 继续只授权 scale-free bearing/atomic turn；
- Jetson executor 继续拒绝任何没有独立 arrival release schema 的 STOP。

因此本次新增不会改变已经部署的 direct-bearing-v2，也不会启动相机、ROS、RTX 服务或
机器人运动。
