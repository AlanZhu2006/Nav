# CEC + Mono NavDP：Unitree Go2 双机部署（2026-08-20）

## 1. 当前冻结架构

实机方法已经从“RGB-D NavDP + CEC”切到与 Full-Mono 仿真实验一致的主架构：

```text
Unitree Go2 / Jetson Orin NX                    RTX 4090 workstation
────────────────────────────                    ───────────────────────
D435i RGB ───────────────── SSH tunnel ───────> one causal RGB stream
                                                    |
                                         frozen LingBot streaming state
                                           /                     \
                           dense short-range readout       sparse long-range proof
                          first-40 scale + mono depth       CEC retrieve/verify
                                           \                     /
                                            frozen NavDP controller
                                                      |
                                           24-point local trajectory
                                                      |
Jetson local tracker <────────────────────────────────┘
  + D435i aligned depth collision safety only
  + stale-plan stop + 0.35 s Go2 watchdog + gamepad priority
```

核心口径是 **one causal RGB stream, two time scales, one frozen policy**。LingBot 不与
NavDP 竞争动作：它提供短程 dense depth readout，以及长程 CEC proof/bearing；NavDP 始终
是唯一轨迹生成器。

## 2. 单目与安全传感器边界

- 导航策略只消费 RGB。上位机 NavDP 固定为 `depth_source=monocular_sidecar`；每步响应
  必须带单目 depth receipt，并证明 `metric_depth_sensor_consumed=false`。
- Jetson 的旧客户端仍可上传 aligned depth，以保持 wire compatibility；hub 明确丢弃，
  不向 NavDP 转发。
- D435i depth 继续在 Jetson `trajectory_control.py` 内做近障急停。这是独立安全壳，不参与
  目标条件、记忆检索、bearing 或轨迹生成。论文必须写“monocular navigation policy with
  a local depth safety layer”，不能写成整台机器人完全没有 depth sensor。
- hub health 固定暴露：
  `navigation_sensor_contract=causal_monocular_rgb_v1`、
  `navdp_depth_source=monocular_sidecar`、
  `metric_depth_sensor_consumed_by_policy=false`。

## 3. 在线决策与故障语义

```text
current RGB + ImageGoal
  -> exactly one MemNav retrieval probe / stream append
  -> CEC certificate
       accept: scale-free bearing x 2.5 m -> frozen mixed NavDP
       reject/error after successful append: exact mono-native NavDP
  -> NavDP requests current depth from the same LingBot sidecar
```

两个“失败”不能混为一谈：

1. certificate 拒绝或 proof endpoint 失败，但本帧已成功写入流：允许精确回退
   mono-native；
2. causal LingBot stream append 本身失败：CEC 与当前单目 depth 同时失去可信状态，hub
   锁存 `reset_required`，不生成新轨迹，由 Jetson stale-plan 和 Go2 watchdog 停车。

旧 protocol-v1 的“MemNav 掉线仍回退 metric native”在 Full-Mono 中不再成立，已经删除。

## 4. 代码与 release

上位机核心：

- `MemNavData/realworld_cec_hub.py`（protocol v2）；
- `MemNavData/run_realworld_policy_stack.sh`；
- `MemNavData/run_realworld_memnav_server.sh`；
- `MemNavData/run_realworld_navdp_server.sh`；
- `MemNavData/run_realworld_cec_hub.sh`；
- `MemNavData/monocular_depth_runtime.py`；
- `NavDP/baselines/{memnav,navdp}` 对应 servers/agents。

下位机 live overlay：

```text
/home/nvidia/twork/NavDP/deployment/go2/offboard
```

内容寻址 release：

```text
/home/nvidia/twork/NavDP/deployment/go2/releases/
  cec_mono_20260820_d656b9d9ae30de73/
```

回滚副本：

```text
/home/nvidia/twork/NavDP/deployment/go2/releases/
  rollback_pre_mono_20260820_d656b9d9ae30de73/
```

完整 checksum 与 smoke 回执：
`REALWORLD_MONO_RELEASE_RECEIPT_20260820.json`。
其中 `release_tree_sha256` 只覆盖四个可执行 offboard payload；后写入的
`RELEASE_RECEIPT.json` 是核验元数据，不参与该内容地址。

## 5. 启动门

Full-Mono scale 必须使用安装后 D435i 光心到地面的真实高度。启动脚本不再静默采用默认值：

```bash
cd /home/asus/Research/Nav-graph-blind
export CEC_CAMERA_HEIGHT_M=<measured-metres>
MemNavData/run_realworld_policy_stack.sh
curl -fsS http://127.0.0.1:18889/healthz
```

未实测高度前禁止正式加载并声称 mono scale 已校准。`0.5 m` 只在 2026-08-20 的
health-only transport smoke 中用于生成 health 字段，没有 reset、推理或运动。

Jetson 顺序保持 fail-closed：

```bash
cd /home/nvidia/twork/NavDP

# 1. tunnel + contract preflight
tmux new -s cec-tunnel 'exec deployment/go2/offboard/run_policy_tunnel.sh'
bash deployment/go2/offboard/preflight_offboard.sh

# 2. camera + disabled adapter；不启动底盘
bash deployment/go2/offboard/run_offboard_stack.sh --with-rviz

# 3. 只有静态验收全部通过后，现场操作者才可另行启动 bridge
bash deployment/go2/offboard/run_offboard_stack.sh --with-go2 --with-rviz
```

即使第三步启动 bridge，adapter 仍为 `enable_on_start=false`。最后的 motion enable 必须由
现场操作者手持遥控器单独执行，不属于自动部署。

## 6. 2026-08-20 已完成

- 本地 hub/mono runtime 与组合协议：34 tests passed，Python compile 与 shell syntax 通过；
- Jetson `tegra-ubuntu` 可达，更新前无 NavDP/ROS/Go2 运行进程；
- release 先旁路同步，远端 SHA 复核后才更新 live overlay；旧文件保留可回滚；
- Jetson 到 4090 passwordless SSH 正常；
- 临时 protocol-v2 health hub + loopback tunnel 的 offboard preflight 5/5 通过；
- smoke 未加载 MemNav/NavDP 权重，未启动相机、ROS adapter 或 Go2 bridge；
- smoke 后上下位机临时进程全部停止，零运动命令。

## 7. 下一次真机验收

1. 实测并记录 D435i optical-center height；
2. 启动上位机 full stack，reset receipt 必须同时证明 CEC、mono sidecar 与冻结 depth source；
3. Jetson 仅启动相机和 disabled adapter，连续运行至少 10 分钟；
4. 审计前 40 帧必须是 `bootstrap_zero_depth`，第 40 帧后 scale receipt 只冻结一次；
5. 每条 trajectory receipt 必须证明 policy 未消费 metric depth；
6. 静态左/右 bearing 符号校准；
7. tunnel kill 与 MemNav kill 两种 fault injection 均应输出零速度并锁存 reset；
8. 最后才做 0.5--1.0 m 系绳、低速、手柄在手的运动测试。

截至本文，不存在 Full-Mono 真机 SR；当前状态是 **代码已部署、transport contract 已通过、
等待物理高度与静态相机验收**。
