# MemNav Real-World 双机同步总账（2026-08-21）

## 结论

截至 2026-08-21 17:51（Asia/Shanghai），Full-Mono CEC 真机软件已经完成
**研究源、独立发布仓库、Jetson live overlay 三方同步**。机器人侧保持停机，
没有启动相机、ROS adapter、Go2 bridge，也没有发送运动命令。

这次同步解决了一个真实的可复现性缺口：Jetson 从 2026-08-20 起已经使用
protocol-v2 的两个新 offboard 脚本，但其独立
`AlanZhu2006/Memnav_Realworld` checkout 仍停在 2026-08-18 的 RGB-D 版本，
表现为两个 tracked files 被本地修改、immutable release 目录未纳入 Git
状态。设备可用，但 fresh clone 无法重建同一架构。

## 当前三方版本

| 位置 | 路径 / 仓库 | 冻结版本 |
| --- | --- | --- |
| 研究工作区 | `/home/asus/Research/Nav-graph-blind` | `70387b63db65fedf1eb74bbe995631139d7b8e18` |
| 独立真机仓库 | `AlanZhu2006/Memnav_Realworld` | `da92b76`（`main` 与 `sync/fullmono-cec-20260821`） |
| 本机独立 checkout | `/home/asus/Research/Memnav_Realworld` | `da92b76` |
| Jetson checkout | `/home/nvidia/twork/NavDP` | `da92b76`，branch `master` tracking `memnav-realworld/main` |
| Jetson live overlay | `/home/nvidia/twork/NavDP/deployment/go2/offboard` | 与内容寻址 release 完全一致 |
| Jetson immutable release | `cec_mono_20260820_d656b9d9ae30de73` | 保留 |
| Jetson rollback | `rollback_pre_mono_20260820_d656b9d9ae30de73` | 保留 |

设备原来的两处本地 overlay 修改先保存为
`stash@{0}: pre-da92b76-device-overlay-20260821`，之后仅执行 fast-forward。
新 Git 提交包含相同 payload，因此 Jetson 工作树现在干净；stash 只作为可恢复
审计副本保留。

## 冻结架构

~~~text
Jetson D435i RGB -> SSH tunnel -> one causal LingBot RGB stream
                                      |                |
                              dense mono depth     CEC proof/bearing
                                      \                /
                                      frozen NavDP
                                           |
Jetson tracker <- 24-point local trajectory
     + aligned-depth collision guard
     + stale-plan stop / watchdog / gamepad authority
~~~

关键合约：

- `navigation_sensor_contract=causal_monocular_rgb_v1`；
- NavDP 固定 `depth_source=monocular_sidecar`；
- D435i aligned depth 不进入策略，只留在 Jetson 做 collision safety 和可选
  arrival audit；
- certificate reject 精确回退 mono-native NavDP；
- causal LingBot append 失败或 NavDP 状态不确定时锁存
  `reset_required`，不能回退 metric depth；
- CEC 通过时只输出 scale-free bearing，经固定 2.5 m residual 交给冻结
  NavDP；NavDP 始终是唯一 trajectory generator。

## 本次进入独立仓库的内容

- protocol-v2 `realworld_cec_hub.py`；
- `monocular_depth_runtime.py` 与 first-40 scale/SHA receipt；
- mono-sidecar NavDP server、policy agent/network state-safe interfaces；
- RTX 4090 的 preflight/start/stop scripts；
- Jetson 的完整 Full-Mono health contract；
- Full-Mono architecture、中文 runbook、current status、source manifest；
- machine-readable `realworld_fullmono_v2.json`；
- 更新后的公开架构 SVG 和 baseline verifier。

模型权重、LingBot/LightGlue 依赖、研究数据、runtime buffer、SSH key 与真机
goal assets均未进入公开仓库。

## 校验结果

- 独立仓库 GPU contract tests：**26 passed**；
- 研究工作区 protocol-v2 hub/runtime tests：**10 passed**；
- Python compile：通过；
- RTX/Jetson 全部 shell syntax：通过；
- JSON 与 SVG XML：通过；
- public baseline verifier：全部通过，failures=0；
- Jetson SSH：正常；
- Jetson 当前 tmux session：无；
- 8888/18888/18889 policy ports：无监听；
- 实际导航相关进程：无。

Jetson 四个 live payload：

| 文件 | SHA-256 |
| --- | --- |
| `preflight_offboard.sh` | `770fe4eb205b6054d6ab50b9bff7fd12b5b587f8eefd1d7fe9bad6e3db8b1d0d` |
| `run_offboard_stack.sh` | `9f363b7bd5aebe783b92b7150dc2afe39580de4e8181918751167784f63f9f24` |
| `run_policy_tunnel.sh` | `eb65fb3c88c0976b17ddc87ee99e6481e6d4d0c718cc7121630446f76006c2c3` |
| `stop_offboard_stack.sh` | `e6b239f1cd2c51d59bd09c57348e037697a7bd4de47c0c9316860c608ed798c3` |
| `fullmono.sh` | `050ba1a45fc4d8d91dfb520bc002654f607bf13634b10373ec7db9afce1420a1` |

原始四文件内容地址 release
`d656b9d9ae30de73f1d70a52b0150318f3dda238d6631dbae42f0a98dec973c2`
仍作为 rollback 保留；当前 `run_offboard_stack.sh` 只新增 fail-closed 相机存活
检查，`fullmono.sh` 只负责编排，不改变导航决策或运动授权。

## 尚未完成，不能误报

- 尚未实测并记录安装后的 D435i optical-center height；
- 尚未在该物理高度下 reset 完整 MemNav + NavDP 权重栈；
- 尚未完成 camera + disabled adapter 的 10 分钟静态验收；
- 尚未审计 frame 0--39 bootstrap 与 frame 40 单次 scale freeze；
- 尚未完成左右 bearing sign 校准；
- 尚未在最终 release 上做 tunnel-kill / MemNav-kill fault injection；
- 尚未做系绳、低速、0.5--1.0 m 的 Full-Mono 真机闭环；
- 因此仍没有 Full-Mono 真机 SR/SPL。

## 下一步唯一安全入口

现场估测的 D435i 光心离地高度为 `0.42 m`；正式实验前仍需在标准站立姿态下
精确复测。双机编排现由 Jetson 统一发起：

```bash
cd /home/nvidia/twork/NavDP
bash deployment/go2/offboard/fullmono.sh start --with-rviz
bash deployment/go2/offboard/fullmono.sh status
bash deployment/go2/offboard/fullmono.sh stop
```

`start` 会通过免密 SSH 启动或复用 4090 的 Full-Mono 策略栈，再启动 Jetson
隧道、D435i 与禁用态 adapter。默认不启动 Go2 bridge；即使显式增加
`--with-go2`，adapter 仍保持锁定，运动必须在现场通过独立 ROS service 授权。
所有静态 receipt 与故障注入通过后，才允许执行该授权。

实际无运动启动审计发现当前 Jetson 未枚举到 D435i，旧 launcher 会留下“服务
已启动”的假象。`run_offboard_stack.sh` 已补充真实 `CameraInfo` readiness gate：
相机缺失时打印 `No device detected`、不启动 adapter，并自动回滚 Jetson 和
4090 两端。该失败路径已实测，两端 tmux session 与 8888/18888/18889 监听均无
残留；因此当前现场阻塞是重新接通 D435i，而不是 NavDP、SSH 或双机编排。
