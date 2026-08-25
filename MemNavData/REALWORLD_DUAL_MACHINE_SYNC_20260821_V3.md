# MemNav Real-World 双机同步总账 v3（2026-08-24 更新）

接续 `REALWORLD_DUAL_MACHINE_SYNC_20260821.md`（da92b76 同步）与
`REALWORLD_REVISIT_PHASE_PROTOCOL_FIX_20260821.md`（v3 修复回执）。

## 结论

Protocol v3（两阶段 episode 契约 + goal-candidate 捕获 + NavDP FIFO 暖启动）
已完成研究源 → 独立发布仓库 → Jetson 的三方同步，并在 RTX 本机对
**真实权重栈**完成端到端相位流冒烟。Jetson 于 2026-08-24 恢复上线，
checkout 已安全 fast-forward 到发布仓库 `main` 的 `6cae26d`；同步过程保留了
现场已有的相机就绪等待 `1 s -> 3 s` 改动，没有启动相机、ROS adapter、
Go2 bridge 或任何运动进程。

## 版本状态

| 位置 | 版本 | 状态 |
| --- | --- | --- |
| 研究工作区 `Nav-graph-blind` | protocol-v3 hub + scorer + 回执文档 | 15/15 hub tests,发布门三件套 31 passed |
| 发布仓库 `AlanZhu2006/Memnav_Realworld` | `f9a1e372c67a95fe0469d205f31c172bc12dbcb9`（`main` 与 `sync/fullmono-cec-v3-20260821` 已推送） | gpu 35 + go2 40 = 75 passed;verify_public_baseline failures=0 |
| RTX 运行时 checkout `/home/asus/Research/Memnav_Realworld` | 同 `f9a1e37`（即工作树本身） | 已即时生效 |
| Jetson `/home/nvidia/twork/NavDP` | `6cae26d71376f1c698bfab6494094e1dba218576` | 已同步；公开清单 `failures=0`；真机运行环境 43 tests passed；保留一项未提交现场改动 |

## 本次进入发布仓库的内容（21 files, +1371/-42）

- v3 `realworld_cec_hub.py`（相位状态机、`/memory_step`、`/goal_candidate`、
  `/begin_revisit`、录制期 goal 查询 400 拒绝、NavDP stride-8 暖启动 + 队列
  硬校验、fail-closed 锁存）与 15 项测试；
- `score_realworld_revisit_goal.py` 弱共视评分工具（冻结 DINO/LightGlue 端点,
  临时阈值 `inliers>=16`、`max_cos<=0.90`,标定待 disabled-adapter walk）;
- 更新后的 `monocular_depth_runtime.py`（含 20260821 depth transaction token）;
- Go2 adapter v3:`two_phase_episode` 参数（默认 false,旧路径不变）、录制期
  逐帧 `/memory_step`、`~/capture_goal_candidate` 与 `~/begin_revisit` Trigger
  services、status 上报 phase/frames/candidates;运动授权、estop、watchdog、
  Go2 bridge 代码零改动;
- `fullmono.sh` 经 SSH 转发 `CEC_CAMERA_HEIGHT_M` / `CEC_GOAL_CANDIDATE_DIR`
  （高度无默认值,保持安全门语义）;`run_offboard_stack.sh` 对 fullmono 栈
  导出 `NAVDP_TWO_PHASE=true`;
- `manifests/realworld_fullmono_v3.json`（v2 保留为历史）与升级后的
  `tools/verify_public_baseline.py`（protocol 3、相位契约、端点声明、暖启动
  receipt 断言）;
- 文档:RUNBOOK v3 任务流程（§8）、ARCHITECTURE 两阶段契约小节、
  CURRENT_STATUS、SOURCE_MANIFEST、README、`FULL_MONO_RELEASE_20260821_V3.md`。

## RTX 真栈端到端冒烟（本机,2026-08-21 晚）

按 `fullmono.sh` 的精确调用方式执行 `preflight.sh`（failures=0）与
`run_policy_stack.sh`,对真实 LingBot + NavDP 权重栈验证:

1. reset → `phase=memory_recording`,protocol 3;
2. 录制期 `/imagegoal_step` → **400**,错误信息含 begin_revisit 指引;
3. 10 × `/memory_step` → `frame_idx=9 / frames_recorded=10`;
4. `/goal_candidate` → receipt(`appended_to_memory=false`,落盘 + sha256);
5. `/begin_revisit` → 暖启动 2 帧(索引 [2,10],stride-8 时序正向),
   `queue_lengths=[2]` 与真实 NavDP `memory_size=8` 校验通过;
6. `/imagegoal_step` → 200,certificate 正确 abstain
   (`minimum_query_coverage`),native 控制器出轨迹,mono receipt
   `bootstrap_zero_depth`(frame 10 < 40,符合契约),
   `metric_depth_sensor_consumed=false`;
7. healthz 终态 `phase=revisit_query / frames_recorded=10 / candidates=1`。

冒烟后已执行 `stop_policy_stack.sh`,三个 policy 端口无监听,tmux 无会话,
恢复文档化安全基线。冒烟前发现并干净关闭了白天失败 trial 遗留的 v2 栈
(21:46 起的 `cec-realworld` 会话)——旧 v2 hub 无相位保护,继续运行本身是
下次 trial 的隐患。

## Jetson 同步完成回执（2026-08-24）

同步前审计：Jetson 位于 `cebc899`，工作树仅有
`deployment/go2/offboard/run_offboard_stack.sh` 的一项现场修改：等待真实
`CameraInfo` 的单次 timeout 从 1 秒改为 3 秒。该修改先进入可恢复 stash，
随后执行 `git fetch` 与 `git merge --ff-only memnav-realworld/main`，最后
`stash pop` 无冲突恢复。同步后：

- `HEAD=6cae26d71376f1c698bfab6494094e1dba218576`，与发布仓库 `main` 一致；
- protocol-v3 对同一脚本新增的 `NAVDP_TWO_PHASE=true` 与现场 3 秒等待同时存在；
- `git diff --check` 通过，工作树仅保留上述一项有意的未提交现场改动；
- `tools/verify_public_baseline.py`：`failures=0`；
- shell 语法、Python compile 均通过；
- 使用真实部署环境 `.venv-navdp` 并加载 ROS setup 后，Jetson go2 tests：
  **43 passed**；系统 Python 缺 Flask、旧 pytest/anyio 自动插件冲突均不是
  部署环境缺陷，测试通过 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 隔离第三方插件；
- live offboard 路径就是同一 checkout 下的
  `/home/nvidia/twork/NavDP/deployment/go2/offboard`，不存在需要额外覆盖的副本。

注意:v3 后旧 adapter 流程会在第一次 goal 查询被 hub 以 400 快速拒绝
(而不是再次静默瞎走),这是预期安全行为;Jetson 必须同步到 f9a1e37 才能
跑 v3 任务流。该版本要求现已满足；本次同步不等于运动授权。

## 剩余门(2026-08-21 晚更正)

操作者确认:D435i 光心高度 0.42 m 为确定值(非估测占位);且此前 RGB-D
NavDP Go2 部署真实有效——相机管线、adapter、轨迹跟踪、stale-plan/watchdog、
bridge 均为 field-proven 组件。剩余未验证面因此收窄到单目新增部分:

1. 真机 D435i RGB 上的 frame 0--39 bootstrap 与 frame-40 单次 scale freeze
   审计(此前真机全部为 metric RGB-D,mono scale 只在仿真与 RTX 真栈冒烟
   中跑过);
2. 真实安装位上的 CEC bearing 左右符号校准;
3. scorer 弱共视 proxy 阈值标定;
4. v3 相位流首次上机(RTX 真栈冒烟已过,机器人端未跑)。

1--3 可合并为一次 adapter-disabled 的 pre-trial 记录行走;之后即可授权系绳
低速 v3 revisit trial。fault injection 降级为建议项(fail-closed 路径已被
测试覆盖,watchdog 层为 field-proven)。仍无 Full-Mono 真机 SR。

发布仓文档同步更正:`6cae26d`(main)。
