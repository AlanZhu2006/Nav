# 后向认证目标：真实转身后重规划（本机，2026-09-08）

## 问题与本次改动

上一轮四历史 Bullet 诊断中，两个 CEC/MPC 查询持续产生零 XY 轨迹。
认证 pointgoal 分别约为 `[-2.421,-0.625]`、`[-2.492,0.204]`；NavDP 原处理把负向前
分量裁成 0，之后网络又会把终点长度不足 0.5 m 的轨迹平移分量置零。
旧 PP 在零参考时仍会前进；MPC 不会。不能把旧 PP 的这种前进算成正常路径跟踪能力。

本次不改模型预处理、短轨迹 mask、critic 或 CEC，只检验一个执行接口方案：
**首次认证目标位于身后 → 用速度命令完成真实转身 → 新画面重新定位/规划 → 同一个 MPC。**
现有 evaluator 的理想化直接改 yaw 功能不启用，也不修改生产 evaluator。

## 冻结设计

- 同一已消费清单的全部四个历史：gxdoqLR6rwA、pLe4wQe7qrG、yqstnuAEVhm、mJXqzFtmKg4。
- 每个历史两臂：CEC/MPC 原流程；CEC/MPC + 首次后向物理对齐。共八条，不新增样本。
- 每个历史同一组私有常驻模型，两臂分别 reset 并重放完全相同的 actual-online A。
- A 仍为旧 metric-depth NavDP 生成；这里只比较查询执行，不能宣称全程 physical/full-mono。
- 所有 CEC/模型参数、2.5 m、MPC 参数、600 个命令预算、8 命令重规划节拍不变。
- 首次认证目标若在前方，不干预；两条原先成功的历史也纳入，检查是否回归。
- 实验臂只对首次 accepted + router active + endpoint bearing 进行一次判断，不循环改 anchor。
- 转身最大角速度 π/4 rad/s，命令 0.1 s，实际反馈误差 ≤1° 结束，最多 80 命令。
- 转身计入 600 总预算；命令平移速度为 0，但真实接触位移照常计入路程。
- 每个转身后画面只向 LingBot 追加一次；原决策时刻仅向 NavDP 重放图像，不抽样。
  转完强制在新画面重新规划，不旋转或继续执行转身前的旧规划。
- 低层跟踪/对齐使用同样的 Bullet 位姿反馈；目标方向只来自 CEC，没有目标 GT 或测地线注入。
  中间转身追加不向 LingBot 注入模拟器运动增量；原正常路径的运动记录接口保持不变。
- 动作不读 NavMesh，也不投影/缩步/瞬移落点。NavMesh 仍只用于原评测的距离检查。
- 保留 0.30 m 半径、1.50 m 高的圆柱代理、重力/接触、任一子步倾斜 ≥5° 即失稳中止。
  失稳无导航 SR 标签，继续另一臂；其他运行错误明确记录并停止，不偷偷换执行器。

固定执行顺序 off→on；本次是已观察病例的机制诊断，不作总体显著性或正式 SR 推断。
待验证的干预包括真实转身及其必要的观测/重规划时序，并非只隔离一个角速度数字。

## 观察而不修改的证据

私有 NavDP wrapper 记录原 pointgoal、裁剪后 pointgoal、0.5 m mask 前后全部轨迹、
critic、最终执行轨迹及当前 JPG SHA/seed。额外读取不调用模型第二次、不消耗随机数。
私有 LingBot wrapper 记录原有 cam_pose9/深度状态，不重新 forward、不修正其位姿。

预先关心：

1. 原流程是否复现后向裁剪与短轨迹置零；原始 diffusion 轨迹是否本来就很短。
2. 转身后重新送入的 pointgoal 是否进入前向域；轨迹是否恢复非零。
3. 恢复轨迹后能否真正推进/到达，还是出现机体接触、局部不可通行或失稳。
4. 实际转身与 LingBot 的相对朝向/平移变化是否一致，是否出现纯旋转漂移。
5. 两个原先成功的控制病例是否保持；完整记录所有八个结果，不只展示救回。

## 已通过预检（尚不是导航结果）

`alignment_primitive_v1/summary.json`：

| 预检 | 命令数 | 实际平面路程 | 最终角度误差 | 最大倾斜 |
|---|---:|---:|---:|---:|
| −165° 速度驱动转身 | 40 | 0.013834 m | 0.2584° | 0.0626° |
| +175° 速度驱动转身 | 42 | 0.018025 m | 0.1852° | 0.0839° |
| MPC 零平移参考 | 20 | 0.0000171 m | 0° | 0° |

转身是物理执行，不是严格零实际位移。以上只验证平地原语，不能代替带家具场景。
7 项 CPU 测试通过，包括原代码 mask 插桩输出/随机状态不变、原 clip 行为、
正负转向、无认证不转、首次前向不转、生产 evaluator 未编辑。
额外覆盖 LingBot 最初 S−1 帧尚未产生 pose 的正常预热状态。

## 启动与输出

```bash
.diagnostics/habitat_bullet_env_20260908/bin/python -u \
  MemNavData/run_habitat_bullet_alignment_local.py \
  --out .diagnostics/habitat_physics_executor_20260908/alignment_pair_v2 \
  --primitive .diagnostics/habitat_physics_executor_20260908/alignment_primitive_v1
```

输出总根：`.diagnostics/habitat_physics_executor_20260908/alignment_pair_v2/`。
各历史有模型源码/权重哈希、输入清单、逐动作物理状态、完整 mask 诊断、
LingBot 位姿记录以及最终实际位置。`progress.json` 是运行状态，不代表成功率。

不启动 HPC、不改论文、不访问机器人、不使用真机端口、不 commit/push。
本轮只启用私有 21680/21681；结束后仅清理本轮拥有的进程。

### 启动修复记录

v1 在首个历史重放的第一个 RGB 就退出：新增只读 pose 记录器错误地假设初始 pose
已经存在，实际 LingBot 前 S−1 帧只缓冲。未进入任何 query 动作，没有 SR。
只修记录器对正常预热状态的表示（`camera_pose9=null`），加回归测试后以 v2 新目录重启。
v1 日志保留，未改模型/控制参数，也没有把它当导航失败或从中筛选结果。

### 完成状态

v2 的四历史、八个臂已全部尝试完成；详见
`MemNavData/HABITAT_PHYSICAL_ALIGNMENT_RESULT_20260908.md`。
原流程 2 到达/2 停滞，对齐臂 3 到达/1 失稳；未放宽预定边界。
