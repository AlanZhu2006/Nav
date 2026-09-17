# X-NavDP × 真实物理执行：当前进展（2026-09-08）

目标：本机检验 X 策略/MPC 是否优于原 NavDP 转身后重新规划，分离策略与执行器影响。
不是 HPC 任务；不修改论文、真机或生产策略，也没有训练或新数据选择。

## 最新纠正：不是完整严格复现，已发现 RGB 适配错误

**最新收口：** 四历史 RGB 配对已完成，最终目录为 `xnavdp_rgb_pair_repair_v1`。
独立复算通过、八段录像已导出。修复臂为 2 到达 / 1 未到达 / 1 代理失稳，
不能报告删去无效样本后的 SR。通用 X 服务已同步修复，两类入口共用正确 RGB 解码；
41 项测试通过。共用入口真实模型短测也已完成并独立验证，actor 输入、候选、Q 值与
初次正确诊断短测完全一致。当前这批本机任务已经结束。
完整结果见 `HABITAT_XNAVDP_RGB_REPAIR_RESULT_20260908.md`。
下面 10:05 的启动状态保留追溯，不代表仍有该旧进程在跑。

v2 已完成四历史三臂并导出录像，但 X 实际收到 BGR，而官方原始客户端—服务端的合成链路给 actor 的是 RGB。
此前的独立复算只验证测量/配对，遗漏了编码端与解码端通道语义。旧 X 的 3/4 到达只能保留为错误输入条件的诊断。
完整审计见 `HABITAT_XNAVDP_REPRODUCTION_AUDIT_20260908.md`；不能再把它解释为 X 能力上限。

修复和执行进度（2026-09-08 10:05 CST 附近）：

- 当前帧和短期历史统一恢复 RGB，CEC/LingBot 的 JPEG 和深度交易不变。
- 19 项单元测试、实际官方编码/解码的 5 个离线案例通过。
- `xnavdp_rgb_smoke_v1` 八命令集成完成，独立 verifier 通过；无 GT sensor depth、无运动 NavMesh 查询。
- 四历史两臂本机配对已启动：`xnavdp_rgb_pair_v1`，batch PID 2612087。
- 旧/新颜色顺序按历史交替；只改 RGB 输入，不同时改距离、速度、MPC 或引入失败回退。
- 结束后自动独立复算并重渲染第一视角视频；最新状态读该目录 `progress.json` 和 `finalization_status.json`。

本次仍是单目/固定残差/Habitat 圆柱的迁移诊断；不等同官方 RGB-D、真实相对 PointGoal、Isaac 具身与到达定义。
以下为此前集成记录，保留追溯，不代表当前状态。

## 已完成

- 11 项 CPU 接口/已有转身测试通过。
- 真实 acados 三种参考测试通过：零参考、向前、向后。
  首次求解分别约 1.74 / 1.12 / 1.10 ms；这不是端到端推理延迟。
  向后非零参考输出约 −0.376 m/s，零参考速度约 1.06e−7 m/s。
- 八命令 X-only 集成已完成并独立复算通过：
  官方 posttrain 权重、原证书、原 2.5 m、逐帧 LingBot 单目深度、真实 Bullet 执行。
  8 命令中 5 个负速度；这只是接口测试，不能按其 `reached=0` 判断方法失败。
- X 上传接口只含 RGB，不上传或读取模拟器深度；原 frame/token 验证保持。
- X 活跃时继续更新 native FIFO；拒绝时仍执行原 native/base MPC。

集成证据：
`.diagnostics/habitat_physics_executor_20260908/xnavdp_integration_smoke_v2/xnavdp_independent_verification.json`

## 三臂完整对照

在同一 4-history 既有池上依次运行：

1. `cec_aligned__mpc`：原 mixed NavDP + 真实转身后新视角规划 + base MPC。
2. `cec__xmpc`：原 mixed NavDP + X MPC，不加转身。
3. `cec_x__xmpc`：官方 X posttrain PointGoal + X MPC，不加转身。

相同 CEC、单目深度、初始 RGB/物理体、600 命令预算；X 的负速度、N=30、
8-command chunk 与官方 RTC 保留。RTC 读理想模拟器里程计，不读 GT 目标/路线。
因此完整 X 臂是 controller stack 对照，不能将差异都归给 MPC。

运行根：
`.diagnostics/habitat_physics_executor_20260908/xnavdp_stack_v2/`

查看顺序：`progress.json` → `partial_results.json` → 完成后的 `summary.json`。
独立复算入口：`MemNavData/verify_habitat_xnavdp_stack.py`。
最终归档状态：`finalization_status.json`；通过后各历史自动导出 `first_person_v1/`。

## 已观察但不外推的 v1 结果

首个 gxdoq 历史：转身+base MPC 到达，125 命令，与上一轮实际物理轨迹逐项一致；
只换 X MPC 跑到 600 命令未到达，终距约 3.670 m。
随后 X 臂在导航前被旧共享评测器固定标签校验拒绝。
不能据此填 X 失败或对 X 总体下结论。

修复发现：旧 X 分支属于旧 geometry router；最新 certificate 路由本身固定调用 mixed。
本次仅在隔离诊断中允许 X 配置，并将已认证请求分派到 X；生产验证/数学不改。
经独立八命令预检后新建 v2 完整运行；v1 保留，不与 v2 当作独立样本合并。

截至本状态记录，v2 的第一个转身基线也已到达，其余臂仍在运行。
后续真实状态以运行根 JSON 为准，不把这段启动时快照当作最终成绩。

## 边界

四历史已消费、查询级；历史 A 是旧 metric online A，不是全物理 mono A。
动作不用 NavMesh 投影；底层理想里程计和评测 GT 到达仍存在。
圆柱不是 Go2；5°失稳不删除，不按幸存样本计算 SR；推理等待期间仿真暂停。
真实机体安全、自主 STOP 与部署端到端延迟仍未由这次测试证明。
