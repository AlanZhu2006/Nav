# X-NavDP 后方目标：本机物理对照（2026-09-08）

## 问题与冻结范围

区分“原 NavDP 真实转身后规划”与 X-NavDP 后方 PointGoal / 倒车组合，
并检查只换 X MPC 是否足够。不是新论文主方法或正式 SR，也不修改真机。
不新增门控、不调 CEC 阈值、不恢复 NavMesh 运动投影。

沿用 4 个已消费历史，顺序 gxdoqLR6rwA、pLe4wQe7qrG、yqstnuAEVhm、mJXqzFtmKg4。
源 `.diagnostics/shared_online_role_pair_natural_heading_v1_smoke_20260814/manifest.json`。
共享 A 仍是旧 actual-online metric A；查询才使用 LingBot 单目深度。

## 三臂

1. `cec_aligned / mpc`：原 mixed NavDP + 已测首次后向真实转身 + base MPC。
2. `cec / xmpc`：原 mixed NavDP，无额外转身，采用发布的 X MPC。
3. `cec_x / xmpc`：官方 X PointGoal posttrain actor + 官方 X MPC，无额外转身。

第二臂用于检查只有跟踪器改变时，零参考是否依然停滞。
它与旧同池 `cec / mpc` 一起解释；不把第一臂与第二臂差异称为纯 MPC 效应。

CEC 拒绝时，各臂仍运行相同 native ImageGoal/base MPC；X 活跃时也保持 native
短期 FIFO 的观测更新，不重复采样。不会向 runtime 传 Novel/Revisit role。

## 发布实现与公平边界

- X 源 commit：`878740a2011856d0e3782dd6ccd880fd2eccd70f`；原 posttrain 权重不改。
- X wheeled actor 8 candidates、官方 Q/RTC；base mixed 16 candidates。
- X RTC 使用当前模拟器位姿作为理想里程计，且会使用自身的 stuck/continuity 逻辑。
  不读取真实目标、最短路或障碍地图。其额外状态用途必须披露；不是单换 MPC 因果实验。
- X MPC 保留官方 N=30、ref_gap=3、T=0.1 和先加原点/从 control[0] 顺序执行的方式。
  每次新 NavDP 规划取前 8 个命令；base MPC 仍按旧实现每命令求解并取 control[1]。
- 共同线速度幅度 0.376 m/s、角速度幅度 π/4 rad/s、8 命令重规划、总预算 600。
  X 允许负速度；reference desired_v 同步为 0.376。公开算法未改，不冒充原作者完整配置复现。
- X 深度接口新增私有 RGB-only 适配：消费相同 LingBot frame/token 深度；不解码上传的 GT 深度。
- 所有臂相同 Bullet 圆柱（半径 0.30 m、高 1.50 m）、重力/接触/2 秒零命令落地。
- 无 NavMesh 运动查询；成功用 GT 平面距离 <1 m，仅供 evaluator；不验证自主 STOP。
- 倾斜达 5° 为代理失效，保留截断轨迹，不删样本后宣称更高 SR。
- 推理等待期间物理暂停；可记录推理/求解耗时，不用视频速度冒充实时部署延迟。

## 执行顺序与通过条件

先 CPU 坐标/深度传输测试及真实 acados 的零、前、后参考三种原语，再启动已有四历史三臂。
第一历史承担同协议的初次集成检查，不选成功病例、不因 SR 调参。
记录所有候选/证书/深度、X RTC 来源、MPC 控制队列、实际位移和接触。
独立复算到达与路程，核对初始 RGB/证书方向；旧转身对照尽量逐动作复现。
如遇依赖/接口错误，保留失败目录，修复后新目录重启，不称之为策略失败。

本轮只改隔离诊断 runner/adapter/tests；生产策略、论文、机器人和 HPC 不动。

## 集成修复记录（不改变实验设计）

首个 v1 中前两臂完成；X 臂在导航前被共享评测器的旧 controller 固定标签限制拒绝。
进一步检查发现 certificate 分支也固定请求 mixed endpoint，旧 X 分支仅接在旧 geometry router。
所以隔离诊断必须同时做两项适配：允许 consumed X 标签、将**已经通过原 certificate**
的 mixed 请求交给 X PointGoal；其余检索/证书/2.5 m 数学和验证均不变。
正式主文件不改，改动后的运行时副本与 source snapshot 随结果保存。
增加八命令 X-only 集成 smoke，再在新目录重启原冻结三臂；v1 不删除、不算 X 策略失败。
