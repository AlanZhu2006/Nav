# 修复版完整单目：独立构造可行性验证

2026-09-09；在本轮重新渲染和任何新 query rollout 之前固定。
范围：两条已使用的新版 mono-A 历史，仅诊断构造；不是 fresh confirmation，
不替换此前固定方向 gate 的零配对，也不修改正在运行的共视总体。

## 研究问题与最小变化

旧 gate 的六条 A 中五条成功，但三条成功历史的合规源帧离终点不足 2 m；
另两条有 standard Revisit，却没有完整的固定朝向 Novel/Revisit 配对。
本次只检验：固定每条历史的单一 Novel 方向层是否额外丢弃了有用的历史。

对 `rJhMRvNn4DS/episode_0000` 和 `jgPBycuV1Jq/episode_0000`：

1. 从原实际 A 的物化 RGB/trace 重放；不重新采集、不输入 expert 轨迹。
2. 按实际 color/depth projection matrix 建立 K，重新渲染浮点深度；每帧 RGB
   与原 trace 的 JPEG SHA 对照。GT 深度和位置只用于离线目标构造与标注。
3. standard Revisit 沿用 frame≥39、尾部排除16帧、stride8、2–9m、
   受控非重复位姿变化，以及最大历史共视 [0.55,0.90] 等原条件。
4. Novel 分别在 front/side/rear 三个方向层各搜索至多5000次；每层一次，
   新 RNG namespace 为 `repaired_fullmono_available_direction_20260909`。
   每个目标方向由独立于模型的原八方向世界 yaw 规则确定。
   距离2–9m、同层容差0.20m、净空≥0.30m、与Revisit间距≥1m、
   对全部实际A历史的最大共视严格<0.10，均不放宽。
5. 保存三层所有成功候选和拒绝计数；若有多个合法Novel，以固定
   SHA256(`repaired_fullmono_choose_20260909/scene/episode/stratum`)最小值选择一个。
   不以DINO、CEC、critic、导航SR或与Revisit方向的一致性选择。

相比旧gate，这里明确改变了Novel方向可用性规则与离线K/深度标注，且最大搜索
预算为3×5000，不假称原协议原样补跑。不移除旧版科学损耗，也不把新旧query混算。
支持区间是评测构造约束，不是对CEC阈值的调整。

## 分析与后续

- 报告每条历史的旧方向、三方向空间/视觉损耗、合法方向集合、标准Revisit及选中Novel。
- 原A已被选择且其集成结果已知，所以本轮永远只作为开发构造验证。
- 若两条都能构造，下一步是独立绑定源任务范围的完整单目采集与三臂评测；
  本作业没有自动下游query任务。
- 若不能构造，不继续自动增加候选、旋转目标或降低共视/距离门槛；保留失败原因。
- 原196条任务的只读清点表明：49个非空scene，52/46/30/68条A分别位于
  [3,4)/[4,5)/[5,6)/[6,9)m。此数是任务起终点测地距离，不是实际A路径、
  expert长度或Revisit距离；本次没有据此筛掉短任务。

## 运行约束

本机先运行源总体清点、真实NavMesh空间诊断及CPU单测，再在HPC已有两scene上渲染。
使用共享 `alantorch` / `yz11502`、已核验A100与原容器；不安装依赖。
1 GPU、4 CPU、16GB、1小时上限；不启动NavDP/LingBot服务，不申请训练。
模型/动作栈仍绑定 `c8cf8c60e7efd55f`，本次只增加独立构造入口。
保存紧凑收据、入选图像/浮点深度，不复制逐帧历史、不删除原输出。
