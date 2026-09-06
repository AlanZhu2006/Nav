# Final14 Table III：精确 SPL 补跑（2026-09-07 冻结）

## 目的与边界

保留论文 SPL。旧 210 条深度消融记录缺最后动作后的落点，旧路径计数还混用了指令步长。
本轮完整补跑 Table III，记录实际位置积分和终点。不是训练、不是 fresh confirmation，
不生成新的 Goal-A，不调阈值，不加入长程或 learned 新机制，不接触真机。

原始 CSV/日志、旧 SPL 上下界全部保留。新结果通过验证前不替换论文；之后使用新轮次
自身的 SR 与 SPL，不把旧 SR、新 SPL 或不同轮次的臂拼成一个配对结果。

## 固定 population 与五臂

- 原 natural-direction Final14：21 histories / 10 scenes。
- 每条原 metric-depth NavDP A 的真实 RGB history 重放，产生一 Novel、一 Revisit 查询。
- 42 queries × 5 arms = 210 rollouts，每条最多 600 actions。
- Arms：mono native、zero native、mono CEC、metric native、metric CEC。
- 每个 history 共享一对持续存活的模型 server；五臂次序按 history index 循环平衡。
- 相同 Goal-A / goal image / start / seed / 1 m 平面到达标准 / exec horizon 8。
- Novel/Revisit 标签仅用于分析，运行时隐藏。拒绝后检查 native 请求与物理轨迹等价。
- 候选、certificate、2.5 m residual、控制器权重及原连续执行器全部保持原配置。

Manifest SHA-256：`7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a`。
模型/执行器来源：`final14_mono_factorial_5690569a4373f2d2`；不打包工作区的新运行时代码。

## 唯一计量改动

冻结 evaluator 的 `run_policy_leg` 完成后读取其返回值，原对象原样返回。
保留原控制器内部路径计数和全部请求；独立记录 terminal position、yaw、实际路径与 SPL。
实际路径为逐动作平面位置积分，包含最后一段；转头和碰撞没有实际位移时不增加路径。
旧 serializer 输出保存在新目录的 `metric_commanded.csv`，精确计量写入 `metric.csv`。
`*_plans.json` 增加 `query_result`，评测用目标位置不会传给策略。
独立 verifier 不调用计分 helper，直接从 raw poses + endpoint 复算 SR、路径和 SPL。

## 提交顺序与资源

1. CPU preflight：实际容器/解释器 import、单元测试、权重和全部冻结 source/scene/trace 校验。
2. index 17：按 metadata 选择最长 prefix（364 observations），完整五臂/10 rollout；
   选择与新结果无关。这一条通过后计入最终 21 histories，不重复评测。
3. 其余 `0-16,18-20%4`：必须依赖 gate 成功，每 history 不拆臂。
4. 全量 summary + independent verification：等待前项结束，缺任一配对块即失败，
   不把 partial 结果重命名成完整结果。

GPU：`h100_tandon,a100_tandon`、1 GPU、10 CPU、72 GiB、每元素 1 小时，最大并发 4。
CPU：`cpu_short`、4 CPU、16 GiB、30 分钟。共享 SSH 按操作手册，不新建认证流程。
scratch 文件数配额接近上限；逐帧 server buffer 写节点临时目录，结束时归档为一个
`runtime.tar.gz` 并保存 SHA，科学日志与完整 poses 仍持久保存。失败不删除已有记录。

## 论文状态

Table III 暂保留 SPL bounds；Table II 已精确更正，不在本轮补跑范围内。
摘要、Introduction、标题不修改。再次运行可能出现 CUDA/硬件导致的 SR 变化，
必须完整报告，不以复现旧成功数作为通过条件。

提交状态和实际 Job ID 另见提交收据，不能仅凭本协议判断任务已在运行。
