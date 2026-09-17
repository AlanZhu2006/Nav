# Table II 共同在线目标：双臂连续本机首测

2026-09-11，运行前声明。用户已同意逐段共同发题。

## 固定开发来源

- 原声明 source index 0，`1LXtFkjw3qL/episode_0000`，序列 NRR。
- 这个来源之前已用于接口开发，已知 GEM 可以完成某组 NRR；本次不是盲测或扩样。
- A 仍为此前同一前向目标。B/C 使用新的、对称共同采样器，不按此前 B/C 成绩挑选目标。
- Native/GEM 各自从空历史实际执行；只在 episode 开始初始化一次。失败后该臂结束，
  另一臂继续；原起始分母不补换。没有合规后继目标时另报构造终止。

## 不变项和本轮改动

模型、安装高度、first40、strict certificate（包括面积条件）、2.5 m residual、
RGB/depth 修复、bounded executor、600 ticks/leg 和 horizon 8 全部不变。
役别只供 evaluator 发题；模型没有接收 Novel/Revisit 标签。

改动只在实验编排：

1. 两臂保持各自持续运行的服务与 Habitat 末态；一臂等候时没有新动作或观测。
2. 共同 Novel 对每个存活臂检查原前向、距离、当前/全历史共视规则。
3. Revisit 在每个存活臂中分别核对有效源帧、0.2–0.8 m / 12–45° 扰动与 MAE、共视。
   提案来自去重后的实际历史帧，按不依赖方法标签的内容地址排序，不固定 native 优先。
4. 每个共同距离档保留首个合规目标，候选顺序不使用导航/critic 结果。距离档地址取
   存活臂距离的均值；每臂自身距离及方向仍完整记录，不新增“双方必须同一距离档”门槛。
   本机单例不能证明正式总体的 A/B/C 距离已经配平，规模化前仍需总体分配器。
5. 候选预算仍为 Novel 最多 10000 个空间提案、每距离档最多 12 个视觉候选；
   Revisit 最多 384 个视觉候选。新增共同约束及对称提案会改变所选 B/C，不能与旧单臂
   选图试验作方法 SR 横减；在线方法本身未改变。

## 显存和可重复性

四个私有服务（每臂 LingBot/GEM + NavDP）使用同一本机 GPU，分段顺序交替执行。
它们是独立持续进程，不宣称同一模型进程共享 state。每段结束后仅调用
`torch.cuda.empty_cache()` 释放闲置 allocator 块，不卸载 live tensors、不清 KV、不 reset。
接口记录释放前后的 allocated/reserved bytes；若 live allocation 改变，终止检查。
本机另有不属于本任务的进程，绝不关闭它们；显存不足将按基础设施问题记录，不能改分。

核对 no-takeover A 的实际 RGB/pose 是否跨两臂完全一致，核对后续共同目标 SHA、各自
真实历史、唯一尺度收据和失败后的未尝试状态。技术验收不要求导航成功。

代码：`table2_common_goals.py`、`table2_continuous_paired.py`。
输出：`.diagnostics/table2_continuous_common_local_20260911_v1/`。
此首测完成前不提交正式 Table II，不改论文数字或旧封存包。

## 本机运行与资源结论（2026-09-11 23:15 已实测）

四个私有服务均启动，native A 成功；保留 native 的实际状态后，GEM A 在 LingBot
深度头计算中 OOM，整个配对中止。这是基础设施失败，没有完整配对 SR。

native A 结束后，LingBot allocated 为 14,342,655,488 bytes；释放闲置块使 reserved 从
24,213,716,992 降到 14,751,367,168 bytes，allocated 不变。故不是“没有清缓存”。
OOM 当时待分配 66 MiB，卡上只剩约 51 MiB。其他常驻进程及显示占用合计约
13,827 MiB（13.5 GiB），未停止它们。
失败日志、native A 和 GEM partial 均保留，全部本任务私有进程已退出。

下一步仅把这个技术首例迁到已验证的 A100 80 GB，同一 GPU 上保留两臂状态；
只申请一个 1 小时作业，不发正式数组。先检查 exact container 导入、CLI、模型和资产；
跑完保存完整归档，再独立复核。上一臂导航失败、episode 结束后可以释放它的服务，
但任何仍存活的臂都不清除状态。

## HPC 执行收据（23:39 CST）

原容器 CPU 预检通过后，已实际提交一个作业 `17374795`；当前在 `a100_tandon / ga016`
运行，1 GPU、12 CPU、128 GB、1 小时时限。只搬迁上述原 source 0 的 NRR 技术首例，
未追加来源或正式数组。源码 receipt 前缀 `150243d7f415a423`；完整参数和路径记录在
`TABLE2_COMMON_GATE_SUBMISSION_20260911.json`。首次实际导航结果尚未产生。
