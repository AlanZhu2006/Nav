# 修复版 HM3D：固定四源追加集成验证

2026-09-08；在运行本追加批次任何 A/query 前冻结。
这是初始两源 gate 的运行栈追加验证，不是正式论文确认，也不是按 SR 再挑四个场景。

## 为什么追加

初始 job 17188986 正常完成，2 A、0 query。唯一成功 A 可构造 standard/hard Revisit，
但在固定方向、目标朝向、距离、净空、楼层和低共视要求下，5,000 次 Novel 提议均未通过。
其中进入最终共视判定的 109 个提议全部被拒绝。另一 A 失败。
这是构造损耗；不修改旧记录，不放松 Novel 共视阈值，不把 Revisit 单独塞进 mixed-role 分母。

## 固定选择、预算和角色

复用 parent SHA `a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5`。
按父清单原顺序取接下来四个场景（从零计数 rank 2–5）各自首条 `episode_0000`。
未读取它们旧 A 结局、旧 query 结局或按场景 SR 排序。

| shard | parent scene rank | scene | A/query seed | Novel 方向层 |
|---|---:|---|---:|---|
| 0 | 2 | nrA1tAA17Yp | 2026082400 | side |
| 0 | 3 | jgPBycuV1Jq | 2026082500 | front |
| 1 | 4 | BFRyYbPCCPE | 2026082600 | rear |
| 1 | 5 | X7gTkoDHViv | 2026082700 | side |

方向层沿用 `assigned_direction_stratum(parent_scene_rank, 0)`；不能在每个分片把场景编号
重置为 0/1，从而悄悄改变构造方向和随机流。原始 gate 的 rank 0/1 不受此修正影响。
每个历史的 arm 顺序按 parent_scene_rank mod 3 轮转；所有 arm 共用同一 GPU/模型进程。

每分片先完成两条新 actual mono-A，再完成两条历史的构造，最后运行全部合法历史的
Natural Novel / standard Revisit × native/raw/CEC。两个分片可并行，但四条 source 和
全部构造规则在启动前已经固定，不能根据另一分片的 query 结果调整。
最多 4 A + 24 query rollout，全部计入损耗账；不因达到好结果提前停，也不自动补第五条 source。
A 失败或无法构造时保留其零查询结局，不拿旧 A 替换。

## 保持不变的科学运行栈

与首次修复版 HPC gate 同一模型权重、输入与执行代码：
RGB 修正、单目深度逆 padding、first40 相机高度尺度、bounded pursuit、一次标准 try_step、
raw/CEC 共有后方目标朝向适配、canonical reference depth、2.5 m residual。
600 tick、8-tick horizon、原证书阈值；不新增学习、搜索、恢复或控制门控。
构造 K 暂时仍保留旧定义；已量化 FY 约 1.16% 差异，不在本追加中同时改变。
运行时不读取 Novel/Revisit 标签；角色和 GT 只用于 task/evaluator 层，不能转交模型。
标准 NavMesh、理想底层状态、GT 平面距离 <1 m 成功仍存在，不宣称视觉自主 STOP 或真机安全。

## 资源、证据与判定

array 0–1%2；每项 1 GPU、10 CPU、96 GB、1 小时；h100_tandon/a100_tandon。
初始两源含启动/构造/复算/视频共 7 分 56 秒，两个 A 实际用时约 46/135 秒。
本追加可能运行更多 query，1 小时只是单项上限，不是完成时间承诺。
复用已通过实际 HPC 运行的 Singularity、两个绝对解释器、独立 OpenCV 和 node-local 临时目录。
不安装或升级共享环境，不覆盖 LD_LIBRARY_PATH，不把逐帧模型 buffer 写回 scratch。

全量保存 source/A/构造损耗、逐动作前后位姿、候选和 critic、深度、末步终点、独立 SR/SPL 与视频。
两个分片全部结束后统一报告，包括合法零历史。至少一个合法历史完成六个 query arms，
并通过 verifier/视频检查，才可以说 HPC 的完整 mixed-role query 链已实际跑通。
几条历史即使有增益也仍是内部小样本；不能替换论文主表，不能称正式统计确认。

不自动提交正式核心批次。若仍零历史或出现错误，先报告具体损耗/错误，再另行设计下一步。
