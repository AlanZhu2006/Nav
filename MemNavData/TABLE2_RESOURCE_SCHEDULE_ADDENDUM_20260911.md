# Table II 调度补充：A 数组并发 2 → 4

时间：2026-09-11 16:18 CST。

这是原冻结协议资源段的显式调度修订，不是构造或方法修订。
不覆盖原协议、plan、首次提交收据或已经完成的结果。

## 依据与动作

- 实查 `gpu48` 的单用户 GPU 上限为 16。
- 实查 `a100_tandon` 分区 QOS 的 GPU 总量上限为 60，实际调度仍受其约束。
- 当前数组 `17357681` 除 QOS 等待外，也出现 `JobArrayTaskLimit`，其原并发为 2。
- 对**同一数组**执行 `scontrol update JobId=17357681 ArrayTaskThrottle=4`。
- 随后 `scontrol show job 17357681` 显示 `ArrayTaskThrottle=4`，待运行范围显示 `%4`。

命令对正在退出/已结束的成员报告了 `Job has already finished` 和 `Unspecified error`；
以随后读取的数组实际状态为依据确认并发已修改，未重复提交或重新运行任何样本。
设置为 4 不等于立即获得 4 张卡；若项目 GPU 配额已满，继续等待集群调度。

## 保持不变

- 同一 144-source A 总体、目标图、seed、顺序身份及已封存候选；
- A100 分区，每项 1 GPU、10 CPU、96 GB、1 小时上限；
- frozen NavDP / GEM / 单目高度标定 / 修复版执行器；
- B/C 单个查询的 native/GEM 仍在同一任务、同机配对；
- 全量 A 完成后再选择 B、全量 B 后再按 50/50 选择 C；
- 不因本次 A 成败更换目标，不新增样本，不改任何成功或授权阈值。

本次只改变 A 数组的最大同时运行项数，不增加声明的任务总量。
不同任务本来就可能调度到不同 A100 节点，不声称跨节点导航逐位确定。
B/C 的提交模板暂仍为原并发 2，后续如作同类调度修订需另外记录。

远端收据：
`table2_full_rerun_20260911_v2/resource_schedule_20260911T0818Z.json`。

原 SOURCE_BUNDLE / plan 的 SHA-256 保持不变，完整运行后继续按原始轨迹和分支身份复算。
