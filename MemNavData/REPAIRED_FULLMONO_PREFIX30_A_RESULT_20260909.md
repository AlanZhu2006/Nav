# 修复版完整单目：首前缀实际 A 采集结果

2026-09-09 16:09（北京时间）。本文件只报告源 A，不报告尚未运行的查询成绩。

## 完整结果

- 原前30个scene位置包含28个非空scene，每个4条源任务，共112条。
- 实际 mono-native A：**88/112 = 78.57%**；24条未到达，全部保留。
- 112条都有动作后终点、实际路径与SPL；28份独立verifier全部通过。
- 28份归档均正常完成，全部成员回读哈希通过；归档总量3,225,415,633 bytes。
- 本次核对完整summary，而非从运行中的进度日志估算。

补充只读核对：112/112条的评分目标XZ与源任务A元数据完全一致。到达仍按原定义
使用平面距离；已到达样本的末端与目标高度差最大约0.294m，不能把该SR改称
三维距离或测地线到达率。本次没有修改成功标签或到达阈值。

| 原scene rank | scene | 到达 / 源A |
|---:|---|---:|
| 0 | rJhMRvNn4DS | 2/4 |
| 1 | 6D36GQHuP8H | 3/4 |
| 2 | nrA1tAA17Yp | 3/4 |
| 3 | jgPBycuV1Jq | 1/4 |
| 4 | BFRyYbPCCPE | 2/4 |
| 5 | X7gTkoDHViv | 4/4 |
| 6 | 5jp3fCRSRjc | 2/4 |
| 7 | dHwjuKfkRUR | 4/4 |
| 8 | AWUFxHEyV3T | 3/4 |
| 9 | 7GAhQPFzMot | 4/4 |
| 10 | vBMLrTe4uLA | 2/4 |
| 12 | z9YwN9M8FpG | 3/4 |
| 13 | SByzJLxpRGn | 3/4 |
| 14 | hkr2MGpHD6B | 3/4 |
| 16 | LNg5mXe1BDj | 4/4 |
| 17 | W7k2QWzBrFY | 4/4 |
| 18 | YRUkbU5xsYj | 3/4 |
| 19 | T6nG3E2Uui9 | 4/4 |
| 20 | SiKqEZx7Ejt | 2/4 |
| 21 | c5eTyR3Rxyh | 4/4 |
| 22 | FnSn2KSrALj | 3/4 |
| 23 | L53DsRRk4Ch | 4/4 |
| 24 | cYkrGrCg2kB | 4/4 |
| 25 | 3t8DB4Uzvkt | 4/4 |
| 26 | RJaJt8UjXav | 2/4 |
| 27 | b28CWbpQvor | 3/4 |
| 28 | QHhQZWdMpGJ | 4/4 |
| 29 | fsQtJ8t3nTf | 4/4 |
| 合计 | 28 scenes | **88/112** |

## 运行与来源

rank0保留已经完成的开发场景 `17253177_0`，不重复采集；其0个合法role pair也保留。
其余108条由 `17253441` 的27个元素采集，全部 `COMPLETED / 0:0`。
任务采用A100原环境、冻结的模型与修复后的输入/执行链；不是训练。

- 源计划SHA：`ba1a12273bf9eeef5024166c4bfa28037ee6076c695d2543959a7e169adafba7`。
- 运行底座：`repaired_fullmono_c8cf8c60e7efd55f`。
- 采集入口：`repaired_actual_a_3e90959723d4b66b`。
- expert资产只提供原起点与Goal-A图；后续历史来自本次实际A执行。
- 新版采用正确RGB、源栅格单目深度、bounded pursuit与标准Habitat碰撞响应；
  不使用旧的自定义snap落点和30%短步重试。计分仍使用仿真GT距离，不是自主视觉STOP。
- 这是已消费HM3D源任务上的新版核验，不是未见scene的独立确认。

原两历史2/2构造门未通过，以及后续完整场景启动条件的修订，均保留在
`REPAIRED_FULLMONO_PREFIX_CONTINUATION_20260909.md`；本结果不改写这些历史。

## 下一阶段与解释边界

构造数组 `17253442` 已开始处理实际A历史，CPU前缀检查 `17253701` 已接好依赖。
只有完整前缀构造达到24 histories / 15 scenes，才封存全部合法role pair并运行
native / raw-fixed / GEM三臂。规模不足时只按原预定义前缀继续，不看query结果选样。

88/112不是GEM的查询SR，不是mixed-role SR，也不证明修复提升或降低了旧版SR。
它说明新输入/执行版本的实际A阶段已完整跑通，后续查询不再需要复用旧执行版A。
正式query增益、构造损耗与新完整系统的论文口径，仍待后续阶段。

远端根目录：
`/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_fullmono_actual_a_20260909/prefix_series_after_scene0_20260909/collection/`

每scene保存 `summary.json`、`independent_verification.json`、`archive_receipt.json`
以及完整 `artifacts.tar.gz`。小收据回传本机：
`.diagnostics/repaired_fullmono_design_20260909/prefix30_progress/collection/`。

本轮未修改论文TeX、Abstract/Introduction或已有表格数字，未commit/push。
