# 修复版 full-mono：完整源前缀与查询总体

2026-09-09 17:36，北京时间。A、构造和封存已完成；查询首项已通过，剩余89项已提交，尚无总体查询SR。

## 已完成的样本流

| 阶段 | 数量 |
|---|---:|
| 完整源前缀的实际mono-A | 112，28 scenes |
| A到达 | 88/112 |
| A未到达 | 24 |
| A到达但可用历史不足 | 2 |
| 历史可用但无standard Revisit | 30 |
| 有standard Revisit但无合规Novel | 11 |
| 最终合法role-pair histories | 45，23 scenes |
| 查询 | Novel 45 + Revisit 45 = 90 |
| 三臂导航总量 | 270 |

损耗按流水线顺序互斥记账：24 + 2 + 30 + 11 + 45 = 112。
无standard Revisit的历史没有继续将其归因为Novel缺失；不是说这些历史一定有合规Novel。
所有源成功/失败和无pair原因均保存，不将构造损耗记作query导航失败。

预定义首scene前缀30包含28个非空scene，完整结束后满足原定最低24 histories/15 scenes。
因此不启动后续36/42/48/54前缀；当前45个合法history全部保留，未截取最先得到的24个。
这些HM3D源场景此前已被使用，本实验检验新版全单目执行链，不称独立新场景确认。

## 实际查询分布

| 项目 | 最小值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| Novel：全部A历史最大共视 | 0 | 0 | 0.093923 |
| Revisit：合规A历史最大共视 | 0.716912 | 0.719975 | 0.755971 |
| Novel：A终点至查询目标测地距离/m | 2.0488 | 4.5144 | 8.9897 |
| Revisit：A终点至查询目标测地距离/m | 2.0140 | 2.8435 | 3.8650 |
| 入选A历史观测帧数 | 96 | 158 | 455 |

Revisit的冻结允许区间仍为0.55–0.90，实际集中在约0.72，来自原构造排序规则；
没有按模型结果调整共视。它是标准支持的完整系统评测，不代替另一个0.1–1共视分层实验。
Revisit共视只在frame≥39的合规历史计算；Novel条件覆盖全部历史。
两角色的距离分布并不相同，不能直接用Novel与Revisit的SR差推断角色本身的因果效应。

每个query内部native/raw/GEM共享同一真实新A历史、目标、起点与seed，六种arm顺序各15次。
源A成功率、条件查询SR、等比例角色合计SR分别报告，不相乘伪造源任务joint SR。

## 运行与核验

- A：`17253177_0`保留首scene；其余为`17253441`。28份A验证和归档回读均通过。
- 构造：`17253200_0`保留首scene；其余`17253442`全部完成。
- CPU封存：`17253701`完成，耗时8秒；封存时query导航为0。
- population SHA：`13d6827a05361c85b96f0196c76913bcff69439baa7999ab51bb9f035bfa9a1b`。
- 原容器实际目标预检：90/90输入绑定通过，没有执行导航。
- 本机独立读取28套scene收据、86份实际构造JSON并复核SHA；45个保留source身份与封存清单一致。
  这是独立文件/规则复算，不是再渲染一次共视几何；目标像素的远端输入核验另由实际loader执行。
- 首个固定任务：`17262832_0`，A100、1 GPU、10 CPU、96 GiB、1小时。
  按sealed index 0选取，不按成功率选取。已用6分39秒完成，三臂执行、独立验证及归档均通过。
  保留首项，其余89项提交为`17263230`，终局CPU汇总为`17263231`。
  首项导航成功或失败都保留，不设SR启动门。

只复用冻结模型与修复执行版：RGB、源图栅格单目深度、bounded pursuit、Habitat try_step、
后方PointGoal先物理转身再重规划；2.5m bearing与certificate阈值不变。
GT深度不作为模型输入；仿真仍有NavMesh碰撞、状态反馈与GT到达评分，不代表真机自主STOP。

## 路径

远端总体：
`/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_fullmono_actual_a_20260909/prefix_series_after_scene0_20260909/prefix_30_seal/population.json`

远端查询输出：
`/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_fullmono_actual_a_20260909/prefix_series_after_scene0_20260909/evaluation_cb812054dc470e53`

本机收据镜像：`.diagnostics/repaired_fullmono_design_20260909/prefix30_progress/`。
提交依据：[首项提交记录](REPAIRED_FULLMONO_QUERY_GATE_SUBMISSION_20260909.json)。
