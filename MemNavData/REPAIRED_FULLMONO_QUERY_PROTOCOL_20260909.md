# 新实际mono-A后的配对查询：执行协议

2026-09-09。此时新正式A/查询尚未运行；本文件是后续入口的设计，不是结果。
本次只完成当前修复版完整单目链条，不改变模型、certificate或控制参数。

## 总体与构造

- 使用 `core_source_plan.json`（SHA `ba1a12273bf9eeef5024166c4bfa28037ee6076c695d2543959a7e169adafba7`）。
  全部196个既有HM3D源任务按原scene/episode顺序，不能挑旧成功A。
- 复用冻结 `repaired_actual_a_3e90959723d4b66b` 采集入口。每条A是真实新版
  mono-native rollout；task的初始状态与Goal-A来自原任务，expert轨迹不成为memory。
- 每scene全部4条A和终点证据均保留。失败A、历史过短、无法生成两个角色的情况
  分开报告，不将未构造查询当成query失败，也不从不同总体乘出joint SR。
- 构造规则与两历史探针完全相同，详见
  `REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_PROTOCOL_20260909.md`：实测K、重渲染浮点深度，
  逐帧RGB对照实际A。GT只用于离线构造/支持标注和仿真评分。
- 每条合规历史有一个standard Revisit（最大合规帧共视0.55–0.90）和一个
  Novel（全部历史最大共视<0.10）。三方向分别搜索，固定hash选一个可用Novel；
  其它方向图像保留为构造诊断，不进入正式SR。
- 场景前缀30/36/42/48/54，只在全部A和构造收据完整后选择最小满足
  24 histories/15 scenes的前缀；到最大前缀仍不足则明确报告underpowered。
  选中前缀内全部合法历史都保留，不截到24条。
- 这些是已消费过的HM3D源任务：用于新版执行与全mono核验，不能称新的独立场景确认。

## 查询与运行

`seal_repaired_fullmono_queries.py` 读取完整构造收据，在任何query rollout前生成
不可覆盖的population，绑定实际A、图像、源计划、runtime与评测入口。未完整时不生成总体。

每个查询三臂：`native / raw_fixed / cec`（CEC即当前GEM），同一个私有GPU模型进程。
任务按固定hash排序，6种arm顺序循环平衡。Native和GEM没有接管时仍核对逐动作精确一致。
运行期不将Novel/Revisit、共视度、GT位置作为模型输入。

`repaired_fullmono_query_eval.py` 只增加实际A来源绑定，复用已封存共视评测的
rollout、模型服务和独立SR/SPL verifier。底层仍为
`bounded_rgb_source_depth_front_goal_v1`，runtime receipt `c8cf8c60e7efd55f`：
RGB、源栅格单目深度、bounded pursuit、标准Habitat try_step、后方目标物理转身后重规划。
不恢复旧落点snap/30%短步重试，不更改2.5m residual与certificate阈值。

查询的回调包装器使用正在运行的 `covis_eval_f284c23d7a98b0c2` 中的
`run_habitat_minimal_repair_local.py`，SHA
`0f1c56cc15ad0be9d2667d9024af14db2f25c5ca9a2865a2103e651efeddda0e`。
原c8底座尚无`query_main`参数，不能直接从底座导入该查询入口；模型与动作实现
仍由原底座提供。第一版入口已在远端CPU dry-run发现这个绑定问题，未运行query；
以新源码包修复，不覆盖旧封存包。

查询worker不重新采集A，因此自己的`new_a_rollouts=0`表示重放已单独采集的新A，
并不表示复用了旧执行版A。总体同时绑定A收据和运行版SHA，可独立区分二者。

## 指标和归档

- 主报同总体三臂Novel/Revisit及合计SR；配对gain/loss、exact McNemar、scene-cluster CI。
- 报精确实际位移SPL，包含最终动作后的坐标。转身角度、规划数、延迟为诊断。
- 单独报告全部源A的成功/损耗及合法role-pair比例。查询SR条件分母不能冒充196源的端到端joint。
- 独立reader复查身份、RGB/目标文件、完整支持曲线及选择规则；它不独立重渲染
  共视几何，不能把JSON/哈希检查叫第二次几何测量。
- A与query的dense日志留node-local，完整回读归档到scratch。构造阶段只长期保存
  必需的RGB历史、输入深度文件、trace与目标图。GT标注深度不进入在线模型。
- 构造1GPU/4CPU/24GB/1h；每查询三臂1GPU/10CPU/96GB/1h，A100原环境。
  先单scene完整采集和构造，再扩完整源前缀；总query人口封存后运行三臂。

本协议和测试通过不等于已完成GPU闭环。必须等实际A与构造入口完整运行，
再从封存总体选择一个固定任务验证三臂运行，失败不能静默丢弃或用旧版本替换。
