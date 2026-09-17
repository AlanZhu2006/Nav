# 2026-09-08 仿真执行链修复：今晚总账

## 一句话结论

问题不只是 NavMesh 投影，也不等于 pure pursuit 这个算法不能用。我们确认并隔离了
**输入表示、PointGoal 输入域、轨迹跟踪与环境碰撞之间的接口错误**；最小候选修复已在本机
完成真实模型配对验证，但没有把 N=4 机制结果替换成论文主结果。

本轮设定的本机审计/修复目标已完成；新版正式评测、生产默认切换和论文更新是后续任务。
随后追加的 **actual mono-A → mixed-role 本机完整集成也已完成**，详见第 8 节；
第 9 节已提交 HPC 两源迁移 gate，正式核心批次仍未提交。
本机阶段未操作 HPC；后续迁移按共享 SSH 手册执行。全程未操作真机或论文，未 commit/push，
保留了已有 dirty worktree 和所有旧结果。

## 1. 最小候选架构

```text
同一因果 RGB 流
  ├─ RGB 通道正确、LingBot 单目深度恢复到原 RGB 像素栅格
  └─ 原有 raw memory / CEC → 原有 fixed 2.5 m PointGoal 或不接管
                         ↓
        若已发出的 PointGoal 在 frozen base 的前向域之外：
        按收到的方向真实转身，每步更新 RGB，转后重新规划
                         ↓
                frozen NavDP 参考轨迹
                         ↓
       有限步长 pursuit：零参考保持，短参考不越过目标参考
                         ↓
         一次标准 Habitat try_step 环境碰撞响应
```

没有增加第三个专家、训练门、critic 门或遇失败才搜索的新策略。后向适配由输入坐标定义触发，
方向只能来自已收到的 PointGoal；raw 与 CEC 使用完全相同的接口。
它确实执行显式 yaw 动作，所以只能说 NavDP **权重冻结**，不能说所有运动都由 diffusion 输出。
CEC 原有拒绝/不接管和上游 NavDP 原有低分搜索仍存在，不宣称整个系统没有 fallback。

保持 Habitat 运动学代理、原角速度/步长/动作预算、现有相机、原证书和原到达半径。
标准 `try_step` 仍使用 NavMesh，低层跟踪仍使用理想位姿；模型不接收 GT 目标或 GT 路线，
不等于仿真环境也没有 GT，更不等于验证了 Go2 全身碰撞安全。

## 2. 已确认的问题

1. 旧 pursuit 对零 XY 参考仍产生平移，短参考也可能被越过。零输出不能被当成移动许可或到达。
2. 自定义 `snap_point` 与 30% 缩步重试改变了动作落点；应与标准环境碰撞分开。
3. 当前 base NavDP 和 ViNT 的真实图像请求链存在 RGB/BGR 交换；仅观察录制 RGB 视频看不出来。
4. LingBot 518×518 pad 深度直接进入 NavDP，深度预测覆盖了 RGB 的 padding 区；
   真实帧中该区域 69.15% 像素收到非零深度。逆变换修正的是像素对应，不是调尺度。
5. base PointGoal encoder 裁掉负 forward；正确的后方 bearing 可能得到非低分零 XY 输出。
   旧 pursuit 的自行推进掩盖了这个接口问题。

额外核对相机 FY 元数据与实际投影约差 1.16%。四历史八查询完整复算没有改变这批构造角色；
这不是本批零轨迹根因，也没有据此修改既有数据。后续生成器应从真实相机矩阵导出 K。

旧 HM3D full-mono 的 168 个原始 query 已只读追溯：CEC-Revisit 17/28 出现
“零候选终点统计、critic 非低分、下一动作仍移动”的暴露，native-Revisit 为 0/28。
旧日志缺完整候选，**不能认定这 17 个成功是假成功，也不能预测修复后损失多少 SR**。
但“大家共用执行器所以偏差抵消”并不成立：ImageGoal 和后方 PointGoal 的触发条件不同。

## 3. 已完成的本机实验

| 独立阶段 | 完成情况 | 主要结果 | 能说明什么 |
|---|---|---|---|
| 执行/颜色桥接 | 4 histories × 6 arms，24/24，600 tick 上限 | native 1/4→0/4；CEC 3/4→2/4 | 旧执行确实影响局部结局；RGB 修正未再改变这四例成功标签 |
| 深度栅格桥接 | 2 histories × 4 arms，8/8，600 tick 上限 | native 0/2→1/2；CEC 1/2→1/2 | 输入对齐能改变真实闭环；不能外推总体增益 |
| 后向目标 off/on | 4 histories × 4 arms，16/16，600 tick 上限 | raw 2/4→4/4；CEC 2/4→4/4；各 +2/−0，p=0.5 | 恢复两条停滞病例；是共有控制接口收益，不是 CEC 独有收益 |
| 非接管真实请求 | 2 Novel histories × 2 arms，4/4，仅 64 tick 前缀 | 两条 CEC 均未接管；native/CEC 输入、完整输出与动作完全一致 | 新适配未改变这两条非接管前缀；不是完整 Novel SR |

三组完整实验和一组前缀实验均通过独立 verifier，全部成功/失败第一视角视频已保存。
这些阶段复用了同一批消耗型历史，**不能把臂数或动作数相加成独立样本量**。
所有 A 都是旧的真实 metric-NavDP-A，而不是 expert-A；新 query 使用单目深度。
因此这批机制实验不重新证明 full-mono-A 或外部泛化。

统一接口/回归入口：`bash MemNavData/run_habitat_contract_preflight.sh`。
最终已保存结果为 **146 passed / 12 warnings**；警告是依赖 Pyparsing 弃用，未隐藏。
单元通过不替代真实闭环；源码在每次独立运行期间均未改变。

## 4. 还没有解决、也没有藏起来的问题

- **纯旋转漂移：**真实平移为 0，LingBot 高度尺度下仍估出 0.095–0.373 m 平移，
  yaw 误差 1.46–6.58°。四例转后仍能到达，不意味着长程也稳定；高度先验不消除漂移。
- **成功口径：**仍由 evaluator 的 GT 平面距离 <1 m 计分，非自主视觉 STOP；不检查目标朝向。
  原地转向消耗动作但不计平移路程，SPL=1 不等于最快或无转向代价。
- **历史版本绑定：**当前源码和部分主线 revision 已审查，完整 HPC bundle 尚未逐表绑定。
- **ViNT：**实际通道链已测出问题，但没有本轮新 ViNT SR；不能替它宣称修复后效果。
- **新主系统尚未正式确认：**生产默认没有切换，修复版 actual mono-A mixed-role 的 HPC 配对尚未提交。
  第 8 节本机集成使用新 A，但只有一条可构造历史；不把它当作整篇论文重跑完成。

## 5. 论文需要重跑多少

**若论文所有闭环表都要代表统一的修复版，多数闭环比较确实需要补跑。**
不只补失败案例、不只跑 CEC，也不能拿新版 mono 横减旧版 metric/native。

最省算力的顺序：

1. 冻结本机验证过的共有接口版本，先采集新版 actual mono-A，再构造同规则的 mixed-role 查询。
2. 先完成一个 native / raw / CEC 的核心配对，报告 A 构造率和查询分母变化。
3. 根据这批结果再统一补跨 controller/dataset、连续目标、深度和授权的必要闭环表。
4. 保留未受影响的离线 retrieval/PnP/certificate 证据；不重跑每个历史支线或已经淘汰的机制。

旧结果保留版本标签和原始记录，不删除、不重贴新版本名称。详细表格依赖见下方重跑清单。

## 6. 为什么这次必须审输入和执行，而不只是继续加哈希

哈希证明运行的是同一批字节，不证明字节被正确解释；严格配对也不证明接口对不同条件的偏差相同。
颜色错误不报 shape 错，padding 深度仍是合法数值，零轨迹经错误 tracker 甚至可以产生成功。
因此增加的是少量语义测试：真实彩色编码链、零/短轨迹、真实 pad 对应、模型 GT 权限边界、
转身新观测与重规划、末动作后位移。它们在提交长任务前运行，不在运行时继续堆门控。

不能保证从此不存在缺陷；可以让这些已识别问题有可复现的反例、修复测试和全量原始收据。

## 7. 文档与证据入口

- [完整接口审计](HABITAT_MINIMAL_EXECUTION_AUDIT_20260908.md)
- [执行/颜色六臂结果](HABITAT_MINIMAL_EXECUTION_BRIDGE_RESULT_20260908.md)
- [深度对齐结果](HABITAT_DEPTH_RASTER_BRIDGE_RESULT_20260908.md)
- [后向 PointGoal 完整结果](HABITAT_FRONT_GOAL_BRIDGE_RESULT_20260908.md)
- [非接管前缀协议与结果](HABITAT_NATIVE_REQUEST_EQUIVALENCE_PROTOCOL_20260908.md)
- [各论文表的重跑范围](HABITAT_REPAIR_RERUN_SCOPE_20260908.md)

全部原始证据父目录：

```
/home/asus/Research/Nav-graph-blind/.diagnostics/habitat_minimal_repair_20260908/
```

其中 `bridge_v1`、`depth_raster_bridge_v1`、`front_goal_bridge_v1`、`native_request_smoke_v1`
各自包含 `summary.json`、`independent_verification.json`、`source_snapshot/`、
`evaluation/` 和 `first_person/`。不同版本结果保持分离。

前一阶段私有模型进程已退出；当时用户驻留 18888/8888 服务未动，读数约 6.8 GB 显存、GPU 0%。
未停止其他工作区的任务，未修改既有生产 evaluator 或用户未提交的源文件。

## 8. 继续执行：新 actual mono-A 全链集成（同日晚追加）

固定两条原 source，重新执行 mono-A，随后构造和运行六个 mixed-role query arm：

- A：1/2 成功；仅 1 个历史可构造，另一 A 的失败完整保留，没有用旧 A 补齐。
- 同一新历史的 Novel：native 1/1、raw 0/1、CEC 1/1。
- 同一新历史的 Revisit：native 0/1、raw 1/1、CEC 1/1。
- Novel CEC 全程不接管，61 次完整候选/critic/depth 与 481 个动作精确等价 native。
- 2 A + 6 query 全部独立复算通过；301 次深度输入检查；8 段完整第一视角视频已导出。
- 首次 verifier 不接受失败 A 的空清单，收尾脚本已修复并保留原失败日志；导航未重跑。
  新增空构造测试后统一预检为 162 passed / 12 依赖弃用警告。

这是 **1 个历史/场景上的完整集成**，不能拿臂数充当样本量，也不证明所有旧论文增益已在新版本恢复。
新 A 的一条失败仍是非低分零参考后的停滞，未调半径或改策略救该例。
下一步是同版本 HM3D 核心配对的可迁移 bundle 与实际环境预检，不先重跑全部历史支线。

- [完整结果与证据](REPAIRED_FULLMONO_LOCAL_RESULT_20260908.md)
- [HM3D 核心评测准备](REPAIRED_HM3D_CORE_EVAL_PLAN_20260908.md)
- 原始根目录：`.diagnostics/repaired_fullmono_local_20260908/e2e_v1/`

本次私有 21710/21711 服务已退出；其他工作区在用的 GPU 服务未动。本次没有 HPC 提交、生产切换或 push。

## 9. 继续执行：HPC 两源迁移 gate 已提交

北京时间 22:16:38 提交 **17188986**：固定 HM3D 两源重新 mono-A → 实际历史构造 →
native/raw/CEC 三臂配对与独立复算、全量视频。资源 1 GPU、10 CPU、96 GB、1 小时；
没有提交正式全量批次或自动下游任务。

165 项本机回归通过；实际远端三组导入、七组 CLI、源数据、原依赖和 Slurm test-only 均通过。
登录节点 `/tmp` 满导致的 callback MemoryError、Habitat 缺 OpenCV 两项迁移错误均在 GPU
提交前处理，修复只影响任务临时目录和独立依赖路径，没有改共享 conda。

提交时为 **PENDING (QOSGrpGRES)**；随后正常完成，ga043，7 分 56 秒。
最终 2 条 A（1 成功、1 失败）、0 个合法 role-pair、0 query arms。成功 A 有 Revisit
候选但没有合法 Natural Novel；独立复算和两段视频完成，不等于完整查询 gate 已通过。
详情及 5,000 次构造损耗分解已写入下方最新状态文档；旧输出全部保留。

按父清单固定追加 rank 2–5 四条 source 已于北京时间 23:59:25 提交 **17196996**，
array 0–1%2、每项 1 GPU / 1 小时，各分片先 A/构造后 query。
23:59:35 提交快照为 PENDING / QOSGrpGRES；9 月 9 日复核已结束：
shard 0 两条 A 成功但无 role-pair，0 query；shard 1 首条 A 在 56 步后 SIGABRT。
不放松共视、方向或距离规则，不重跑旧失败来替换旧结果。
追加源码仅改变固定分片/父编号传递及相应校验，不改变 NavDP、CEC 或执行修复；
167 项本机回归与远端实际环境预检通过。

- [最新迁移/提交/状态文档](REPAIRED_HM3D_GATE_STATUS_20260908.md)
- [机器可读提交收据](REPAIRED_HM3D_GATE_SUBMISSION_20260908.json)
- [四源追加状态与下一步](REPAIRED_HM3D_EXTENSION_STATUS_20260908.md)
- [四源追加提交收据](REPAIRED_HM3D_EXTENSION_SUBMISSION_20260908.json)

## 10. 9 月 9 日：构造损耗本机复现、A100 exact retry

本机 CPU 重放保存的 A trace / execution.navmesh，不调用模型、不渲染，
精确复现了两条源帧列表和 5000 次 Novel 几何筛选：短 A 合规历史距终点最多 1.104 m；
另一 A 的空间有效 Novel 候选 558 个 rear / 1 个 front，原冻结层 front，
最终 1 个共视检查也被拒绝。这是构造问题，尚不是 CEC 查询结果。

仅失败分片补交 **17201733_1**，原 sealed bundle 不变、A100、1 小时，
新增 faulthandler；00:43:49 在 ga015 启动，00:48:48 正常完成。
两条 A 均成功、verifier=true、视频完成；合规源帧距终点最多 0.976 / 1.348 m，
同样不足 2 m。四源追加最终 A=4/4、query=0，停止自动扩样；当前没有在跑任务。
原已完成分片不重跑，失败日志不覆盖。
新增 CPU 审计测试 3 项与现有 integration 回归 21 项通过。

- [详细审计与证据](REPAIRED_HM3D_CONSTRUCTION_RUNTIME_AUDIT_20260909.md)
- [A100 补跑协议](REPAIRED_HM3D_A100_RETRY_PROTOCOL_20260909.md)
- [A100 提交收据](REPAIRED_HM3D_A100_RETRY_SUBMISSION_20260909.json)
- [最终四源结果绑定](REPAIRED_HM3D_EXTENSION_RESULT_20260909.json)

## 11. 9 月 9 日：已有合法 Revisit 的独立集成已完成

用户同意后，复用 rJh / jgP 两条 actual mono-A 和原已选标准 Revisit，
独立运行 native / raw_fixed / CEC 六条查询；不重跑 A、不放宽构造条件，
不把它们伪装成 Natural Novel/Revisit pair 或并入原 mixed-role 分母。
**17208480** 于北京时间 01:13:51–01:27:44 在 ga008 / A100 完成，用时 13 分 53 秒。
03:40 再读取最终结果：native 1/2，raw fixed 2/2，CEC 2/2；独立复算通过，六条第一视角录像齐全。
N=2 只支持集成可运行，不能替换主表或证明 CEC 优于 raw。81 本机测试、三臂本机与远端 CLI、
6 项远端测试通过；runtime 复用 c8cf8c60e7efd55f，新 addon 为 b90fce8509fd376e。

- [独立 Revisit 集成最新状态](REPAIRED_HM3D_REVISIT_INTEGRATION_STATUS_20260909.md)
- [协议](REPAIRED_HM3D_REVISIT_INTEGRATION_PROTOCOL_20260909.md)
- [提交收据](REPAIRED_HM3D_REVISIT_INTEGRATION_SUBMISSION_20260909.json)

## 12. 9 月 9 日：按用户要求准备完整 0.1–1.0 共视分层

不是再调整证书，而是固定 28 条实际单目历史及各自物理目标，改变目标图 yaw，
构造 [0.1,0.3)、[0.3,0.5)、[0.5,0.7)、[0.7,0.9)、[0.9,1] 五档，
之后每个查询完整配对 native / raw fixed / CEC。使用修复后的 query 执行栈；不重采 A。

构造作业 **17222616** 已提交（A100，28 elements，最多 4 并发，单任务 1 h），
截至 03:40 尚因 QOSGrpGRES 排队。本阶段只生成目标与共视注释，没有新 SR。
旧 Natural Novel 作为单独对照，不能将原地换朝向得到的低共视自动叫 Novel。

- [设计与分析协议](HM3D_COVISIBILITY_REPAIRED_PROTOCOL_20260909.md)
- [构造程序](build_hm3d_covisibility_repaired.py)
