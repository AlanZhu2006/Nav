# Table I：三种冻结 controller 的修复版记忆对照

2026-09-10。在本轮导航输出产生前固定。复用原查询，不根据中途结果换样本或调参数。

## 问题与总体

问题：在一致的因果历史、原始目标和修复执行器下，GEM 对不同冻结 controller 的
Revisit utility 和 Novel interference 分别如何？不是比较完整官方导航系统的 published SR。

HM3D：原 Table I 28 histories / 21 scenes，56 queries。
MP3D：原 Table I 42 histories / 25 scenes，84 queries。
NavDP、ViNT、NoMaD 各运行 native/GEM 两臂：210 controller-history cells，840 rollouts。
原 manifest 路径/哈希见 `TABLE1_REPAIRED_QUERY_PREPARATION_20260910.md`，并进入 plan。
本次复用旧 actual-online A；不能称新执行链的 A+query 全流程或新 held-out 场景。

## 方法与输入

所有权重冻结、无新训练、无角色输入。GEM 保持 strict certificate、geometry-first、
top8、原 inlier/coverage/RMSE 条件、canonical LingBot reference depth、固定2.5m方向残差。
正在进行的去 coverage 四臂消融不改变本次默认 GEM。

| Controller | native | GEM accept | GEM reject |
|---|---|---|---|
| NavDP | 原始 ImageGoal + 共享单目深度 | 同一原始图 + 认证bearing的PointGoal + 单目深度 | 原始NavDP请求 |
| ViNT | 原始 ImageGoal，RGB-only | 认证历史anchor的RGB作为ImageGoal | 原始ViNT请求 |
| NoMaD | 原始 ImageGoal，RGB-only、goal mask=0 | 认证历史anchor的RGB作为ImageGoal，mask=0 | 原始NoMaD请求 |

ViNT/NoMaD 不接收深度或PointGoal。认证bearing只供公共物理朝向适配使用，
不是伪装为它们具备PointGoal接口。成功始终按原始目标计分，不换成anchor位置。
GEM geometry stream故障为异常/停止，不能悄悄换controller完成任务。

## Controller 输入修复，不按导航结果调参

- ViNT/NoMaD 的 JPEG 均按 RGB 解码，取消旧 server 的 BGR 交换。
- NoMaD 使用官方导航的mask=0；旧本地step_imagegoal中mask=1实际屏蔽目标。
- 每次NoMaD采样使用真实paired diffusion seed，8个候选、10个denoising步，固定选sample0；
  不调用导航GT挑样本。不再把它错误标作deterministic controller。
- 两个图像controller均保留原始action输出，不采用NavDP-baseline wrapper附加的
  `predicted_distance > 7`清零规则；distance head只记录。不是寻找更有利的替代阈值。
- ViNT确定性输出不假称消费diffusion seed；NoMaD相同seed的所有候选必须逐项一致。
- 仍沿用各发布模型的context、normalization和共享baseline轨迹插值；不宣称复现官方topomap系统。

官方依据（2026-09-10读取）：
[导航部署源码](https://github.com/robodhruv/visualnav-transformer/blob/main/deployment/src/navigate.py)
的NoMaD导航分支使用zero mask，按sample0取动作；预测距离用于选择topomap节点，
不是对输出轨迹做大于7的二值清零。
[NoMaD模型](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/nomad/nomad_vint.py)
定义0为不mask、1为mask。这里没有加入官方完整topomap搜索、离线地图或新探索器。

因此新旧Table I差异包括controller输入合约修复，不单独归因于pursuit或GEM。

## 执行与评分

600实际ticks、每8步重新规划；bounded pursuit，零参考不平移，短参考不超程。
环境采用一次标准Habitat try_step碰撞响应，不做自定义snap/30%重试控制。
后向认证方向由公共适配器执行每步最多4.5°物理旋转，新RGB持续写LingBot；
转动期间不重新采样policy，完成后新观察立即规划。所有动作计入同一预算。
仍是运动学仿真、理想低层状态和GT平面距离1m计分，不是实机碰撞或自主STOP验证。
SPL按逐动作实际位移和最后落点精确重算；原地转身不增加SPL路径距离。

每cell四个rollouts在同节点、同服务实例、同权重上运行，各臂独立reset/replay同一A。
Novel/Revisit各两臂的执行顺序按history索引交错平衡。baseline始终为选中controller本身。
不把NoMaD/ViNT的GEM失败回退给NavDP。未经GEM接管的query须与该controller native逐动作一致。

## 本机与HPC放行

本机先用已消费的MP3D首条完整history，三controller各Novel/Revisit×native/GEM，
共12次、600ticks上限。该history是旧metric-A，仅用于接口和闭环验证，不计入正式总体。
放行看reset、RGB、seed、history、接管、朝向、实际位移和独立验证，不以GEM赢作为条件。
真实权重CPU检查另验证mask作用、HTTP输入、重复seed和多次reset。

HPC exact container/解释器/源码闭包/输入检查通过后，首history×三controller×两dataset
六个完整cells作为正式集成检查，其余204个cells依赖它们全部运行成功；六项计入210，不重复采样。
优先A100、gpu48，逐cell1小时上限（本机实测与HPC gate再核对）、最多4并发，不干预其他现有任务。
逐帧证据写node-local，整cell压缩归档后持久化；只重跑基础设施失败的整cell，不混partial arms。

## 报告

六组各自报告Novel/Revisit及等比例合计SR、SPL、GEM对native增/损、exact McNemar，
场景聚类bootstrap95%CI（20000次，seed20260910）。六个Revisit对比组成Holm比较族。
Novel单独报告接管、损失、exact-native比例，不把未显著受损称作总体安全保证。
保留全部预定查询、失败、零增益与负结果，不按成功率追加样本。
该协议不改论文数字、现有默认方法、真机服务或其他冻结实验。
