# 仿真执行链审计与最小修复

## 本轮范围

2026-09-08，用户设定今晚 goal：最小修复当前仿真缺陷，并审计相邻接口，避免只修一个现象。
保留 Habitat/NavDP/CEC；不迁移 X/Isaac，不动真机和 HPC，不改论文旧数字，不 commit/push。
保护已有 dirty worktree。本轮生产 evaluator、NavDP/CEC 默认行为均未修改，候选修复在私有 runner 中启用。

以下不是“保证没有 bug”的声明。单元通过、源代码审查、真实闭环分别列出，不互相冒充。

## 当前收口状态

- 执行/颜色桥接：24/24 完成，独立核验与 24 视频完成；见独立结果文档。
- 深度栅格：7 项真实 resolver/HTTP 测试通过，4 臂真实模型短测完成并核验；
  两历史八臂 600-tick 对照完成，291 份深度转换独立复算，8 视频完成。
  native 0/2→1/2，CEC 1/2→1/2；见 `HABITAT_DEPTH_RASTER_BRIDGE_RESULT_20260908.md`。
- 历史追溯：已只读审查旧 HM3D full-mono 的 168 个 query；CEC-Revisit 17/28 出现
  非低分零候选终点后的运动，不能把这一暴露比例解释为修复后 SR 损失。
- 后方 PointGoal 的域适配仍未并入主方法；仅修零参考会暴露停滞，不代表完整导航问题解决。
  21 项单元/接入检查通过，四臂 64-tick 本机短测已核验并导出视频；
  四历史完整 off/on 配对 16/16 完成并核验；raw 与 CEC 各自 2/4→4/4（+2/−0，p=0.5）。
  见 `HABITAT_FRONT_GOAL_BRIDGE_RESULT_20260908.md`，不把机制样本升格为论文确认。
  最后两历史四臂的真实 native/非接管短测已完成；两条 Novel 查询均未接管，
  64 动作前缀的全部规划输入/输出及执行逐值一致。16 对深度收据也完全匹配，
  32 个深度数组独立复算通过，4 个视频全部导出。该项不是完整 Novel SR。
- 生产默认、论文数字、HPC 和真机均未改；当前修复仅在独立测试配置中执行。
- 本轮三组完整机制对照与一组前缀检查均收尾，私有模型服务已正常退出。
  最终入口见 `STATUS_20260908_EXECUTION_REPAIR.md`。不同阶段复用了历史，不能相加为独立样本。

## 一、审计发现

### 1. 原 PP 零参考仍推进：已确认，候选修复已测试

`eval_2leg_habitat.py:pursuit_step` 的速度不随参考长度归零。
新 `bounded_pursuit.py` 分离命令与碰撞；维持原转向/速度上限，仅使零参考保持、短参考限制步长。
非约束参考的 200 个随机测试与原始命令逐值一致；另测零参考任意朝向、短线段、世界坐标旋转平移、
碰撞后实际路径计量及坏输入。不把 hold 解释为成功，不加搜索动作。
这仍是 forward pursuit，不承诺跟踪任意闭环或后向轨迹，也不声称实现了 MPC。

### 2. 自定义落点吸附/缩步：实现及逐动作影响已确认

旧函数直接 `snap_point`，超容限则 30% 重试；新接口仅调用一次标准 `try_step`。
同时读取本机 Habitat-Sim 0.3.3 的真实 `Simulator.step_filter` 源码确认：NavMesh 已加载且
`allow_sliding=True` 时调用的正是 `pathfinder.try_step(start_pos,end_pos)`；非滑移模式另有
`try_step_no_sliding`。这是与标准环境步进的对应关系，不是把 NavMesh 改名成无 GT 碰撞。
标准环境仍使用 NavMesh，允许其标准滑移；没有改相机、包络或路径规划，也不称无 GT 物理仿真。
策略默认 server 选择支路已用禁止访问对象测试：选择轨迹时不会读 GT 目标、当前位置或 Pathfinder。
环境的碰撞响应与模型获得 GT 路线是不同权限；但它仍不能验证 Go2 全身接触安全。

### 3. base NavDP 颜色集成错误：本轮新确认

不是只有 X。当前 base 的链路是：Habitat PIL 标准 RGB JPEG → server RGB2BGR →
模型 process_image 保持通道，最终模型收到 BGR；本地官方 NavDP 原客户端将 raw RGB 直接
交给 OpenCV JPEG，再经过同一服务端转换，两个交换相抵，模型实际收到 RGB。

红色测试 [220,60,20]：官方入口 [220,60,20]；现集成 [20,60,219]；修复 [219,60,20]。
当前帧、goal、history 均测试。JPEG 引入的约 1 像素值差不是通道差异。
私有修复只在模型图像预处理前恢复 RGB，原 JPEG/SHA、LingBot、depth transaction 均保持。
合同在 reset 固定，不能从 SR 或 critic 结果临时选择。生产默认未变。

证据：`.diagnostics/habitat_minimal_repair_20260908/base_rgb_probe_v1/summary.json`。
这只确认当前源码。各历史表格是否用了完全相同解码，仍需核对其冻结版本，不能自动全项目外推。

扩大接口审计后，**当前 ViNT transport 也确认相同通道问题**：实际 `vint_server.py`
当前帧/goal/context 解码后仍是 BGR，`base_agent.process_image` 和 `transform_images`
不再交换。通过实际 RGB normalization 反算，测试红色 `[220,60,20]` 变成约
`[20,60,219]`。没有加载 ViNT 权重或跑新 ViNT SR，也未修改其生产服务。
结果见 `base_vint_rgb_probe_v2/summary.json`。这不是仅因“NavDP 错了”而推断 ViNT 也错。

Git 主线 `932506e`、`00643bb`、`87006fb` 中 JPEG encoder 与 base image preprocessor
的函数 AST 均与当前一致，server 都含 RGB2BGR 转换；说明这不只是今天新写出来的行为。
但主线 revision 不是实际 HPC bundle 收据，正式表格覆盖仍须逐项绑定。
前两版 pursuit 的位置/转向公式未改；当前差异是此前把返回路程从命令长度改为实际位移。

### 4. 单目深度 padding：合成、真实 payload 与独立闭环均完成

实际 LingBot loader 的 270×480 → 518×518 变换将有效内容放在行 `[112,406)`；
NavDP RGB 224×224 的有效行为 `[49,175)`。直接将整幅 518 深度送入 NavDP，
既没有去掉 LingBot 白色补边上的预测，也有约一行的有效区边界差异。

只读 `/monocular_depth_query` 的已物化 token 分支，不调用 reset/add_frame 或无 token 的预测：
180 秒内取得 1 个有效样本，19 次 token 已过期按 HTTP 409 拒绝，未改为读取不匹配帧。
`gxdoqLR6rwA / CEC / bounded / RGB / frame 328` 的真实收据经现有 NavDP 预处理后，
21,952 个 RGB padding 像素中仍有 15,179 个非零深度（69.15%）。这不是仅凭形状猜测。

新增独立 `lingbot_depth_raster.py`：按实际 loader 的 resize/pad 坐标先裁回有效内容，
恢复到该帧原 RGB 栅格，再交给未改动的 NavDP depth encoder。它不改变深度单位、
尺度估计、证书或历史 PnP 栅格。15 项测试通过，包括直接比对真实 loader、横竖画幅、
常量米值、轴向、零深度和方形恒等。
上述真实 payload 离线回放后，RGB padding 非零深度从 15,179 变成 0。

证据目录：`.diagnostics/habitat_minimal_repair_20260908/actual_padding_probe_v1`；
原始 payload SHA：`f9ad9eee2934f2a006dfd75255920df21481f3388f59254b6c6bf6fb3baea547`。
这是一个帧的输入契约修复证据，**不是准确深度、更高 SR 或总体影响大小的证明**。
本轮已完成的六臂未启用此修复，不能把其结果称为所有输入修复后的版本。

新增私有 `navdp_depth_raster_audit_server.py`，将逆变换接在原 HTTP 深度交易验证后、
未改动的 NavDP encoder 之前。真实 resolver + Flask 测试验证 reset 模式、原缓存不变、
bootstrap/无效尺度仍为零、metric-depth 对照恒等，以及原始/转换后深度归档。
这是请求链测试，模型闭环另按 `HABITAT_DEPTH_RASTER_BRIDGE_PROTOCOL_20260908.md` 执行。

### 5. 旧正式 HM3D 日志也出现零终点后的运动：新增只读追溯

读取已落本地的 full-mono natural-direction 原始日志，28 histories × 2 roles × 3 arms，
共 168 个 query 文件；没有跑策略、修改数字或重新挑选样本。共享 A 在每历史六份日志中逐值一致。
以日志原值 `candidate_endpoint_length_mean = std = 0` 为描述条件；再单独列出 critic ≥ −0.5
的记录，避免把明确低分搜索混入同一描述。下一动作位移直接来自相邻保存坐标。

| 查询分组 | 至少出现一次上述零终点统计 | critic ≥ −0.5 且下一动作移动 >1 μm 的查询 |
|---|---:|---:|
| CEC / Revisit | 17/28 | 17/28 |
| raw fixed / Revisit | 18/28 | 18/28 |
| native / Revisit | 0/28 | 0/28 |
| CEC / Novel | 2/28 | 2/28 |
| raw fixed / Novel | 20/28 | 18/28 |
| native / Novel | 2/28 | 2/28 |

CEC-Revisit 的 29 次“非低分、零候选终点后仍移动”记录全部具有后向 memory pointgoal。
这不能被描述为“17 个成功都是假的”：**旧日志只有端点统计和 selected-trajectory SHA，
没有完整候选坐标；也没有修复后的反事实结局。** 它证明该接口问题在旧主要结果中确实有
暴露线索，而不是只在本轮四个消耗型病例里出现。所有旧 SR 标签保持原样。

结果：`.diagnostics/habitat_minimal_repair_20260908/archived_fullmono_zero_endpoint_motion_v1.json`。
脚本 `audit_archived_zero_reference.py` 保存逐文件 SHA、逐计划统计和真实下一步位移；
3 项测试明确不把缺失终点当作零运动、不生成新的成功标签。
该追溯尚未绑定完整 HPC 源码 bundle，不能替代冻结版本审计或配对重跑。

## 二、其余边界审查

| 边界 | 当前证据与状态 |
|---|---|
| 坐标、单位 | Habitat xz world 与 local forward/left 变换已读源码；新命令刚体变换测试通过。v_max=0.0376 是每 tick 位移，不是 0.0376 m/s。未更改步长/角度。 |
| 后向 PointGoal | frozen base 将负 forward 裁为 0；模型另有短轨迹 XY mask。私有四历史 off/on 对照已完成，raw 与 CEC 各 2/4→4/4；不改模型，也未默认启用转身。 |
| 低 critic | 上游可替换为侧向搜索参考；不是 STOP 或碰撞安全证书。新 tracker 没有引入新 fallback，但不能宣称上游不存在该行为。 |
| 重规划/FIFO | 当前 8 tick 重规划；NavDP 记录 decision frames，LingBot 接收各动作观测。共享 A 只 replay 原 decision frames 进 NavDP。相关 replay/goal-switch/native-first 测试通过。 |
| causal history | 当前诊断 A 是已执行 metric-NavDP-A，不是 expert；新旧查询共享同一历史。没有重新证明 full-mono-A。runtime query 去除 role 字段的已有测试通过。 |
| GT 授权 | server 选轨迹不读 GT，oracle selector/forced anchor/graph rescue/terminal U-turn 全关闭。旧通用执行收据可携带位姿差，endpoint bearing 不消费这些 route 专用数值；不宣称整个旧 HTTP 对象从未携带理想里程计。 |
| 单目深度 | 首 40 帧/尺度冻结/相同 JPEG 不同 frame token 的缓存测试通过；真实短测确认 sidecar 且 sensor-depth-consumed=false。尺度可能 clamp，不能将高度先验说成精确尺度保证。 |
| 深度栅格 | 实际 loader、合成与真实缓存确认不一致；逆变换/HTTP 测试、四臂模型短测和独立八臂闭环完成。结果见专门文档，不将 N=2 外推为总体增益。 |
| 到达 | 当前是 evaluator 平面 GT <1m，不是视觉自主 STOP，也不要求到目标朝向。其他协议的 3D/终端阈值不能混算。 |
| 停滞 | evaluator 按实际位移窗口提前结束。这是评测退出条件，不是 NavDP 自动停止；本轮固定不改。 |
| SPL | 当前 measurement 保存末动作后坐标，逐动作实际位移求和，失败 SPL=0；不再用命令长度替代真实路径。缺终点时返回缺失而不是补零。 |
| 延迟 | 每 8 tick 是仿真控制时间，不包含 GPU/HTTP 等待；离线视频不是端到端实时时间。 |
| 数据/环境 | 四场景资产、起点、初始 geodesic 与渲染预检通过。私有端口 21690/21691；18888/8888 驻留服务未操作。 |

### 相机元数据的附加发现（未混入本轮修复）

独立构造同一 Habitat camera 并读取实际 projection matrix，480×270 的实际焦距为
`fx=355.8146095, fy=355.8146060`。生成器元数据写的是 `fx=355.81464, fy=351.687`，
fy 有约 1.16% 偏差。此处是实际渲染矩阵对账，不是只按配置名推算。

- 当前 base NavDP 只在轨迹可视化中使用这份 `image_intrinsic`；其 RGB/depth encoder 不消费它。
- CEC PnP 使用 LingBot 预测的内参，不使用生成器这个常量。
- 生成器的共视投影和相机元数据确实使用该 FY。四历史的 8 个原查询已只读重算：
  实际目标重渲染 JPEG 均与保存图 SHA 相同，旧参数的完整共视曲线均逐值复现。
  改成真实 fy 后，4 个 Revisit 的最大共视和最佳帧均不变；其中 gxdoq 的部分曲线点发生小变化。
  Novel 中仅 gxdoq 最大共视从 0.0518084 变为 0.0531118，最佳帧 12→13。
  全部曲线点的最大绝对变化为 0.0029326。四个 Novel 仍 <0.08，四个 Revisit 仍 >0.80；
  不改变这 8 个查询的构造角色结论，不能把 N=8 外推成所有数据均无影响。
- 本轮使用封存的既有查询，不改变相机、数据或标签。后续重新生成新版本 A/query 时，应从
  实际渲染相机矩阵导出 K，并单列构造敏感性，不再另手写一个 fy。

本轮没有把该元数据问题解释为后向零参考的根因；后者已有同状态同参考的直接执行证据。
脚本：`audit_habitat_focal_metadata.py`；结果：
`focal_metadata_sensitivity_v2_raw_goal.json`（同一诊断父目录）。

一次重要的复算纠正：最初 v1 直接使用保存的目标 depth PNG，无法重现部分旧曲线。
`uint16 × 10000/m` 只能表示至 6.5535 m，而原构造将重渲染的 float goal depth 用于 3D 点采样。
改为从原 pose 重渲染目标 float depth 后，8/8 旧曲线完全复现。因此不能把 v1 的复算差异
指控为原始标注错误；v1 保留为输入表示不一致的诊断，不用于新标签或论文数字。

## 三、测试与运行进度

- 87 项主要链路测试通过，12 条依赖 Pyparsing 弃用 warning 保留在日志中，未隐藏。
- 另 3 项 GT/测量边界测试通过，合计 90 项。
- 随后深度 raster 新增 15 项通过；合计 105 项（不是 105 次独立导航实验）。
- ViNT 实际通道链新增 1 项；统一入口实跑为 **106 passed / 12 warnings**：
  `bash MemNavData/run_habitat_contract_preflight.sh`。
  测试包括显式复现旧缺陷以及候选修复；通过不代表生产默认或全部旧实验已修好。
- HTTP 深度 resolver 与保存收据测试新增 7 项，统一入口最新 **113 passed / 12 warnings**；
  JUnit：`preflight_tests_v4.xml`（同一诊断父目录）。另有上述旧日志追溯 3 项独立测试通过。
- 原有 `test_lingbot_pnp_localization.py` 9 项通过；稀疏匹配先转入 LingBot pad 像素坐标，
  不受此次仅改变 dense readout 的裁剪影响。没有把 depth adapter 用到历史 PnP 上。
- 汇总上述测试、旧日志追溯和后向域适配后，统一入口最新 **146 passed / 12 warnings**；
  JUnit：`preflight_tests_v7.xml`。警告均为依赖的 Pyparsing 弃用警告，未屏蔽。
- 首历史六臂真实模型 8-tick smoke 完成，独立 verifier=true；native 与 CEC 各自同色首次计划逐值相同。
  同色新旧实际短路径分别为 native 0.2983260399 m、CEC 0.2896977399 m；零参考未触发。
  各臂预算只有 8 tick，未到达不代表正式导航失败；短测不加入完整 600-tick 结果。

协议：`HABITAT_MINIMAL_EXECUTION_REPAIR_PROTOCOL_20260908.md`。
正式机制配对为四历史 × 六臂、每臂 600 tick，结果写入新的 bridge 目录。
源码快照/共享历史/原始完整预测/动作前后坐标/图像通道/深度交易均保存。
独立复算与全部第一视角视频由同一任务收尾，不依赖另一个未启动的 watcher。

完整任务已在 `.diagnostics/habitat_minimal_repair_20260908/bridge_v1` 完成 24/24；
独立核验与 24 个第一视角视频完成，原源码/权重没有运行中变更。
全部结果见 `HABITAT_MINIMAL_EXECUTION_BRIDGE_RESULT_20260908.md`：旧 native 1/4、CEC 3/4；
新执行 native 0/4、CEC 2/4，正确 RGB 未再改变四例成功标签。
随后才扩展私有 runner 以支持独立 depth-raster study；旧归档保持不变。

### 逐例机制观察（全量结果见上方独立报告）

第一条 `gxdoqLR6rwA` 六臂完成：三种 native 均未到达，三种 CEC 均到达。
旧 BGR 下新旧执行器分别完全复现 native 600 tick/22.2249 m、CEC 117 tick/3.9950 m；
正确 RGB 的 CEC 为 115 tick/3.9053 m。该条没有触发零参考或实质碰撞修正。

第二条 `pLe4wQe7qrG` 六臂完成，三个 CEC 臂均未到达；三个 native 臂也均未到达：

- 首次请求 pointgoal 为 `[-2.417701,-0.636177]`，处于后方；
- 离线 evaluator 注释的目标直线 bearing 误差约 `0.2303°`，不能在这条上先归咎于定位漂移；
- base encoder 将负 forward 裁到 0，首次返回的 16 条候选全部零 XY；
- 旧 PP 仍从这个零参考生成了第一次运动，后来走 3.2330 m 后停滞，未到达；
- bounded BGR 与 bounded RGB 都在 151 tick 停滞退出，实际位移为 0；正确颜色没有解决后向输入。

这是“输出语义问题被 tracker 自行推进掩盖”的实际案例，不是仅有合成测试。
它也说明本轮零参考修复不承诺 SR 提升。下一步若引入真实朝向调整，应作为独立的
PointGoal 输入域适配对照：旋转期间保持因果新 RGB，完成后重新定位/规划；
不能复用转身前轨迹、用虚拟旋转替代真实视图，或假定 LingBot 纯转身一定稳定。
本轮六臂没有偷偷加入该调整，也没有将此一例推广成全项目失败原因。

## 四、论文对应边界（本轮只报告，不改稿）

当前 `/home/asus/Research/Memnav_Paper/sec/5_experiments.tex` 仍写：
“Simulator geometry is used only for query construction and arrival scoring.”
这句话与现有环境碰撞及理想位姿跟踪不完整对应，不能继续原样保留。
投稿前应明确：模型不接收 GT 路线/目标位姿；执行评测使用运动学 agent、
理想状态反馈和 Habitat NavMesh 碰撞近似；它不是原 NavDP 的 IsaacSim+MPC 物理复现。

官方核对（2026-09-08）：

- [Habitat PathFinder](https://aihabitat.org/docs/habitat-sim/habitat_sim.nav.PathFinder.html)：
  `snap_point` 找邻近可导航位置，`try_step` 约束从起点可行地到达终点；二者都属于 NavMesh API。
- [NavDP 官方仓库](https://github.com/InternRobotics/NavDP)：官方 benchmark 使用 IsaacSim/IsaacLab、
  异步规划与 MPC 跟踪。当前 Habitat 迁移不等同于该官方物理执行。

本地官方 `eval_imagegoal_wheeled.py:89` 同样只将预测的前两个平移坐标传入跟踪器；
`tracking_utils.py` 给参考 yaw 填零且 yaw 误差权重为零。
因此不能把第三输出维未经核实就解释成应当执行的原地转身指令。

## 五、如何减少再漏问题的概率

此前封存、配对和哈希主要回答“是否运行了同一版本、是否用了同一数据”，并不自动回答
“这个版本的接口语义是否正确”。RGB/BGR 形状和数值范围都合法，视频也可能一直是正确 RGB；
零候选经旧 PP 仍然移动，甚至有成功结局；深度 padding 也不会必然触发异常。
因此只看 HTTP 成功、代码可运行或最终 SR，能让这些相邻接口问题长期隐藏。
同一个缺陷又可能对 ImageGoal 和后向 PointGoal 触发得不同，配对本身不能消除这种交互。

保留一套短小的必跑边界测试，不靠堆运行时 fallback：

1. 彩色而非灰色输入，从真正客户端到真正图像预处理。
2. 零轨迹、短轨迹和受阻运动，核对命令与实际位移，不只查最终 SR。
3. 用禁止访问对象验证默认选轨迹不读取地图和 GT 目标。
4. 每次颜色/坐标/深度改动，都检查当前帧、目标、历史三条入口，不能只修一个。
5. 对真实短测保留第一视角、完整规划、最后动作后坐标；源码检查不能替代真实调用。
6. 对原地转身检查真实新图、FIFO 时序、转后重规划、位姿漂移和原停滞预算；不只检查 yaw 到位。
7. 离线重算先复现旧量，再改变一个定义；本轮 float goal depth 与存盘 PNG 的区别就是实例。

不以 N=4 无显著差异证明等价；不把新版实测结果回填旧表格；不只录成功视频。
小规模验证已经完成。修复版主表以及受影响闭环消融的重跑范围，见
`HABITAT_REPAIR_RERUN_SCOPE_20260908.md`；本轮不自行提交正式任务或替换论文数值。
