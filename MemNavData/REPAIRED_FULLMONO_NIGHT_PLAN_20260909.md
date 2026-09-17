# 今晚目标：修复版GEM的核心证据与论文版本闭合

## 2026-09-10 终局更新（北京时间 01:02 核验）

两批冻结数组及最终 CPU 汇总全部完成：共视 159/159、新 full-mono 90/90，
合计 747 条三臂导航，无待补失败项；核查时队列为空。
新 full-mono 的 Novel 为 native/raw/GEM=15/45、9/45、15/45，
Revisit=18/45、45/45、45/45；GEM 对 native +27/−0，p=1.49e-8，
对 raw +12/−6，p=0.2379。共视支持查询为38/131、127/131、105/131；
24条 raw 成功/GEM失败中23条未接管且 exact native。
完整终局 SR、SPL、多重比较、分母和来源见
[最终结果](REPAIRED_HM3D_FINAL_RESULTS_20260910.md)。

两批结果不混成同一总体。当前优先事项转为离线分解授权损失和整理修复版本的论文证据；
本轮没有调阈值、增加新导航或改 LaTeX。下文所有“在跑”“尚无终局”均为保留的旧时间线。

## 2026-09-09 工作记录（保留）

2026-09-09，用户已授权设为持续goal。本文件记录实际进展，不表示全部目标已完成。
主线仍是因果单目历史的双时间尺度记忆；不启动新训练、GOAT或真机分支。

**20:15 最新概览**：新版mono-A、全部构造与CPU封存均已完成。
112条A/28个scene中88条到达；最终45对Novel/Revisit查询，覆盖23个scene，
合计90个查询、270次三臂导航。保留首前缀全部合法history，不再向后扩源。
源A和构造完整损耗均已独立复算，不能将88/112写成query SR。
90个实际查询输入的原容器预检全部通过；固定index 0三臂`17262832_0`已完整通过，
6分39秒，独立动作/深度/SPL与归档验证均通过。保留首项，剩余1–89已提交`17263230`，
CPU终局汇总`17263231`已绑定依赖；导航成败没有作为启动门。
共视`17240715`已有147/159项完整归档并通过验证；新full-mono共50/90项
完整归档并通过验证（含保留首项）。没有失败/不完整归档，两份最终汇总尚未生成。
两批后续项会因QOSGrpGRES等待；新full-mono曾短暂没有运行单元，随后已恢复。
四路并发上限、1小时时限及冻结总体均不变。
论文TeX和全部保护数字不变；下文为完整时间线与实现依据。

用户已明确要求查看当前SR，故在20:15另存只读中途快照：
新A查询中，Revisit native11/30、raw/GEM均30/30；Novel native/GEM9/20、raw6/20。
共视支持查询描述性汇总为native34/124、raw121/124、GEM101/124；
其中21条raw成功/GEM失败有20条完全未接管并exact native。
这表明低共视授权覆盖是实际代价；不是完整总体结论，不由此调阈值、删样本或停止数组。
见 [完整中途SR及损失核查](REPAIRED_HM3D_INTERIM_SR_20260909_2015.md)。

17:27：首项`17262832_0`已在A100/ga029启动。为给它调度机会，曾短暂将共视后续
并发上限设为2；首项启动后已恢复4并回读确认，没有取消或重启正在运行的共视任务。

当前下一步：完成两项冻结数组 → 独立复算各自完整SR/SPL → 按执行版本更新论文证据，
缺失任务明确列出，不混旧执行臂、不由部分SR改变总体。
本轮提交阶段已完成，**不表示90项或159项终局评测已经结束**。
新总体与损耗详见 [完整源前缀与封存结果](REPAIRED_FULLMONO_PREFIX30_POPULATION_20260909.md)。
首项结果和后续作业见 [三臂首项核验](REPAIRED_FULLMONO_QUERY_GATE_RESULT_20260909.md)。
终局统计已另做交叉检查：12,880个McNemar组合、三组Holm手算例，以及两个baseline
各20,000次按scene展开的bootstrap均一致；未改变冻结分析，也未读取部分SR作选择。
见 [统计入口检查](REPAIRED_QUERY_STATISTICS_AUDIT_20260909.md)。
补充的真实归档reader检查与共视CPU汇总启动检查均通过，未安装依赖。
另外直接读取冻结policy确认：新RGB持续更新几何状态，但当前目标的检索
候选只来自查询开始前历史；本轮未改变该边界，见 [状态与历史审计](REPAIRED_FULLMONO_SOURCE_RESET_AUDIT_20260909.md)。
另已检查首个完成的Novel归档：native/GEM都未到达、没有接管、逐动作一致；
267张native-B观测及其完整终点均可恢复。未来连续导航可先审计这些实际B，
减少重复采集，但本项失败不能作为成功前缀，本轮未新增C评测。
见 [native-B归档检查](REPAIRED_FULLMONO_NATIVE_B_ARCHIVE_AUDIT_20260909.md)。
下文带旧时间的“尚未提交/未封存”均为当时状态，不能替代本节最新概览。

源A完整逐scene结果见 [首前缀采集结果](REPAIRED_FULLMONO_PREFIX30_A_RESULT_20260909.md)。

16:11构造已核验rank0/2/3/4：合法role pair分别0/1/1/0，共2对；rank1仍运行，
其余按队列推进。这是完整前缀中的部分构造结果，不能提前封存或据此更改规模目标。
全部112条A的小收据已回传本机，重读得到同样的88条到达、28份归档回读通过。

本轮也已把首8个完整scene的构造小收据回传并在本机重新汇总：32源中，
9对合法查询、11条A失败、1条历史过短、9条缺standard Revisit、2条缺unsupported Novel。
这是部分构造损耗，不是全前缀结论。Revisit支持区间指frame≥39的合规历史最大值；
全部历史的最大值可能更高，不能混用。Novel则仍对全部历史要求最大共视<0.1。

## 已完成

1. 按共享SSH手册核验 `alantorch` → `yz11502`，默认master正常。
   11:27共视为19完成/1运行/139等待、0失败；11:45已有20完成、2运行。
   截至12:10的最近一次查询已22完成，剩余137个配对任务和构造探针等待QOSGrpGRES配额；没有新失败。
   scratch `myquota` 为1.32TB/5TB、3,494,312/5,000,000文件（69%）。
   不因为排队或部分SR而修改159个冻结目标。
2. 本机以原始任务parent SHA清点全部196源/49非空scene；未读取query outcomes，未筛选。
   A任务测地距离[3,4)/[4,5)/[5,6)/[6,9)m分别52/46/30/68条。
   全部源中位数4.999m；只取episode0000会得到49条、中位数4.332m。
   这些不是实际A路径长度，也不是query距离。
3. 对两条实际新版A的NavMesh，CPU独立扫描前/侧/后三方向。

   | 历史 | front空间候选 | side空间候选 | rear空间候选 |
   |---|---:|---:|---:|
   | rJhMRvNn4DS | 104 | 0 | 49 |
   | jgPBycuV1Jq | 3 | 0 | 494 |

   这是新seed的几何诊断；不测视觉共视，不等于有这么多合法Novel，不能当SR。
   原5000提议的精确重放仍在旧审计文档；本次没有改写原计数。
4. 独立构造程序、本机和独立staging各23项CPU测试通过；旧构造函数没有编辑。
   新入口只在自己的进程内使用实测K与浮点深度，离开上下文恢复原函数。
5. 论文工作区新增 `EXECUTION_VERSION_MATRIX_20260909.md`，并给README、会议矩阵、
   证据总账添加新版本说明。四张表旧版实验都已完成，Table III精确SPL已补齐；
   新问题是修复版闭环对应。全部LaTeX、标题、Abstract/Introduction、四表逐字节不变。

## 已提交

**17251134：两历史构造可行性验证**，A100/1GPU/4CPU/16GB/1h。
不加载NavDP/LingBot、不采集A、不跑query，只验证新构造规则的实际图像支持。
先在每个方向层各找合法Novel，再按固定hash选择；支持/距离阈值不变。
同时明确变更了离线K与深度标注，不能当原固定方向协议的exact retry。

- [冻结构造设计](REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_PROTOCOL_20260909.md)
- [提交收据](REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_SUBMISSION_20260909.json)
- 程序：`probe_repaired_fullmono_construction.py`
- 远端：`/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_fullmono_design_20260909/construct_b9297569ff0805d7`

登录节点预检先触发手册已记录的满`/tmp`/llvmlite MemoryError；用任务专属scratch临时目录后，
同一封存代码23项测试、源文件检查全部通过，才提交GPU任务。未重装环境。

## 下一步（依赖顺序，不按结果择优）

1. 查看17251134的实际构造终局；成功和失败都保留。不能把合法新目标叫导航成功。
2. 新正式源方案保留原196个任务及其scene/episode顺序，不只挑长任务、旧成功A或旧28历史。
   原scene前缀[30,36,42,48,54]对应112/132/152/172/196个实际源；
   使用最小满足24 histories/15 scenes的预定义前缀，或到全量明确报告不足。
   只根据A与构造可行性决策，query arms全部在population封存之后启动。
3. 为新完整mono-A批次准备collection→construction→native/raw/GEM配对，
   保留所有A损耗与exact terminal SPL；每个配对单元同进程，不混旧native。
   新方向构造协议只能用于新命名总体，不覆盖旧gate/共视/主表结果。
4. 核心新版结果后更新会议Table I的query portability，再依保留的claim处理Table II/III/IV。
   固定旧历史的query消融可继续使用旧因果历史；端到端mono主张必须用新实际A。
5. 并行准备短视频与最终稿件检查，但不擅自上传、投稿或推送Git。

当前完整mono-A正式批次**尚未提交**；等待新构造的真实图像验证及正式入口预检。
正在跑的共视评测不是这项主实验的替代。

正式入口的当前准备状态：

- `plan_repaired_fullmono_population.py` 已生成 `core_source_plan.json`，保存原scene/episode
  排名与seed；6项测试覆盖前缀完整性、全量保留、不读取query outcomes及不足的明确终态。
- `collect_repaired_fullmono_a.py` 已实现每scene全部4条任务的A-only采集，复用原私有模型服务、
  A调用和逐动作SR/SPL verifier；不调用任何query构造/评测。4项CPU测试和CLI帮助检查通过。
- `run_repaired_fullmono_a_collection.sh` 与对应SBATCH已准备，逐帧证据在node-local，
  复用已有逐文件回读归档工具。合集33项CPU测试通过；独立封存collector包及远端原容器
  各14项测试通过，rJhMRvNn4DS四个任务的真实文件绑定与原A CLI dry-run全部通过。
  仍待单scene完整运行，**不能把这些CPU检查写成端到端验证通过**。
- collector封存为 `repaired_actual_a_3e90959723d4b66b`；源计划SHA为
  `ba1a12273bf9eeef5024166c4bfa28037ee6076c695d2543959a7e169adafba7`。
  调度器test-only接受；输出的17251620是测试编号，不是实际job。
  [完整采集协议](REPAIRED_FULLMONO_A_COLLECTION_PROTOCOL_20260909.md)、
  [远端预检收据](REPAIRED_FULLMONO_A_COLLECTION_PREFLIGHT_20260909.json)。
- 没有为了凑24条只取前24个合法history；选中前缀中的全部合法历史都保留。

为避免探针完成后等待人工轮询，已准备一个独立的CPU续接入口
`launch_repaired_fullmono_a_after_probe.py`：只读核验两条probe的目标资产和构造边界，
仅在两对都完整时提交原rank0 scene的四条A。它不修改探针或collector，不启动全量扩样，
也不启动query评测。四项本机测试覆盖不完整、构造失败、支持/距离放宽及源身份错误。
**17251808** 已于12:21提交并核查为Dependency等待，`afterany:17251134`。
原容器CPU侧四项测试与调度器test-only通过。它是条件提交器，不是已启动的A或query。
[提交收据](REPAIRED_FULLMONO_DEFERRED_A_SUBMISSION_20260909.json)。

本机也已补齐这两条历史对应的HM3D场景资产（共56,376,640字节），与原parent SHA一致。
当前4090有另一工作线的真机服务占用，因此本轮只下载资产与做CPU准备，不停止服务、
不与真机抢占GPU；HPC正式探针保持原提交。

### 后续入口准备（尚未运行正式构造）

`construct_repaired_fullmono_queries.py` 已实现按scene接收完整A归档：提取全部实际trace，
包括失败A；用既有materializer逐帧验证真实RGB，再复用已声明的实测K/浮点深度和三方向
构造规则。每条历史只把固定hash选出的一个Novel与一个standard Revisit纳入后续配对，
其余方向仅保留构造诊断。未选中历史也保留原因；不会被记成query导航失败。

该入口尚未打包到正式GPU任务，不能称已完成实际构造。独立reader覆盖任务身份、
实际A/RGB哈希、目标文件、所存共视曲线及固定选择；它**不独立重渲染共视曲线**。
这项边界已写入输出，避免把JSON检查冒充另一轮几何测量。
当前全套设计/采集/构造/续接CPU测试42项通过，包括从无损归档逐字节恢复全部四条A的测试。
截至12:38，GPU探针与剩余共视任务仍等待QOSGrpGRES；续接器等待探针依赖。
尚未形成新完整mono-A/query结果，不能把已提交的CPU续接器计作正式A或SR。

13:05，共视已有23完成、2运行、134等待，无失败；GPU配额开始释放。
构造探针仍待组配额，CPU续接器仍待依赖。没有读取这批新增任务的SR作决策。

### 查询入口与封存（13时推进）

已补 `seal_repaired_fullmono_queries.py`：只读A/构造收据，缺任何前缀来源就不生成
population；选中前缀中的全部合法配对保留，分别记录源A损耗与条件query分母。
`repaired_fullmono_query_eval.py` 复用现有三臂rollout与精确SPL verifier，只增加
新实际A的来源绑定。入口及协议见 `REPAIRED_FULLMONO_QUERY_PROTOCOL_20260909.md`。

本机项目Habitat Python的55项合集测试通过。首次误用系统默认Python，缺少quaternion，
改为原Habitat环境后通过，未安装依赖。远端第一版入口`59d517911e4ada61`的10项
测试通过，但六臂CLI预检在首项发现旧c8包装器没有`query_main`参数；无GPU/query运行。
修正为复用当前已运行共视bundle中的同SHA包装器，形成独立v2包
`cb812054dc470e53`，没有覆盖第一版源码。原容器10项CPU测试、6个角色/方法CLI
组合全部通过；两种SBATCH的lint/test-only通过。打印的17252851/17252852只是假提交
检查编号，不是新的运行job。详见 `REPAIRED_FULLMONO_QUERY_PREFLIGHT_20260909.json`。
该入口尚无完整GPU构造和三臂结果，源总体也尚未封存。

截至13:12，共视25/159完成、4运行、130等待，0失败；已恢复原4路并发。
构造探针17251134仍待QOSGrpGRES；续接器17251808待Dependency。
不得把资源预检打印的job编号当作已经提交正式查询。

### 构造终局与首scene实际采集（13:24更新）

探针17251134在A100正常完成（5m08s），最终1/2合法pair：jgP的Novel共视0.042157、
距离8.073555m，Revisit共视0.720098、距离2.532646m；rJh仅有Revisit，没有合法Novel。
原2/2自动续接条件未满足，17251808按该条件退出，**没有提交A**。没有重复它或将它改成通过。
[完整结果与后续开发检查边界](REPAIRED_FULLMONO_CONSTRUCTION_DESIGN_RESULT_20260909.md)。

随后已明确告知用户，单独开展原source rank0的完整scene开发检查：四条A全保留，
再按同样规则构造，包括合法的无pair结果，不以A成功率或pair数作为软件通过门。
源计划、模型/执行器、支持/距离、顺序与seed不变，不挑唯一有pair的jgP替换首scene。
它不是通过旧2/2门后的正式全量扩样，使用独立run root。

- **17253177_0**：四条actual mono-A已运行，ga002；私有memnav服务已ready。
- **17253200_0**：构造待`afterok:17253177`，失败依赖会取消，绝不拿不完整A构造查询。
- 新query导航和源前缀扩样尚未提交。
- [实际提交收据](REPAIRED_FULLMONO_COMPLETE_SCENE_SUBMISSION_20260909.json)。

同一时刻共视已30/159完成、4运行，无失败。论文TeX仍未修改。

Table-II来源追溯也纠正到实际正式v3，而非前一轮中间v2。两个实际候选模块的旧执行
函数逐字节一致；论文内部版本矩阵已更新。原表格、摘要、引言仍全部未改。

此外直接读取HPC Table-I/III封存源码，确认旧执行器存在于实际旧bundle；两个Table-I
总体的ViNT server/agent哈希也与本机通道审计相同。复用了已有CPU测试，3项通过，
没有另写一套重复颜色实验。论文版本矩阵已增加这些绑定，尚不据此推测SR影响大小。

### 完整场景终局与首前缀提交

17253177/17253200 均已完成：四条 A 正常终止，2 条到达；其中1条历史不足，
另1条没有合法Novel，最终0 pair。逐动作、精确末步SPL、1,060文件归档和全部来源
构造核验通过。这是软件管线完成与真实数据损耗，不是query SR。

在新查询导航前已另立
`REPAIRED_FULLMONO_PREFIX_CONTINUATION_20260909.md`，明确修订工作流启动条件：
原两历史2/2门仍未通过；以完整scene能正确保留全部损耗为依据，继续原源前缀。
source rank0的零pair完整保留，不挑成功scene、不修改阈值、不重跑这四条A。

**17253441** 已提交其余108条A；**17253442** 以同scene index的`aftercorr`续接构造。
加上已完成首scene，共覆盖原前30scene位置中的112源/28非空scenes。
全部112源的实际文件和Goal-A绑定已在原容器预检通过。
两数组均已回读确认为A100/1h，最多各4路；正式query仍未提交、population未封存。
先前两个仅PENDING的误分区提交已撤回，无导航输出；完整记录见
`REPAIRED_FULLMONO_PREFIX_SUBMISSION_20260909.json` 与HPC手册第16节。

同轮共视核查为38/159完成、2运行、119等待，全部已有终局exit 0:0。
排队原因是QOSGrpGRES，不改总体或方法参数来改变等待状态。

CPU源前缀检查 **17253701** 也已提交，等待上述两个数组完整结束。
它复用已有sealer：来源不完整则列缺失，规模不足则给下一预定前缀；
只在完整规模满足时封存全部合法查询。不会自行启动query，也不是query SR汇总。
具体依赖、脚本SHA与资源见 `REPAIRED_FULLMONO_POPULATION_SEAL_SUBMISSION_20260909.json`。

另补最终三臂统计入口 `summarize_repaired_fullmono_queries.py`，严格分开源A和条件query分母，
同时报告两个角色与等比例合计、GEM-vs-native和GEM-vs-raw。
本轮8个模块37项测试通过；独立汇总包本机/原HPC容器各10项测试通过，
CPU汇总SBATCH lint/test-only通过（17253609仅为测试编号，没有提交汇总job）。
封存为 `repaired_role_summary_0790a21e7f856d83`；未改变活跃A/构造/共视源码。
详见 `REPAIRED_FULLMONO_QUERY_ANALYSIS_20260909.md`、
`REPAIRED_FULLMONO_SUMMARY_PREFLIGHT_20260909.json`。

再次核对论文保护SHA：main、main_lg、Abstract、Introduction及四张表与本轮基线全部一致。

补充只读复查了首scene四条A在同一服务中的状态重置：每条从frame0/观测数1开始，
各自在自己的0..39帧建立不变尺度，没有继承上一条A的尺度。冻结底座的reset也清空
LingBot与camera head的KV缓存。数值采用原有1.15地面偏差修正，本轮没有调参。
见 `REPAIRED_FULLMONO_SOURCE_RESET_AUDIT_20260909.md`；这不构成两条A失败的因果归因。

### 配额调度与第二个完整A场景（14:33）

14:22暂将共视数组17240715的新启动并发上限从4降至2，让实际mono-A及构造有机会获得
同一QOS的GPU配额。已运行任务不取消、不重启，159个目标、方法及配对顺序均不变。
`scontrol`虽然对已结束/已展开元素打印错误，父数组回读明确为`ArrayTaskThrottle=2`；
以后判断这项调度操作应以实际父数组和待启动元素为准，不盲目重复提交。
这是资源安排，不是实验协议修改；首轮完整A/构造结束后恢复原上限4，当前仍待恢复。

17253441_1已在A100 ga039正常完成（6m13s）。6D36GQHuP8H四条A结果为0/1/1/1，
独立验证通过，归档已保存；对应17253442_1依赖已经解除，等待GPU配额。
它与rank0合计8条A、5条到达。全前缀计划仍为112源/28非空scenes，不根据这8条结果
修改源顺序或构造规则，也不把还没有生成的query记作成功或失败。

14:44完成后续预定义前缀的资产预检：原scene rank≥30的84源/21scenes全部通过
冻结collector的`bind_source`文件哈希与Goal-A图绑定。连同此前112源，原196源均已预检。
未提交新增前缀、未运行查询，是否扩源仍由当前完整前缀的构造结果决定。
原容器日志：`/scratch/yz11502/Research/Nav-axis-uturn-results/fullmono_future_source_preflight_20260909_NYXUBv/preflight.log`。
日志SHA：`c24b463bf23f291fd98f5ec82c5962837c0e0bf6cbb1febcae7cbb37050d2aba`。
14:48新A和构造仍为有效PENDING，原因QOSGrpGRES，调度器尚无预计开始时间；
共视task56运行中。未重启、未重复提交，也没有将排队当成任务失败。

14:54新A/构造仍未获得后续调度，两路共视在运行，因此临时把共视新启动上限从2降至1。
父数组回读为`ArrayTaskId=59-158%1 ArrayTaskThrottle=1`；task57/58继续运行，没有取消或重启。
这只改变共享配额下的作业交错机会，不改源码、目标、种子、方法或配对顺序。
此前承诺的恢复点不变：本轮源A与构造完成后恢复共视原上限4，当前仍待恢复。

15:00新A的scene rank2/3已分别在ga024/ga031启动，同时共视task58仍运行，
两条主线均在推进。源A已完成数仍为8，尚未把运行中的A或未构造查询计入最终结果。

### 六个完整A场景（15:12）

以下均有完整summary、independent verification和逐成员回读的archive receipt，
不是从partial progress推算成功数。rank0沿用已声明保留的完整开发场景，其余为本批数组。

| 原scene rank | scene | 完整A | 到达A |
|---|---|---:|---:|
| 0 | rJhMRvNn4DS | 4 | 2 |
| 1 | 6D36GQHuP8H | 4 | 3 |
| 2 | nrA1tAA17Yp | 4 | 3 |
| 3 | jgPBycuV1Jq | 4 | 1 |
| 4 | BFRyYbPCCPE | 4 | 2 |
| 5 | X7gTkoDHViv | 4 | 4 |
| 合计 | 六scenes | 24 | 15 |

17253441中五个新scene均COMPLETED/0:0；相应构造1..5为PENDING/QOSGrpGRES，
不是依赖错误。尚未读取或运行正式query结果，不据这部分A统计调整源前缀或阈值。

15:16新增rank6 / 5jp3fCRSRjc：四条A中两条到达，完整核验与归档回读通过，
累计28条A/17条到达。前六个scene的summary、verification和archive receipt已回传
本机`.diagnostics/repaired_fullmono_design_20260909/prefix30_progress/collection/`并复核，
总占用128KiB，未重复下载完整轨迹归档。正式query仍未运行。

15:20为让一个已完成A的构造先获得调度，尝试对原数组尚未启动的19个A元素短暂hold。
命令均返回`Unspecified error`；随后逐项回读，19项仍为PENDING/QOSGrpGRES、
Priority=12056，没有任何JobHeld状态，实际没有暂停A。未继续尝试其它优先级修改，
也未取消或重提任务。A无需release；共视临时并发1的既有恢复约定仍然有效。

15:56已逐项读取24个scene的完整summary与independent verification：96条A、75条到达，
全部verifier通过。源前缀仍是112条，rank26运行、27/28/29等待；构造仍仅rank0有终局。
当前没有新的三臂query SR。共视task64运行，0..63均已正常结束。
Table-II的旧实际A/B采集版本也已追到封存launcher与evaluator；论文内部版本矩阵
已补来源，不再只以最终C入口代替全流程来源。没有更改论文数字或TeX。

## 本机产物

`.diagnostics/repaired_fullmono_design_20260909/`：

- `source_inventory.json`：完整196任务清单及距离分布；
- `rJh_spatial.json`、`jgP_spatial.json`：真实NavMesh空间诊断；
- `initial_a/`：小型原始A trace/NavMesh副本；
- `addon_v1/`：只读构造源码包，receipt `b9297569ff0805d7`；
- `repaired_fullmono_b9297569ff0805d7.tar.gz`：21,709字节源包，无模型或数据集。

研究与论文文件均未commit/push。用户已有改动保留；摘要和引言的保护SHA见论文版本矩阵。
