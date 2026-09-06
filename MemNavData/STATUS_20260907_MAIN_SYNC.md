# CEC 项目总账与 main 归档（2026-09-07）

这是本轮仓库整理的最新入口。详细代码地图见
[REPOSITORY_GUIDE](../docs/REPOSITORY_GUIDE.md)，实验数字与来源见
[EXPERIMENT_INDEX](../docs/EXPERIMENT_INDEX.md)。旧日期文档保留其当时快照，
不再从旧文档的“running/pending”推断现在的作业状态。

## 1. 现在的项目是什么

**Certified Episodic Compass：因果单目 RGB 历史为冻结导航策略提供当前深度与历史目标方向。**

同一 RGB 流产生两个不同对象：可寻址的 RGB/DINO 历史，以及 LingBot 连续几何状态。
前者回答目标可能对应哪段历史；后者提供当前深度、历史深度与相机关系。
目标经局部匹配与 PnP 验证后，CEC 只把方向变为固定 2.5 m PointGoal，原始 ImageGoal
仍保留。NavDP 的冻结 diffusion decoder/critic 产生轨迹，CEC 不直接发动作。

短程分支使用 first-40 因果观测与相机高度先验形成尺度 receipt；warmup/无效 receipt
给 zero depth。长期分支使用 DINO top-8、时间分散、SP/LightGlue、Fundamental-MAGSAC
排序及单候选 PnP。certificate 固定为有效 PnP、inliers≥16、双侧 hull≥.05、RMSE≤2 px。
证据不足时保持 native 请求；共享 geometry transaction 损坏时停机，二者不同。

主方法没有 task-specific training，不把 Novel/Revisit 标签输入运行时，也不是两个策略
竞争动作的 mixture-of-experts。代码里保留的 learned、metric-distance、route-tangent、
U-turn/oracle 路线是归因与探索分支，不自动成为论文部署方法。

## 2. 已成立的结果与剩余边界

| 证据层 | 当前结论 |
|---|---|
| 跨 controller/dataset | HM3D/MP3D × NavDP/ViNT 四组正式配对均有 Revisit 增益；具体 N、+/−、p 见实验索引 |
| 完整单目 | actual mono A + mono queries：native17/56、CEC32/56；不再仅有 query-mono |
| 连续第三目标 | conditional Revisit-C8/20→17/20；Novel-C4/20不变；不是3段 joint17/20 |
| 深度消融 | Final14 mono CEC28/42、metric CEC26/42；Revisit均20/21；不能据此宣称 mono 严格优于metric |
| CEC 与 raw 的差别 | 记忆收益可靠；strict certificate 的额外总 SR 优势并未普遍成立，最新 HM3D raw35/56、CEC32/56 |
| 空间长距离 | 原48条受控长度诊断和后续23条同层实验已完成，但未证明20–30 m可靠导航 |
| 学习替代 | 有表征/定位诊断，没有已确认可替代生产几何的 learned relocalizer |

近期主要主查询约2–9 m；长历史保留和大空间绕行是两个问题。相机高度校准不能消除
相对旋转漂移，也不等于拥有可行绕障路线。长程 oracle 正式16-history批次未完成，
不能把失败完全归因于 NavDP 或 LingBot。

## 3. 9月6日至7日完成的稿件计量修正

活动论文：`/home/asus/Research/Memnav_Paper/main.tex`。本研究目录内旧 `paper/` 不参与
当前 Overleaf 主构建。本轮 Git 任务不修改或推送独立论文仓库。

1. Table II：459条记录有终点，已按真实平面位移重算。Revisit-C SPL从native .149到CEC .762；SR不变。
2. Table III：210条旧记录缺最后一步坐标，不能把指令步长当精确 SPL。
   应作者要求保留该列，当前显示向外取整的 SPL bounds：

   | 条件 | SPL bounds |
   |---|---:|
   | Metric native | [.118,.120] |
   | Zero native | [.060,.061] |
   | Mono native | [.121,.123] |
   | Metric CEC | [.443,.454] |
   | Mono CEC | [.492,.504] |

   这是缺失末步的数值边界，不是统计置信区间。待新补跑整批核验后，用新 SR+SPL 一起替换。
3. Analysis：`69/479` 是相对 geodesic 首段的差，实际目标直线 bearing 的 >90° 错误为`20/479`；已纠正比较对象与数字。
4. Results：完整 HM3D 四臂消融已紧凑写入，不只保留有利于 strict CEC 的旧结果。

稿件在前述修正及保留 SPL 后已编译为8页，无未定义引用或图表溢出；标题、Abstract、
Introduction、公式和引用保持原作者版本。详见
[实施记录](PAPER_EVIDENCE_CLOSURE_20260907.md)与[保留 SPL](PAPER_SPL_RETAINED_20260907.md)。

## 4. 刚提交的 HPC 补跑

作者要求先补跑再做 Git 整理，本轮已按此顺序提交。

| Job ID | 内容 | 2026-09-07首次提交后核验 |
|---|---|---|
| 17057430 | CPU 环境/冻结依赖预检 | COMPLETED，1分25秒，exit0 |
| 17057431，index17 | 最长prefix的完整五臂验证 | PENDING，QOSGrpGRES |
| 17057432，0–16/18–20 | 其余20 histories，每history五臂 | 等待上一项成功，最大4并发 |
| 17057433 | 全量 summary 与独立原始计分核验 | 等待评测结束，缺块不输出完整结果 |

固定21 histories /10 scenes、42 queries、210 rollouts。保留原 metric-A 历史，不重新采集。
H100/A100、1GPU/10CPU/72GiB、每history1小时；逐帧 buffer 写节点临时盘，归档后保存。
新增代码只在完整 rollout 返回后读取末位姿及轨迹，原 frozen policy、请求和内部计数
保持不变；旧 serializer CSV 在新输出目录内另存，不覆盖历史结果。

预检包含本机及 HPC 各8项计量测试、实际模型/renderer依赖、全部21份source/trace/
scene/parquet核验和权重哈希。**预检通过不等于GPU闭环完成，目前没有新SR或SPL。**

协议：[FINAL14_TABLE3_EXACT_SPL_REPLAY_PROTOCOL](FINAL14_TABLE3_EXACT_SPL_REPLAY_PROTOCOL_20260907.md)。
收据：[FINAL14_TABLE3_EXACT_SPL_REPLAY_SUBMISSION](FINAL14_TABLE3_EXACT_SPL_REPLAY_SUBMISSION_20260907.json)。
结果根：`/scratch/yz11502/Research/Nav-axis-uturn-results/final14_table3_exact_spl_20260907`。

## 5. 本机学习支线的真实状态

V0固定anchor关系训练已完成，训练集可拟合，但20个内部验证pair的平均位置误差1.3054 m，
明显未达到现有几何参考。本轮没有因其可训练就升级为主方法。

后续几何条件自动链在本次只读检查时，`workflow_v1/status.json` 为：

```text
stage: verify_overfit8
status: failed
exit_code: 1
```

日志为 `verify_anchor_relation_geometry_probe.py` 的
`math.isclose(observed, reference, abs_tol=1e-8)` 断言失败。只能确定报告/独立复算
不一致，尚未归因到精度、公式或实现问题；不能直接放宽容差，也不能称泛化三种子实验已完成。
本轮仅归档代码和真实状态，没有改动这个独立研究分支或重新启动训练。

## 6. 此次“完整整理”具体包括什么

- 根 README 升级为当前 CEC 入口，明确 DINO archive 与 LingBot state 的独立角色。
- MemNavData README 改为研究/评测导航；旧 expert 数据生成说明逐字保留为
  `README_DATA_GENERATOR_LEGACY.md`，防止历史协议与正式 online 实验混淆。
- 新增代码地图与实验索引，区分论文主证据、负结果、未完成工作和学习分支。
- 归档长程构造/归因/可视化、相机高度尺度、动作/SE2路线实验、Pi3X/anchor relation
  probe、HM3D四臂补跑、SPL计量修正、单元测试与提交收据。
- 已有主方法和各探索开关代码原样纳入版本管理；没有重新定义方法来追求表面整齐。
- 补齐3个旧测试文件的构造器替身字段，不改变生产默认逻辑；不会影响已冻结HPC bundle。
- 补充 cache/凭据/环境文件忽略规则；不上传checkpoint、数据、逐帧输出、视频、密钥或诊断目录。
- 不移动冻结脚本，不删除旧结果，不强推，不覆盖远端已有main提交。

开始前文件备份位于 `.diagnostics/git_main_sync_20260907_XQXM5q/`：
`before.patch`、`before_working_files.tar.gz`、`inventory.json`和测试报告均保留。
首次盘点342个变更文件约3.17MB，无>1MB文件、无高置信度secret-pattern命中；随后新增索引
文档。最终Git与回归验证收据另列，不把本段数量当作最终提交文件总数。

## 7. 接下来的顺序

1. 等当前SPL补跑完成原始核验，再同步Table III的新SR与精确SPL。
2. 保持论文证据主线：因果单目记忆提高冻结策略的Revisit；不要把strict-over-raw
   未成立写成普遍优势。
3. 学习分支先解决独立复算一致性；长程分支先补全定位/路线/controller归因。
4. 这些研究分支与当前投稿主表解耦，不因此重跑所有benchmark；真机继续由独立工作线负责。
