# Table II：PT1 连续 Novel 场景容量本机预检

日期：2026-09-11。**离线构造诊断，不是新的导航 SR；没有更改现有 HPC pilot、论文数字或 GEM。**

## 1. 结论

PT1 的轨迹确实较长，但不能直接把旧 `Novel → Novel → Revisit` 路线改名为三个 Novel。
需要在设计阶段检查整条序列的**未见视觉空间余量**，而不只是累计路程或建筑面积。

本轮已经完成本地 40 scenes / 76 histories 的路程、几何供给与完整前缀共视预检：

- 76 条旧历史的累计实际记录路程中位数为 **23.72 m**，28 条超过 30 m。
- 固定候选预算下，**38/76 条历史找到了至少一个 C-Novel 候选**。
- 加上旧 A/B 记录末态之间的 2–9 m 条件、B/C 均有 Novel 候选，可保留 **22 条诊断来源，覆盖 18 scenes**。
- 这些仍然使用 **expert A/AB 前缀**。它们只提供新构造的候选场景，不是实际 mono NavDP 轨迹，也不是正式评测总体。
- A 的候选有约四分之一在初始 RGB 中就可见，而 B/C 的严格 Novel 必须没有历史支持；这项构造不对称需要和方向、距离一起处理。

因此下一步应从场景资产重新设计共同难度的目标构造，用实际 mono A/B 验证；不能仅依据“PT1 长”复用旧三段路线。

## 2. 范围与权限边界

输入是本地已有的 PT1 / 已审计 gapfill 监督副本，不是 PT1 全部轨迹，也不是新的 unseen confirmation：

```text
.diagnostics/anchor_relation_geometry_20260906/inputs/unpacked/supervision_only
```

逐条读取 `meta/gen_meta.json` 和保存的 `action` 位姿 parquet。76 条均为三段，后两目标的角色均为 `novel / revisit`。
源图形资产来自 `/home/asus/Research/datasets/mp3d`。

- 没有读取 native/GEM 的查询结果来选择这 76 条来源。
- 没有启动 NavDP、GEM 或学习模型推理，没有训练，没有机器人动作。
- NavMesh、GT 位姿和渲染深度只用于**离线构造及可观测性测量**，未进入任何策略请求。
- 没有把 expert 历史替换进 actual-online benchmark。
- 没有修改正在运行的 Table II 采样器、600-tick 预算、角色标签隐藏规则或方法阈值。

## 3. 路程复算

使用连续保存位置之差累计，不用帧数乘名义速度估算。坐标转换复用已经测试的
`parquet_data_pose_to_habitat`，处理旧 identity camera-extrinsic 约定。

| 量 | 中位数 | 最大值 |
|---|---:|---:|
| 三段累计记录路程 | 23.72 m | 90.50 m |
| A 段记录路程 | 6.23 m | 9.22 m |
| B 段记录路程 | 8.82 m | 42.17 m |
| C 段记录路程 | 9.42 m | 41.85 m |
| 元数据中的 A 最短路 | 5.97 m | 8.92 m |

其中 28/76 条累计超过 30 m。第一帧之前没有保存的移动不额外补算。
精确切分将 switch 边界的位移记到新一段；此前快速统计的 A 中位数约 6.25 m，本次按边界归属统一后为 6.23 m，总路程不变。

按空间位置测量，B 段在旧 A 路径 1 m 邻域外的路程比例中位数为 47.60%；
C 段在完整旧 A+B 路径 1 m 邻域外的比例中位数为 0%。这是空间诊断，不是视觉 Novel 标签。

最长 90.50 m 的历史，其 A/B 记录末态之间在重算 NavMesh 上的最短路约为 38.68 m；
它本来就不是当前每段 2–9 m 的等难度任务。不能直接用最长轨迹作为新三段评测来源。

## 4. 几何供给检查

每个场景从 GLB 重新计算 NavMesh：agent radius 0.30 m、height 1.5 m，和当前构造参数一致。
随资产附带的旧 NavMesh 半径为 0.10 m，未直接拿来作本次容量结论。
重算在无 renderer 的 Habitat 实例中执行，没有占用策略 GPU 推理。

每条历史检查三个状态：

1. 第一条保存位姿，使用独立采样的新 A 朝向；未执行新 A。
2. 旧 expert A 最后一帧，继承该帧朝向。
3. 旧 expert B 最后一帧，继承该帧朝向。

每个状态固定 4,000 次空间提案；目标最短路 2–9 m、目标净空至少 0.30 m、
整条最短路线相对起始高度差不超过 0.20 m；按 0.5 m 平面格去重。
不移动历史起点来改善供给。全部 228 个状态均成功测量。

| 状态 | 合规空间格数量中位数 | 有前/侧/后三种方向的历史数 |
|---|---:|---:|
| A 起点 | 78 | 31/76 |
| A 末态 | 74 | 37/76 |
| B 末态 | 74 | 29/76 |

在 B 末态，距离完整旧路径超过 2 m 的空间格数量中位数仍有 30.5。
但是“没走到这个点”不等于“没看过它所拍到的表面”，所以没有把这些格直接标为 Novel。

场景面积仅记录为所有楼层总可行走面积，不能当成当前同层可用面积。
同层结论来自逐候选连通性和整条路线高度检查，不来自建筑包围盒。

## 5. 重渲染后的视觉供给

先按几何供给选择两条历史作有限探针，C 仅得到 0/16、1/16 个 Novel 候选。
随后建立一个新的、固定的全量预检清单，覆盖全部 76 条；不因为某条结果好就停止或只保留它。
最初两条保留并明确作为重复诊断，不作为独立确认；两次对应候选的完整共视曲线逐项一致。

全量规则：每个非空的“距离桶 × 方向桶”取提案顺序最早的两个候选。
距离桶为 `[2,4) / [4,6) / [6,9] m`，方向为 front / side / rear。
每个目标只用事先固定的一个独立八档 yaw；本轮没有枚举方向直到找到成功目标。

使用相同 480×270 相机、原始内参和 0.5 m 高度，重渲染完整 expert A/AB 前缀。
每个候选与前缀**所有帧**逐一计算遮挡检查后的共视，使用现有共视函数和深度容差。
非空有效表面且最大历史共视 `<0.10` 才记为该 expert 前缀下的 Novel 候选。

共测量 **2,440 张目标图，228 个阶段状态**；全量渲染/共视检查约 **226.9 秒**。

| 阶段 | 测量候选图 | 无历史支持候选图 | 找到至少一个的历史 | 找到前向 Novel 的历史 |
|---|---:|---:|---:|---:|
| A：空历史 | 817 | 811 | 76/76 | 61/76 |
| B：相对完整 A | 846 | 317 | 75/76 | 25/76 |
| C：相对完整 A+B | 777 | 78 | 38/76 | 16/76 |

**本表不是 SR。**38/76 表示固定候选预算下发现了可构造目标；其余条目不能被解释为
“这个场景绝对不存在第三个 Novel”，更不能算成 NavDP 导航失败。

进一步的构造差异：

- A 的 811 张有效空历史候选中，只有 605 张同时满足当前视图共视 `<0.10`；
  其余 **206/811=25.40%** 在初始视图中已有至少 0.10 的共视。
- B 的 317 张、C 的 78 张合规 Novel 全部同时满足当前视图共视 `<0.10`。
  当前视图本身已经属于相应历史，因而这是条件上的差异，不是策略能力改变。
- B 的合规 Novel 图中 front/side/rear 分别为 **37/93/187**；后向占 **58.99%**。
- C 的合规图中 front/side/rear 为 **26/32/20**；不能把 C 的供给问题一概解释成后向偏置。
- 找到全部三个方向 Novel 候选的历史：A 31 条、B 10 条、C 0 条。
  因此不能要求每条历史填满所有距离/方向格；正式配额应检查整个固定来源池的共同供给。
- A/B/C 各发现 6/1/3 张完全没有有效表面点的候选图，本次全部排除。
  空表面的共视函数返回 0 不能当成有效 Novel。没有证据据此归因已有 HPC SR。

这支持：场景容量要按“还能生成什么目标视图”检查；扩大面积或避开旧路径本身不够。
它**不证明**场景容量解释了旧 SR 差距的全部，也不证明新在线轨迹一定复现相同供给。

## 6. 当前可继续的来源

在旧 A/B 记录末态最短路仍为 2–9 m、并且 B/C 都发现 Novel 候选的条件下，得到：

- 22 条诊断来源；
- 覆盖 18 个场景；
- 明细见 `independent_verification.json` 的 `spatially_screened_visual_followup_sources`。

这不是“22 条正式三 Novel 已经构造完”。来源仍是为 NNR 设计的旧 expert 前缀，
不应据其目标供给优劣断言场景对所有新 NNN 路线的能力。下一轮需重新构造 A/B，
不直接继承旧 B 末态及旧 C 目标。

## 7. 下一步设计，不修改已提交 pilot

1. 将整条序列的空间和视觉供给预检放到长闭环之前；PT1 只提供场景、相机与预定物理起点。
2. 新 A/B/C Novel 继续使用共同 2–9 m 距离范围，独立目标照片方向；B/C 保留实际末态朝向。
3. 正式版须明确共同的初始可见性条件，避免 A 可起步看见目标、B/C 却必须完全无支持。
   一种与现有 Novel 定义一致的方案是三段都要求当前共视 `<0.10`；这是新的构造条件，
   **尚未写入当前运行中的 sampler**，需在新协议中固定并核对供给。
4. 按固定来源池的共同距离/方向供给安排配额，不按导航成败挑场景，也不通过 GT 预转朝向。
5. 先用少量实际 mono A/B 本机验证，重新检查完整真实历史，再确定 C；旧 expert 可作预检，不能作 runtime memory。
6. 保留 B-N/B-R 分支隔离、C 的 native-B 参考前缀、两种 B 来源 50/50 及阶段条件分母。
7. 新样本如在 MP3D/PT1 构造，单独报告，不拼入 HM3D 分母；已消费场景不重新称 fresh/unseen。
8. 目标是可比难度，不是保证 A/B/C 的 SR 相等。新 A/B 实际走向会改变后续供给，构造损耗仍需完整保留。

## 8. HPC Table II 快照

2026-09-11 本轮检查时：

- A 已完成 8/8，native 成功 **5/8**；不是扩样正式 SR。
- 5 条成功 A 均构造出 B-Novel，其中 4 条同时构造出 B-Revisit。
- 这 5 条 A 末态中，4 条在该 pilot 的固定几何提案预算内没有前向合规候选。
- B 数组 `17348183_[0-15%2]` 等待 `QOSGrpGRES`；尚无 B/C 新 SR。
- C 来源选择 `17348184` 等待上游依赖；没有更换参数、重提交或取消现有任务。

远端只读收据：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/table2_mixed_pilot_20260911/A/task_*/summary.json
```

## 9. 代码、复算与结果入口

新增代码均与导航入口分离：

- `MemNavData/audit_pt1_multinovel_capacity.py`：旧轨迹复算、0.30 m NavMesh 和三状态空间供给。
- `MemNavData/probe_pt1_multinovel_support.py`：先固定候选清单，再完整前缀重渲染共视。
- `MemNavData/verify_pt1_multinovel_capacity.py`：独立读取 raw parquet 和共视曲线复算。
- `MemNavData/test_pt1_multinovel_capacity.py`：8 个测试通过，包括阶段边界不漏步、空深度不冒充 Novel、保留历史、同层路线和候选顺序。

完整结果根目录：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/pt1_multinovel_capacity_20260911_v1/
  trajectory_inventory.json
  spatial/summary.json
  visual_probe_plan.json
  visual_probe/summary.json
  visual_all76_plan.json
  visual_all76/summary.json
  independent_verification.json
```

独立验证：`verified=true`。所有 76 条 parquet 路程、228 个阶段、候选序号、完整共视曲线长度和接受计数复算一致。
本次图形资产加载仍打印旧 `.scn` semantic-descriptor 提示；本检查不使用语义标签，
40 个场景均完成 mesh 重算，228 个状态及全部渲染完成，未把提示当成导航失败或忽略实际失败。

最小复算命令：

```bash
/home/asus/miniconda3/envs/memnav/bin/python -m pytest -q MemNavData/test_pt1_multinovel_capacity.py
/home/asus/miniconda3/envs/memnav/bin/python -m MemNavData.verify_pt1_multinovel_capacity \
  --root .diagnostics/pt1_multinovel_capacity_20260911_v1 \
  --out .diagnostics/pt1_multinovel_capacity_20260911_v1/verification_recheck.json
```

复算文件使用 exclusive-create；已有同名结果不会被覆盖。未 commit 或 push。
