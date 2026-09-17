# 在线几何复用：跨视角与无匹配查询的本机结果

日期：2026-09-07。全部完成，38 个分臂记录独立 CPU 复算通过。

## 结论

**可以继续做小规模配对导航，但现在仍不能替换正式 CEC 或挪用旧 SR。**

把初始 CEC 的参考深度从 historical dense replay 换为历史时刻保存的主在线深度，
在本轮 **4 histories／4 scenes／19 unique queries** 上没有改变授权决定：

| 查询类型 | N | canonical replay 接受 | saved online 接受 |
|---|---:|---:|---:|
| Revisit | 9 | 9 | 9 |
| Novel／低历史支持 | 10 | 0 | 0 |

**这是定位／授权结果，不是导航成功率。** 本轮没有运行 NavDP trajectory decoder
或执行导航动作；`navigation_SR=null`。其中 `plan(retrieval_only=True)` 仅用于
核对检索接口返回的 shortlist，不执行导航。

9 个接受查询的输出 bearing 差异：

- 中位数：**0.0686°**；
- 均值：0.2170°；
- 最大值：**0.8537°**，来自一个低共视 Revisit。

两条低共视 Revisit 都保持接受。接受数相同不意味着 PnP 数值相同：inliers、
reprojection error 和深度确实会变化。这个输入变体仍需闭环检验。

## 1. 相比上一轮，补上了什么

上一轮只给预选单 anchor，目标还是历史中真实出现过的 JPEG。本轮：

1. 目标来自已经消费过的重渲染构造，19 张目标均不与自身历史 JPEG 的 SHA 重复。
2. 使用完整 DINO top-8、temporal gap=4，再由原有 SP／LightGlue 和几何排序选择
   anchor；不是手工指定正确位置。
3. 包含 10 个无／低历史支持目标、9 个不同视角 Revisit；其中 2 个 Revisit 的
   session max-covis 约 0.4，而不是上一轮全部近邻图像对。
4. 使用完整 201／213／240／405 帧实际 online-A RGB prefix，而不是只到预选
   anchor + 40。它们仍是已消费诊断历史，不是新的 held-out 确认集。
5. 查询前以 stride=1 保存全部可查询历史帧的深度，不再只保存事后指定的一帧。

原始三套构造在历史内按目标 SHA 去重得到 19 条。不同构造只当作静态定位查询，
不混算 natural-direction 与 support-controlled 的导航 SR。

特别注意：角色和 covis 只参与清单／分层报告，没有输入检索器、匹配器或证书。
current 使用最后一帧已观察 RGB 对应的位姿，不使用原 trace 最后动作后的未观察终点。

协议：[CEC_STREAM_DEPTH_ROLE_QUERY_PROTOCOL_20260907.md](CEC_STREAM_DEPTH_ROLE_QUERY_PROTOCOL_20260907.md)。

## 2. 配对与验证

- 同一个 GPU／模型进程，baseline ingest 与 writer ingest 交替顺序、相同 seed。
- 所有候选的 SP／LightGlue 只算一次；两臂读取相同的落盘匹配数组。
- 同一参考／当前 pose、PnP seed、16 inliers／0.05 hull／2 px 证书。
- 4/4 histories 的主 KV、DINO 特征、camera poses、最后聚合特征及 first40 receipt
  一致，first40 scale 均有效；查询也没有改变这些主在线状态。
- CPU 独立进程从保存的 152 组匹配数组重建几何排序，再从保存的 depth／pose
  重算 PnP、certificate 和 bearing，**38/38 arm records 一致，verified=true**。

复算使用同一冻结 PnP 实现，不是另一套独立实现的定位算法。DINO shortlist 在 GPU
驱动中与 `plan(retrieval_only=True)` 返回核对；CPU verifier 核对的是已冻结 shortlist
后的证据，不重新运行 DINO 网络。

排序在读取深度前完成，所以 selected anchor 相同是这次控制变量设计的预期，
不能算作一项独立性能增益。

## 3. Novel 拒绝到底测到了哪里

10 条 Novel 中：

- 4 条在 Fundamental／coverage precheck 已拒绝，两臂都没有调用参考深度；
- **6 条实际进入 PnP**，更换深度后仍未获得授权。

因此并不是所有负例都绕过了被修改的模块；但前 4 条也不能被用于声称在线深度
提高了安全性。0/10 只是一项小型构造集观察，不是零误激活保证。

Revisit 的 GT 直线 bearing 误差（仅离线报告）：

| 深度来源 | 均值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| canonical replay | 2.2314° | 1.1865° | 8.0496° |
| saved online | 2.0472° | 0.8961° | 8.0279° |

不能把这点均值差称为定位提升。重要的观察是：在这批非重复目标图上，深度变化
没有造成明显的新增方向失真或授权改判。

## 4. 成本：主要减少新 anchor 的首次重放

15 条实际调用 PnP 的查询都首次使用各自选定 anchor 的 canonical depth cache。
其余 4 条预检查拒绝不参与下表：

| 阶段计时，N=15 | 中位数 | 范围 |
|---|---:|---:|
| canonical 证书查询 | 17.183 s | 0.693–60.270 s |
| saved-online 对应证书查询 | 14.61 ms | 9.29–30.95 ms |

这些计时 **不含 DINO 检索、SP／LightGlue、NavDP 和通信**；CUDA 已同步。
模型初始化和状态审计也不计入。

本轮 DINO 检索中位数 19.67 ms，完整 top-8 图像匹配中位数 108.18 ms。
逐查询把“检索＋匹配＋saved-online 证书”的已测分项相加，19 条的中位数
为 **138.92 ms**，范围 98.79–231.09 ms。这只是分项和，**不是独立实测的机器人
端到端延迟**，也不包含 NavDP 执行。

同一个 anchor 的后续 canonical 查询原本也能命中缓存，不能把这个首次收益套到
每个 replan。相较旧 eager 双流缓存，本轮没有维护第二份 dense aggregator。

## 5. 全历史保存成本

主流通常已经为 flow-gate 计算过 depth，writer 复用这一输出，只新增保留／CPU
拷贝；没有每帧重跑一份完整 LingBot。实测如下（单次交替顺序配对）：

| 历史 | RGB 帧数 | 普通 ingest | writer ingest | 缓存帧数 | CPU depth/conf 缓存 |
|---|---:|---:|---:|---:|---:|
| gxdoqLR6rwA / 0000 | 240 | 44.42 s | 44.09 s | 232 | 474.94 MiB |
| mJXqzFtmKg4 / 0001 | 201 | 39.40 s | 39.59 s | 193 | 395.10 MiB |
| pLe4wQe7qrG / 0000 | 405 | 90.14 s | 90.52 s | 397 | 812.72 MiB |
| yqstnuAEVhm / 0001 | 213 | 44.31 s | 44.40 s | 205 | 419.67 MiB |

共 1,059 帧，普通 ingest 218.27 s，writer 218.60 s。这个单次差值不支持宣称
零开销或写入更快；它只显示本轮主要代价不是重复网络推理，而是随帧数线性增长的
CPU 存储。每帧约 2.05 MiB，1,000 个缓存帧约 2 GiB（线性估算）。

缓存从第 8 帧开始覆盖当前运行时代码可查询的历史帧；它不同于前一轮只在一个
预选 anchor 上额外调用 depth head 的 13 ms microbenchmark。两轮不能直接混成
统一的“每帧开销”。

## 6. 对架构的实际意义与下一步

这条路线的价值是：**历史 RGB、位姿和深度可以在观察时绑定保存，查询时读取
同一主流已经产生的几何**，避免为每个新 anchor 临时重放历史。它是简化实现和
减少重复计算的证据，不是新的训练贡献，也不代表已解决长程漂移／绕障。

下一步最有信息量的是小型本机导航，而非继续扩大纯离线接受率：

- 使用这 4 条已消费历史中的 8 个 natural-direction 查询（4 Novel、4 Revisit）；
- canonical CEC / saved-online CEC 两臂，同一历史与起点，共 16 个 rollouts；
- 相同 monocular depth、NavDP、固定 2.5 m residual、执行与成功判据；只改变
  参考深度来源及其写入缓存；
- 记录 SR、实际路径、末端坐标／SPL、首次接管延迟和缓存量；
- 作为工程／机制闭环验证，不作为新的正式泛化 SR。

Agent 已有 `reference_depth_source` 实验参数，但当前研究仓库的 HTTP
`/certified_relocalize` 入口尚未转发它；后续闭环需要一个显式实验臂，**不能以为
现在正式服务器已切换方案**。本轮未改 server、evaluator、正式 CEC 默认值或论文。

即使小闭环通过，也不能把旧 SR 自动写成新版本的 SR；是否扩大确认应再由成本
收益和稿件需要决定，不必立刻重跑全部论文表格。

## 7. 工件与其他任务

- 驱动：`MemNavData/benchmark_cec_stream_depth_role_queries.py`
- 独立复算：`MemNavData/verify_cec_stream_depth_role_queries.py`
- 输出根：`.diagnostics/cec_stream_depth_reuse_20260907/role_queries_v1/`
- 清单／全量结果：`manifest.json`、`summary.json`
- 复算：`independent_verification.json`，38 records，verified=true
- 逐查询保留 top-8 matches、源文件 SHA、两臂 receipt 与使用过的 anchor depth。

与上一轮 3 个场景有重叠，不能声称“两轮累计 7 个独立场景”。本轮进程已正常退出，
没有停止或改动其他常驻服务，没有提交新的训练／HPC 作业。

2026-09-07 05:40（CST）通过 `alantorch` / `yz11502` 共享 PTY 核对：Table III
exact-SPL 仍为 5/21 histories 完成，16 条 `QOSGrpGRES` 排队，summary 等待依赖，
没有新增失败。待整批复算通过后才把新 SR＋SPL 一起更新论文。
