# 在线历史深度复用：真实跨视角／无匹配查询配对

2026-09-07；本机组件诊断，不改变正式方法、论文或真机，不运行 controller。

## 为什么要做

上一轮 3 历史／9 对相邻历史 JPEG 的证书相同，不足以说明跨视角／开放集查询也
保持一致。本轮用已消费的真实 online-A 历史和独立重渲染目标图，固定完整 DINO
top-8、SP／LightGlue 匹配、PnP 和证书，只替换参考深度来源。

## 固定数据

原始历史为 `shared_online_a_v0v1_pilot_native_20260812` 中全部 4 条：
gxdoqLR6rwA/0000、mJXqzFtmKg4/0001、pLe4wQe7qrG/0000、yqstnuAEVhm/0001。
分别 240、201、405、213 RGB 帧，保留原 seed、episode length 与 0.5 m 相机高度。

查询取以下三个已消费构造目录的全部现有 query，按历史内目标 SHA 去重：

- `shared_online_role_pair_natural_heading_v1_smoke_20260814`；
- `shared_online_role_pair_heading30_v3_smoke_20260814`；
- `final14_population_v3_consumed_20260817/finalizer_consumed2_run/benchmarks/hard_support`。

预期 19 个唯一查询：9 Revisit（其中 2 个低共视构造）、10 Novel。去重、角色和
共视度只用于构造清单及报告，不输入任何 matcher、排序或证书逻辑。所有 query
SHA 必须不在自身 RGB history 中，不能重复上一轮 exact-JPEG 测试。

这是旧构造 smoke 的机制复用，不是重新开放当前 Final14 正式测试集。不同协议
的查询只作为固定图像定位样本，不混算这些协议的导航 SR。

## 配对与资源

1. 单模型同进程分别 ingest 普通流、stride=1 历史深度保存流，交替顺序。保存发生
   在每帧实际被观察时、目标查询前；不是看到目标后倒选要保存的 anchor。
2. writer 使用已有在线 flow 计算的 depth/confidence，缓存全部可查询帧，没有第二
   份 dense aggregator。记录全历史 CPU 成本与 ingest 时间，不外推单帧成本。
3. 两次主在线状态及 first40 receipt 必须相同；否则停止输入来源比较，先归因。
4. current 采用最后一帧已记录 RGB，不冒充原 trace 最后动作后的未观察终点。
5. 使用当前生产 DINO top-8／temporal gap=4 的全 shortlist，并逐查询与
   `plan(retrieval_only=True)` 的 shortlist 核对。SP／LightGlue 每对候选只计算一次。
6. 同一个 shortlist、匹配数组、reference/current pose、PnP seed、16-inlier／
   0.05 hull／2 px 证书，比较 canonical replay 与 saved-online 两臂。交替顺序。
7. geometry-first 排序发生在深度之前，因此 selected anchor 在此配对中本来就
   应相同；不能把它当作又一项独立正结果。关键是证书改判、PnP 与 bearing 差异。
8. GT pose／covis 只从既存 metadata 离线评分；不加载 simulator depth，不调用 Habitat。

时间拆分：ingest、检索、匹配、证书查询分别报告；CUDA 同步。canonical 同 anchor
可复用缓存，不能将所有 query 称为冷启动。额外的状态 hash／复算不算部署耗时。

## 结论边界

- 相同决策不能证明同等 SR；Novel 的 precheck 若已经拒绝，也不能说新深度
  改善了 Novel 安全性——其本轮根本没有被调用。
- 若出现改判，逐条核对对应 PnP、coverage、误差，不调整证书阈值来抹平差异。
- 即便全部相同，也仅支持进入小规模闭环；不挪用旧 CEC SR，不启动长训。
- 405 帧是比首轮更长的 prefix，不等于已经验证 20–30 m 长程导航。

实现：`MemNavData/benchmark_cec_stream_depth_role_queries.py`。
输出：`.diagnostics/cec_stream_depth_reuse_20260907/role_queries_v1/`。
