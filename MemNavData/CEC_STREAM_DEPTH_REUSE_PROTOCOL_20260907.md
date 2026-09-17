# 主在线几何能否复用于 CEC 初始定位：本机配对诊断

日期：2026-09-07。范围：论文／仿真侧；不更改真机、主方法、论文或训练。

## 问题与既有工作的区别

当前初始 CEC 证书默认使用 dense historical replay 的参考深度。已有 eager
cache 通过维护第二份 dense 状态保持等价，但并未去掉重复几何计算。本轮只检验：
能否把**同一主在线流在历史时刻产生的深度**保留下来，直接用于初始 PnP。

运行代码已有 `route_sparse` 实验入口。它不是当前正式初始证书的默认输入；即使
本轮成功，也不能直接继承 canonical CEC 已有的闭环 SR。

## 冻结的最小构造

- 复用 8 月 18 日延迟测试已消费的三条 RGB 历史，不读取新测试集。
- `gxdoqLR6rwA/episode_0000`：anchor 80，current 120，目标帧 79／75／60。
- `pLe4wQe7qrG/episode_0000`：anchor 120，current 160，目标帧 119／115／100。
- `yqstnuAEVhm/episode_0001`：anchor 160，current 200，目标帧 159／155／140。
- 保留原 episode length、seed、flow-gate；相机高度 0.5 m，first40 scale 不变。
- 目标是历史 JPEG，候选 anchor 预先指定；**不是新 Revisit benchmark，也没有
  Novel 负例**。三个目标之间相关，不能把 9 对当作 9 个独立场景。
- current 固定为 anchor + 40，避免把紧邻 anchor 的零距离方向用作主要证据。

## 配对与测量

1. 同一个本机 GPU／模型进程重放同一 RGB prefix 两次：普通 ingest；在指定
   anchor 额外 materialize 主在线深度。跨历史交替顺序，reset 同一 seed。
2. 比较主流 KV、camera poses、DINO 特征、最后聚合特征以及 first40 receipt。
   若不一致，停止并报告，不把状态变化混入深度来源比较。
3. 每个图像对只运行一次 SuperPoint／LightGlue，保存匹配数组；两臂收到相同
   候选、匹配、PnP seed、证书阈值、相机 pose。只切换 `canonical`／`route_sparse`。
4. 测 depth／confidence 差异、accept/reject、原因、anchor、输出 bearing 差。
   GT 仅从既存 trace 离线核验图像 SHA 后评分，不输入任何模型。GT 平面距离
   小于 0.5 m 的方向误差标记为病态，不计作有意义的角误差。
5. 分开报告写入开销、CPU 缓存、第二份 dense GPU 状态、共享匹配时间、查询
   时间。CUDA 同步计时；canonical 首次查询为冷缓存，后续同 anchor 查询是热缓存。

本轮只 materialize 每条历史的一个预选 anchor，不代表全历史缓存的摊销成本，
也不代表机器人重规划延迟。状态 hash 的审计成本从模型操作计时中排除。

## 决策边界

- 结果改变：这是估计器变体，不是透明缓存优化；不得替换主方法或挪用旧 SR。
- 结果相近：只说明值得在更有挑战的已消费／训练定位样本上继续验证；没有新
  SR、没有 Novel 安全性证明，不自动提交长闭环或长训。
- 目前优先保证可复算的最小结论，而非宣称一定能合并两种几何读出。

实现：`MemNavData/benchmark_cec_stream_depth_reuse.py`。
输出根目录：`.diagnostics/cec_stream_depth_reuse_20260907/pilot_v2/`。
`manifest.json` 在模型运行前生成，保存 RGB、trace、源代码与固定选择的摘要。

`pilot_v1` 在 manifest 构建时因归档 JPEG 使用六位编号而退出，没有加载模型，
没有任何方法结果。修复文件名解析后保留相同样本选择，从新目录运行。
