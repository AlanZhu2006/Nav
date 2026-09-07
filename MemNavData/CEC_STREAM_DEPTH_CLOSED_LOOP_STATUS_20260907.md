# 历史在线深度复用：本机闭环运行状态

2026-09-07，约 06:30–07:11 CST。**16/16 全部完成，配对与实际终点 SPL 复算通过；没有提交 HPC。**

本轮承接 `CEC_STREAM_DEPTH_ROLE_QUERY_RESULT_20260907.md` 的组件正结果，
不是启动新训练或更改论文方法。完整设计见
`CEC_STREAM_DEPTH_CLOSED_LOOP_PROTOCOL_20260907.md`。

## 已完成

- 4 个 consumed scenes 的 scene asset、原始 episode parquet、实际 online-A RGB
  和自然方向 role-pair 查询均在本地；不需要下载新的数据。
- 4 场景分别实际初始化 Habitat、渲染 270×480 RGB，并重算全部 8 条初始 geodesic，
  与冻结 metadata 差异均不超过原协议的 0.05 m。
- source/reset 绑定、NumPy 深度缓存计量、实际终点 SPL 及已有路径计量测试：14 passed。
- 独立 MemNav / NavDP 使用端口 21560 / 21561，完成后自动关闭；原有 18888 / 8888 服务未操作。
- 4 histories 的查询初始主流状态一致；16 条最后动作后坐标、路径、success 和 SPL 复算通过。

## 最终结果

| 指标 | canonical replay | online history |
|---|---:|---:|
| Revisit SR | 3/4 | 3/4 |
| Novel SR | 0/4 | 0/4 |
| Revisit 平均实际 SPL（含失败） | 0.66303 | 0.65403 |
| Revisit 首次证书阶段中位耗时 | 20.330 s | 0.126 s |

8 个配对查询为 +0/−0，不能用 N=4 scenes 宣称统计等效。
4 个 Novel 全程拒绝，两臂的物理轨迹及终点完全相同；没有另加 native 第三臂。
4 个 Revisit 均授权，3 个成功；pLe4wQe7qrG 两臂均在距目标约 1.839 m 处 stuck。
Revisit 轨迹存在微小分歧，不能将此次变更称为数值无损缓存。

online 查询开始时 CPU depth/conf cache 为 397–815 MiB；首次证书耗时包括该函数内
匹配／PnP，但不含外部 DINO probe、NavDP、通信和机器人执行，不能称完整端到端耗时。

## 已执行协议

4 histories × 2 natural queries × 2 depth sources = **16 rollouts**。
600-step query budget，8-action horizon，1 m position success。两臂都使用 mono
observation depth，只切换历史深度重放／在线保存来源及对应 writer。

原始 A 是既存 metric-NavDP 的实际轨迹，query 使用 mono，因此不能称本轮重新证明
了 full-mono-A。Novel / Revisit 标签不送入 runtime。

## 输出和检查入口

输出根目录：

`.diagnostics/cec_stream_depth_reuse_20260907/closed_loop_v1/`

- `manifest.json`：样本、臂顺序、预算、源代码／权重 SHA。
- `owned_processes.json`：本轮自己的模型进程 PID，不包含真机服务。
- `logs/preflight.log`、`logs/memnav.log`、`logs/navdp.log`、`logs/<index>_<source>.log`。
- `evaluation/<scene>/<source>/certificate_calls.jsonl`：逐规划原始定位和计时收据。
- `terminal_measurements.json`：每条导航结束立即保存末动作后坐标及实际 SPL。
- `metric_actual.csv`：完成该臂两条查询后输出；原始 `metric.csv` 仍保留，不覆盖。
- `independent_verification.json`：`verified=true`，16 records／8 pairs／4 histories。
- 若出现 `failure.json`，应先读对应日志；不把未完成数据算成有效 SR。

执行代码是独立的 `MemNavData/run_cec_stream_depth_closed_loop.py`。实验 hook
只存在于新启动的进程中，未修改生产 policy/server/evaluator 默认逻辑。

本轮未 commit / push，未改论文，未重启真机。后续用户已同意把在线深度复用整合为
新主线配置，并同步真机代码；这不改变本轮冻结脚本和旧论文数据的实现来源。
集成说明见 `CEC_ONLINE_HISTORY_INTEGRATION_20260907.md`。
