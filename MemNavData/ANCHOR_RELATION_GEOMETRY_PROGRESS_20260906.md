# 真实历史几何定位对照：实现与运行进展

快照：2026-09-06 23:17（Asia/Shanghai）。**已通过真实数据 smoke，完整几何提取正在本机运行；
新的有/无几何训练尚未完成，没有新 SR，也没有替换生产 CEC。**

协议：[ANCHOR_RELATION_GEOMETRY_PROTOCOL_20260906.md](ANCHOR_RELATION_GEOMETRY_PROTOCOL_20260906.md)。
上一版：[ANCHOR_RELATION_LEARNING_RESULT_20260906.md](ANCHOR_RELATION_LEARNING_RESULT_20260906.md)。

## 1. 本次已完成

- 按 HPC 共享 SSH 手册复用 `alantorch/yz11502`。默认 master 可用；no-PTY identity
  command 超时后使用同一 master 的 PTY，身份和主机已核对。没有要求重新登录、
  复用其他账户或关闭 master；本次 PTY 已退出。
- 从旧 PT1 / 已审计 gapfill 中解析固定 123 对样本对应的 74 条历史，未更换样本。
  首次计划检查发现 YmJ 场景不在原 PT1 overlay；原 collector 的 sbatch 明确额外
  挂载该历史补全场景。恢复这个既有只读挂载后，全部 query/anchor JPEG 哈希匹配。
  没有重新生成任何 episode，也没有把缺失样本删除。
- 只传输 13,841 张必要历史 RGB、少量历史相机预测、因果尺度收据和独立监督审计文件，
  不传全部 PT1、场景资产或多 GB KV。最小归档通过共享 SFTP 传输：
  696,391,680 bytes，SHA `7fef76a04b13036a3eb96b43da44348f1613c62fc83e7201ee6e7c1ad17fbb37`。
- 本机逐项验证 13,995 个 payload 文件；123 对 GT anchor-to-goal 位置从原始 parquet
  和 goal metadata 独立复算，**最大差值 0.0 m**，没有新增标签定义。
- 几何提取只输入历史 RGB。首8帧初始化，其后逐帧完整 replay，保持 window32；
  每个 anchor 当时的深度立刻缓存，不使用后续帧改写它，也不输入 goal RGB。
- 与 DINO 的 pad-to-518、14×14 patches、37→8 adaptive pooling 对齐；新增有效
  mask，在 attention 和 pooling 中排除无效位置。有/无 XYZ 两臂共用同一 mask。
- 米制换算沿用旧诊断的因果 first64 相机高度收据，全部 query decision>=90；
  不使用全轨迹地面尺度，也不冒充生产系统的 first40 标定。

## 2. 已完成的真实几何 smoke

只验证输入与提取链，不评价导航或模型泛化：

| 历史 | anchor | 有效 coarse tokens | anchor 深度归一化对应米数 |
|---|---:|---:|---:|
| 17DRP / episode_0000 | 125 | 32/64 | 1.8485 m |
| 17DRP / episode_0000 | 211 | 32/64 | 3.8148 m |
| 17DRP / episode_0001 | 113 | 32/64 | 2.6012 m |

- 2条历史、326帧，共56.4秒提取计算；不包括权重载入、下载或训练。
- 提升/投影往返最大误差约 `3.05e-5 px`。这证明坐标实现一致，不代表预测深度的
  真实误差只有这么小。
- 已目视检查真实 RGB、预测深度、padding/有效区域。
- 图：[geometry_witnesses.png](../.diagnostics/anchor_relation_geometry_20260906/geometry_smoke/geometry_witnesses.png)。
- 32/64 是原始270×480图像 pad 后的内容占比，不是几何定位 recall 或 certificate 通过率。

## 3. 当前运行及后续自动步骤

本机 workflow PID `949653`，当前提取子进程 PID `949654`；这仅是启动时身份，
之后状态以 receipt/log 为准，不应凭此文档假定进程长期存在。

23:17快照：smoke 2/2完成，余下提取9/72完成，即总计11/74历史已完成。
提取日志持续推进，父子进程均在运行；23:16 GPU 快照为100%利用率、约36.4 GiB显存。
按已完成历史的逐帧成本估算，剩余提取约35--45分钟，实际受GPU共享和历史长度影响。

自动链：

1. 剩余72段历史提取；
2. 原 train split 中8-pair过拟合检查：两臂、seed11、600steps；
3. 独立重读预测复算；
4. 固定103/20 pairs、32/8 scenes，对有/无 XYZ 两臂各跑 seed11/23/37、1200steps；
5. 独立重读CSV核对位置/方向/有效分母与scene-macro统计。

两臂共用目标归一化、mask、基础参数初始值、batch次序和米制loss；XYZ臂额外有几何
编码层。因此新 `visual_normalized` 才是直接输入对照，不能把它当上一版完全未变的
米制纯视觉模型。任何改善仍须结合多种子、scene分布和几何参考判断，不自动授权部署。

核心输出目录：`.diagnostics/anchor_relation_geometry_20260906/`

- `input_verification.json`：原文件与GT监督核验，verified=true。
- `geometry_smoke/`：已完成的两条真实历史。
- `geometry_remainder/`：剩余历史，逐历史持久化NPZ与progress。
- `workflow_v1/status.json`、`extract_remaining.log`：当前阶段与实际输出。
- `overfit8_geometry_v1/`：完成提取后生成。
- `scene32_8_geometry_v1/`：完成提取后生成正式内部对照，不是论文正式评测。

本次没有提交 HPC GPU 作业，没有重新评测 NavDP，也没有启动长模型训练。
耗时主要是一次性历史特征生产，不是小 head 优化，更不是部署查询延迟。

## 4. 测试与兼容性

- 新几何/原关系模块及旧 CDEC、Pi3X 相关测试合计 **20 passed，1个既有 PyTorch warning**。
- 语法检查通过，`git diff --check` 无问题。
- 原 V0 checkpoint 仍可 strict load；CPU复算20条验证预测与旧GPU保存输出最大差
  `1.31e-6 m`，不声称跨设备逐位一致。
- LingBot构造期打印 `flashinfer not available` / 空 `pretrained_path` 载入提示，
  随后完整指定checkpoint严格匹配通过；此次使用SDPA，真实几何计算成功。
  不把这些已解释的构造提示隐藏或误报为权重未加载。

## 5. 代码与改动边界

- `prepare_anchor_relation_geometry_inputs.py`：原始依赖打包。
- `audit_anchor_relation_geometry_inputs.py`：独立源文件/GT标签核验。
- `anchor_relation_geometry.py`：padding、投影、patch对齐与尺度规范。
- `extract_anchor_relation_geometry.py`：真实因果历史几何提取。
- `anchor_relation_decoder.py`：仅为实验读出增加可选mask，旧默认接口保留。
- `train_anchor_relation_geometry_probe.py`：固定有/无XYZ训练。
- `verify_anchor_relation_geometry_probe.py`：独立预测复算。
- `run_anchor_relation_geometry_local.py`：提取→训练→复算自动链。
- `test_anchor_relation_geometry.py`：新增7项测试。

未改生产 CEC、NavDP、论文或真机；保留工作区原有24个tracked修改，未commit/push。
训练样本全部来自原 train40，使用expert历史这一点不变。对 actual-online 分布、
完整 top-8 候选、无支持查询以及最终闭环能力，本轮尚无新结论。
