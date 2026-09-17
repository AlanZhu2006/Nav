# 修复版 HM3D 两源运行栈验证

2026-09-08。HPC 提交前固定；这是可迁移运行栈 gate，不是论文正式确认。

- 使用原 HM3D parent SHA `a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5`。
- 固定前两 scene `rJhMRvNn4DS`、`6D36GQHuP8H` 的 `episode_0000`，
  seed 分别为原 collection 规则的 2026082200、2026082300。没有查看旧 A/queries 结局来选样。
- 两条 A 都重新执行 mono-native；两条全部结束后，才从新 A 构造查询，再跑全部六个 role-arm。
  每 source 最多一个 standard-Revisit/Natural-Novel 历史；A 失败或无合法查询时保留损耗，绝不补历史。
- 执行参数与本机通过版相同：RGB 正确、单目深度逆 padding、bounded pursuit、标准 `try_step`、
  raw/CEC 共用真实后向适配；canonical reference depth；600 tick、8-tick 规划、1 m GT 成功。
  没有 metric sensor depth 输入，也不从 evaluator 向模型传 GT odometry。
- 此 gate 保留本机已测的旧构造 K 定义，明确 FY 与真实 renderer 约有 1.16% 差异。
  本次专门验证迁移，不同时变更几何构造；正式新版构造是否统一真实 K 必须在采集/选样前另行冻结。
- Novel/Revisit role 仅供 evaluator 取目标图与报告。三个方法共享新 A、目标、起点、GPU/模型进程。
- 保留全部动作前后位姿、完整 NavDP 候选与 critic、深度数组、末步终点、独立 SR/SPL 复算及视频。
  失败 A/查询也保存。单纯 CPU tests、服务 ready 或合法零历史，不算完整 query gate 已通过。

## 资源和数据

一个 GPU 作业，h100_tandon/a100_tandon，1 GPU、10 CPU、96 GB RAM、1 小时上限。
本机两源实际完成 8 rollout 约 13.7 分钟（不含全部外围开销），HPC 两源最多 14 rollout，
首次 1 小时是受限试运行预算，是否足够必须由这次实测决定，不是性能承诺。

仅上传源码包；场景与大权重复用已校验路径，不重新下载 PT1、Globus 或数据集。
用户 scratch 文件数约 97%：逐帧模型 buffer 和工作目录放节点临时目录；科学输出持久保存。
代码只保存一次 immutable bundle 引用，不在每个 source/arm 复制整个源码树。

依赖验证使用实际 Habitat / MemNav 绝对解释器及相同 Singularity mounts，启动前检查
RGB/depth/pointgoal 关键模块来源、7 组 CLI、权重、源文件和 renderer；端口用现有 node-local flock。
不得覆盖 `LD_LIBRARY_PATH` 中 Singularity 注入的 GPU 库。仅清理本作业拥有的服务。

HPC Habitat 解释器未安装 OpenCV。仅复用现有 MemNav 环境的 OpenCV-headless 4.9.0
`cv2`/二进制库到本任务的只读依赖目录，已验证其 abi3 扩展在 Habitat Python 3.9 /
NumPy 1.26.4 下导入和 resize；不把整个 Python 3.10 site-packages 加入 Habitat。
不安装或升级共享环境。临时目录在首次依赖导入前绑定，避开登录节点已满的 `/tmp`。

## 判定与下一步

先看输入/执行/收据是否一致以及实际耗时，再决定是否提交正式批次。
任何 query SR 不用于本轮改阈值、选更多 source 或换方法。
空构造可以是合法科学输出，但须明确尚未测试到六个 query arms，不能报告 gate 全通过。
即使通过，两个 consumed source 的结果也不能替换论文主表。

标准 NavMesh 碰撞、理想底层状态与 GT 到达仍存在；不是刚体真机安全或自主视觉 STOP 验证。
