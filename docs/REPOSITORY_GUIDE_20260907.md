# 仓库地图与复现边界（2026-09-07）

## 1. 三个仓库，不混用工作副本

| 对象 | 当前目录 | GitHub |
|---|---|---|
| 仿真与研究主仓库 | `/home/asus/Research/Nav-graph-blind`；同仓库 main 工作树为 `/home/asus/Research/Nav` | `AlanZhu2006/Nav` |
| 活动论文 | `/home/asus/Research/Memnav_Paper`，入口 `main.tex` | `AlanZhu2006/Memnav_Paper` |
| 真机部署 | 独立 real-world 工作区 | `AlanZhu2006/Memnav_Realworld` |

本仓库的 `paper/` 是 ignored 的旧本地副本，不能据此判断 Overleaf 最新内容。
论文当前 Method/Problem 读取 `sec_lg/`，标题、摘要与 Introduction 按作者版本保护。
本轮 Git 整理不等于已经推送论文或更新机器人。

Git remote `fork` 指向 `git@github-alan:AlanZhu2006/Nav.git`；`origin` 是旧上游
`glbreeze/Nav`，不可把 `origin/main` 当作本项目发布状态。此次 SSH 验证返回
`Hi AlanZhu2006!`；本机 `gh` 默认登录另一个账户，不用于此次写操作。

## 2. 正式 CEC 的最短代码阅读路径

| 环节 | 文件 | 要理解的内容 |
|---|---|---|
| 总运行类 | [policy_agent.py](../NavDP/baselines/memnav/policy_agent.py) | 因果 RGB、DINO index、LingBot state、goal session、proof cache |
| 服务边界 | [memnav_server.py](../NavDP/baselines/memnav/memnav_server.py) | 请求与帧身份、depth/readout、控制授权 |
| 稀疏定位 | [certified_relocalization_runtime.py](../MemNavData/certified_relocalization_runtime.py) | 匹配、历史深度、PnP 与 certificate |
| 授权定义 | [certified_relocalization_contract.py](../MemNavData/certified_relocalization_contract.py) | 操作性证据判定，不是形式化安全证明 |
| 局部深度 | [monocular_depth_runtime.py](../MemNavData/monocular_depth_runtime.py) | first-40 相机高度尺度与当前帧 depth receipt |
| 方向接口 | [revisit_bearing_adapter.py](../MemNavData/revisit_bearing_adapter.py) | 单位 bearing → 固定 2.5 m residual；研究臂与默认分开 |
| 冻结策略 | [NavDP server](../NavDP/baselines/navdp/navdp_server.py)、[NavDP agent](../NavDP/baselines/navdp/policy_agent.py) | RGB/depth、ImageGoal/PointGoal、diffusion/critic、FIFO |
| 配对执行 | [eval_shared_online_role_pairs.py](../MemNavData/eval_shared_online_role_pairs.py)、[eval_2leg_habitat.py](../MemNavData/eval_2leg_habitat.py) | 原 A replay、隐藏 role、query rollout、计分 |
| 实际路径计量 | [executed_path_metrics.py](../MemNavData/executed_path_metrics.py) | 每动作位置积分、末位姿、精确 SPL/缺失末步区间 |

该运行类兼容许多已停用或探索性分支。复现实验时应读取该实验的固定配置和 source SHA，
不能仅运行当前 HEAD 的全部默认参数并声称复现了早期结果。

## 3. 状态与输入

```text
causal RGB archive + DINO keys     ← 可寻址、跨 goal 保留的 episodic history
LingBot streaming state           ← 连续几何；depth、pose、KV
NavDP FIFO / goal / diffusion RNG  ← 原控制器状态
```

DINO 不是 LingBot 的网络 head，RGB buffer 也不等同于 rosbag。三种状态通过帧索引/
图像身份对齐。目标切换保留长期历史，清除目标条件缓存；当前目标不能把后来的帧回填成
已有 Revisit 证据。运行时不消费 Novel/Revisit 标签。

主线接受时只给方向，不给真实目标距离或 GT pose。高度先验用于单目深度尺度；
环境 GT 只用于生成评测目标、标签/距离计算和显式 oracle 诊断，不属于部署输入。

## 4. 数据与权重：不随 Git 推送

| 数据来源 | 用途 | 注意 |
|---|---|---|
| PT1 / train40 expert 轨迹 | 早期训练、离线定位监督、可控诊断 | 不冒充当前策略实际观察到的历史 |
| actual-online A、A+B | 主表、full-mono、连续第三目标 | source trace、RGB、goal 和查询固定 |
| controlled causal RGB survey | 约 10–30 m 长度/路线诊断 | 不是 NavDP 自主完成第一段；需披露构造与分层混杂 |
| scratch 冻结结果 | 原始 CSV、plans、endpoint、completion 和 verifier | 文档是索引，原始记录与冻结源码是证据 |

权重、MP3D/HM3D 场景、视频、逐帧缓存、NPZ、source overlay 和大结果归档留在原存储。
`.diagnostics/` 包含本机诊断、稿件构建和本轮修改备份，保持 ignored；没有删除。
小型协议 JSON、SHA sidecar、提交收据和结果 Markdown 则应版本管理。

## 5. 环境与测试

- `memnav` 环境：模型/契约/统计测试；本次使用本机现有 Python 3.10 环境。
- `habitat` 环境：场景生成、renderer、`quaternion`；现有 Python 3.9 环境。
- 不把两个环境简单合并，不为获取干净日志隐藏环境错误或更改依赖版本。
- 本机 Habitat 无 pytest；unittest 测试用 `python -m unittest`，无 fixture 的纯函数
  测试逐个调用其断言。正式 GPU gate 使用 HPC 原容器和解释器预检。
- `bash -n` 只验证 Shell 语法；CPU 测试不代替真实模型启动与闭环评测。

## 6. HPC 默认操作

先读 [共享 SSH 与 Slurm 手册](../MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md)。
复用 `alantorch` 的已认证共享通道，核对身份 `yz11502`，不要因某个无 PTY 命令卡住
就断言 HPC 不可用。身份验证不需要在仓库内保存密钥、token 或临时登录码。

本轮 SPL 重跑：`h100_tandon,a100_tandon`、`gpu48`、1 GPU/10 CPU/72 GiB/
1 小时每 history、至多 4 并发。具体时间预算因实验而异，不直接复制旧 4–8 小时示例。
scratch 容量与文件数配额分别检查；此次 inode 接近上限，逐帧运行缓存写 node-local，
结束后归档和校验。大数据传输依手册使用 Globus/归档，不重复展开海量小文件。

## 7. 发布与清理原则

本次整理是入口、索引、计分补跑和已有研究文件归档，不是未经验证的模型架构迁移。
不移动冻结脚本，不删除实验失败记录，不清空用户 diagnostics，不重写远端历史。
先提交研究分支，再基于远端最新 main 非破坏合并，保留另一工作树已有主线提交。
