# 仓库与实现入口（2026-09-17）

## 一个工作区，三个独立仓库

| 内容 | 相对路径 | 远程 |
| --- | --- | --- |
| 导航研究与仿真 | `.` | `AlanZhu2006/Nav`、`glbreeze/Nav` |
| 当前论文 | `projects/paper` | `AlanZhu2006/Memnav_Paper` |
| 真机部署 | `projects/realworld` | `AlanZhu2006/MemNav-RealWorld` |

从根目录的 `GEM.code-workspace` 打开三个项目。导航远程 `fork` 和 `origin`
分别指向上述两个导航仓库，当前 `main` 同步到两端。进入各仓库分别提交；根目录忽略
整个 `projects/`。旧路径通过兼容链接保留，新开发使用上述入口。

论文以独立仓库 `main.tex` 的实际输入及最新 Overleaf 内容为准。根目录 `paper/`
是旧本地副本。布局见 [WORKSPACE.md](../WORKSPACE.md)。

## 最短代码阅读路径

| 环节 | 文件 |
| --- | --- |
| 记忆对象与状态归属 | [memory.py](../NavDP/baselines/memnav/gem/memory.py)、[bindings.py](../NavDP/baselines/memnav/gem/bindings.py) |
| 因果观测写入 | [writer.py](../NavDP/baselines/memnav/gem/writer.py)、[episodic.py](../NavDP/baselines/memnav/gem/episodic.py) |
| 历史几何档案 | [archive.py](../NavDP/baselines/memnav/gem/archive.py)、[support.py](../NavDP/baselines/memnav/gem/support.py) |
| 当前深度和目标读出 | [dense.py](../NavDP/baselines/memnav/gem/dense.py)、[sparse.py](../NavDP/baselines/memnav/gem/sparse.py) |
| 接受规则 | [certified_relocalization_runtime.py](../MemNavData/certified_relocalization_runtime.py) |
| Agent / HTTP | [policy_agent.py](../NavDP/baselines/memnav/policy_agent.py)、[memnav_server.py](../NavDP/baselines/memnav/memnav_server.py) |
| 深度尺度处理 | [monocular_depth_runtime.py](../MemNavData/monocular_depth_runtime.py) |
| NavDP 方向接口 | [revisit_bearing_adapter.py](../MemNavData/revisit_bearing_adapter.py) |
| ViNT / NoMaD 扩展 | [image_controller_goal_adapter.py](../MemNavData/image_controller_goal_adapter.py) |

RGB 档案、流式几何状态和控制器的短期队列有不同生命周期。目标查询读取历史；
切换目标保留观测历史、重建目标条件缓存。NavDP 保留原目标图像并接收方向提示；
ViNT/NoMaD 通过已接受的历史图像接入。细节集中在
[记忆模块说明](../NavDP/baselines/memnav/gem/README.md)。

## 配置与证据

兼容默认仍为 `legacy / dense / native`。原生 interval7 写入、支持像素档案和 KV
存储方案由配置显式选择。分页、INT8、坐标重连接等研究实现保留各自的实验状态。
真机默认配置和独立延迟回放见真机仓库。

正式实验按冻结输入、源码和配置复算。研究脚本保持原路径，以保留 manifest 和 source
bundle 的引用。当前证据从 [实验索引](EXPERIMENT_INDEX.md) 进入；旧指南保存在
[2026-09-07 快照](REPOSITORY_GUIDE_20260907.md)。

## 数据与环境

大型数组已转存 `/data`；Scene-N1 与可重建 Survey 缓存已清理。原始图像、轨迹、
统计与权重保留。复算前先看 [存储与恢复](LOCAL_STORAGE.md)。

- `memnav` 用于模型接口与 CPU 检查，`habitat` 用于场景和渲染。
- 部分 GEM 检查需要设置 `LINGBOT_REPO` 并将该目录加入 `PYTHONPATH`。
- 真机仓库按自己的 `AGENTS.md` 使用语法、配置和完整性检查，不运行单元测试。
- HPC 操作遵循 [共享 SSH 手册](../MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md)。
  旧日期报告中的队列状态仅是当时记录。
