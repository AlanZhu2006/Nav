# GEM 工作区

统一从 [GEM.code-workspace](GEM.code-workspace) 打开项目。

| 内容 | 相对路径 | Git 仓库 |
| --- | --- | --- |
| 导航与仿真 | `.` | `AlanZhu2006/Nav`，同时同步到 `glbreeze/Nav` |
| 投稿论文 | `projects/paper` | `AlanZhu2006/Memnav_Paper` |
| 真机部署 | `projects/realworld` | `AlanZhu2006/MemNav-RealWorld` |

三个仓库各自提交。`projects/` 被根仓库忽略。当前论文入口是
`projects/paper/main.tex`，根目录中旧 `paper/` 不参与主稿构建。

新机器可在导航 checkout 内补齐两个项目：

```bash
mkdir -p projects
git clone git@github.com:AlanZhu2006/Memnav_Paper.git projects/paper
git clone git@github.com:AlanZhu2006/MemNav-RealWorld.git projects/realworld
```

## 本地资料

- `.diagnostics/`：统计、轨迹、图像和复核证据；部分大型数组已归档。
- `.workspace-assets/`：权重、原始 Survey 和仍被脚本引用的依赖快照。
- `.workspace-maintenance/`：整理清单、修改前备份、校验和恢复记录。

这些目录不进入 Git。用途和恢复条件见 [存储说明](docs/LOCAL_STORAGE.md)。

本机主目录为 `/home/asus/Research/Nav-graph-blind`。旧 `Memnav_Paper`、
`MemNav-RealWorld`、`Nav` 和 `Nav-axis-uturn` 路径保留为兼容链接，不是额外开发
工作区。部署配置仍可引用它们；不要在旧依赖快照中继续开发。
