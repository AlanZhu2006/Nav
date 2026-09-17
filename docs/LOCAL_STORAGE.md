# 本地存储与恢复（2026-09-17）

本机工作区是 `/home/asus/Research/Nav-graph-blind`，整理后约 115 GiB。
源码、论文、权重、原始 Survey RGB、轨迹和统计保留。下列本地归档不随 Git 下载。

| 内容 | 本地处理 | 恢复方式 |
| --- | --- | --- |
| Scene-N1 模型、纹理与下载包 | 清理约 155 GiB | 固定源版本下载 |
| 13 份 Survey 状态 | 清理约 121 GiB | 从原始 RGB 重新 Prepare |
| 旧真机重复解压目录 | 清理约 27 GiB | 原压缩包留在真机 runtime/experiment_archives |
| 历史实验数组 | 约 128 GiB 转存，压缩后约 125 GiB | 从 /data 按需恢复 |

## 实验数组

归档：`/data/workspace-archives/gem-storage-prune-20260917/experimental-arrays.tar.zst`。
SHA-256：`493ce60f4e7ab75ad7ea73fbfa373bff9ed309915872f28bd831d8ddbf1f7fa9`。

302,267 个原文件逐项比对通过后移出工作区。恢复单个实验目录，例如：

```bash
tar -I zstd -xf /data/workspace-archives/gem-storage-prune-20260917/experimental-arrays.tar.zst -C /home/asus/Research/Nav-graph-blind --wildcards '.diagnostics/gem_connected_memory_20260913/resources_001/*'
```

同目录 `cleanup-records.tar.zst` 保存清单、校验与 Survey 输入备份。本机详细记录也在
`.workspace-maintenance/20260917-storage-prune/`。HPC 核对覆盖 88 份一致的关键文件，
并非所有本地原始产物；独有数组依赖上述归档恢复。

## 场景和状态

Scene-N1 来自 Hugging Face 数据集 `InternRobotics/Scene-N1`，固定 revision：
`2195d46aaab0ff48673b275fdfdc0731075b5ff2`。需有数据集访问权限的账号。源文件、
SHA-256 和展开布局见上述审计目录的 `scene_download_sources.json` 与 `RECOVERY.md`。
场景恢复前，相应本地仿真不能直接运行。

Survey 缓存缺失时，部署代码从封存 RGB 重放并保存新状态。首次 Prepare 会增加耗时
和缓存占用。审计目录的 `survey-cache-inputs.tar.zst` 只有输入和元数据，应解压到
临时检查目录，不能当作完整快照直接放回运行缓存。
