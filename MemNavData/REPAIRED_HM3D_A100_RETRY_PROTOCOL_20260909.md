# 修复版 HM3D：仅补失败分片的 A100 重试

2026-09-09（北京时间），在本次重试开始前记录。

## 已知事实与范围

原 array `17196996` 的 shard 0 在 H100 gh003 正常完成，两条 A 均成功，
但没有完整 Natural Novel / Revisit 配对；0 query，independent verifier=true。
不重跑该分片，不修改其零查询结局。

shard 1 在 H100 gh008 的第一条 A 上收到 SIGABRT，运行 19 分 58 秒，
仅留下 56 个完整动作；第二条 A 尚未执行。两者均没有终局收据，不能计作导航失败。
原输出 23 个文件、3,324,063 bytes 全部留在原目录，不覆盖或删除。

崩溃前 memory HTTP 完成日志间隔约 19 秒；这不等于 LingBot forward 时间。
相似旧事故已有 A100 原配置补跑记录：`17089916` 九项及 `17089989_16`
均 COMPLETED，后续 `17089991` 也正常完成。它支持运行规避选择，
不证明 H100 普遍有错，也不证明本次根因已解决。

## 唯一重试范围

只补原 shard 1：父 manifest rank 4/5，`BFRyYbPCCPE` 与 `X7gTkoDHViv`，
各 `episode_0000`，seeds `2026082600` / `2026082700`。
重新从各源起点实际执行 mono-A，不从未完成的 56 步继续拼接。
仍先完成两条 A 与固定构造，再运行全部合法历史的六个 role-arm。

完全复用原 sealed bundle：

```
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f
SOURCE_BUNDLE.sha256 = c8cf8c60e7efd55f1b360bb5a2d5fd92850dc87d6a310b05e77f0a9239cdfc08
```

不改 source、模型、权重、RGB/depth、导航接口、CEC 阈值、2.5 m、seed、
first40、query 构造、预算或 CUDA 同步开关。资源限定为原允许池内的
`a100_tandon`，array=1，一项 1 GPU / 10 CPU / 96 GB / 1 小时。
新增 `PYTHONFAULTHANDLER=1` 仅用于 native abort 时输出 Python 堆栈。
保留原节点内临时 buffer，不把连续图像缓存复制回接近文件数配额的 scratch。

使用独立输出，不替换初始失败目录：

```
/scratch/yz11502/Research/Nav-axis-uturn-results/repaired_hm3d_extension_20260908/a100_retry1_20260909/shard_1/integration
```

提交前验证原源码 SHA、Slurm 模板语法与 test-only；复用已通过的实际解释器预检。
GPU 节点运行时仍会再次检查三组导入、所有 CLI、源数据、依赖和两场景渲染。
不自动增加场景或触发正式论文作业。若再次出错，保留现场后用新堆栈定位；
若仍无 query，保留构造损耗，不放宽规则换取非空分母。

## 构造问题独立处理

对已有成功 A 做本机只读审计，区分：历史帧/距离不满足、方向/楼层/净空拒绝、
最终共视拒绝。固定方向或历史 warmup 约束可能降低可构造率，但必须由实际轨迹验证。
不得把这些候选筛选称为 CEC 证书拒绝，也不修改已冻结构造来冒充原协议结果。
