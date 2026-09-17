# Table II 前向 Novel 单变量审计

日期：2026-09-11。用户要求：统一前向 Novel 构造，不引入其他变量。

## 1. 变更边界

对照基准是已经封存、运行过的 **v2 修复版**，不是旧论文中的旧执行器。
本轮唯一任务条件变化：A、B-Novel、C-Novel 只接受起始最短路线首段相对实际 yaw
在 **±60°** 内的目标，沿用已有 `front` 定义。

选择目标，不旋转机器人。B/C 继承真实前段末态的位置、朝向和历史。
这不限制途中转弯，不强制目标照片朝向与接近路线对齐，也不把 GT 路线方向传给 NavDP。

| 项目 | 本轮处理 |
|---|---|
| Novel 方向 | 原九格中的前向子集；仍按原匹配目标平衡三个距离格 |
| Revisit 方向 | 前、侧、后均保留，原九格匹配不变 |
| 来源 | 原 18 scenes / 144 个声明源，源顺序、carrier、起点、独立 yaw、policy seed 不变 |
| 目标距离 | 原 `[2,4) / [4,6) / [6,9]` m，不缩短距离来提高 SR |
| 目标照片 | 原八档世界朝向及随机数命名空间，不改成面向接近路线 |
| Novel 支持 | 当前图共视 <0.10、完整历史 max-covis <0.10；A 不豁免当前图检查 |
| Revisit 支持 | 原 [0.55,0.90]、合法历史支持 ≥0.55、位置/朝向扰动、像素差条件均不变 |
| 构造预算 | 原 10,000 次空间提案、每格最多 12 个 Novel 视觉候选、384 次 Revisit 视觉检查 |
| 感知输入 | 原 RGB、相机内参/高度、LingBot 单目深度、first40 高度尺度；不加 sensor/GT metric depth |
| 策略和记忆 | 原 frozen NavDP/GEM、检索、证书参数与 2.5 m residual，不训练或调阈值 |
| 执行和评分 | 原 `bounded_standard/rgb_v1/source_rgb/heading_on`、600 ticks、horizon=8、1 m 到达标准 |
| 历史和配对 | actual-online；B 分支隔离；C 仅成功 native-B、两种 B 角色来源 50/50；同查询 native/GEM 同任务配对 |
| HPC 环境 | 原 A100 / 1 GPU / 10 CPU / 96 GB / 1 h、容器和依赖；继承已记录的 A 并发 4、B/C 并发 2 |

schema、文件路径和协议摘要更新是新实验身份，不得据此改变随机数种子。
特别保留 `table2_balanced_sampling.SCHEMA` 的旧随机数命名空间：同一起点的同一 proposal
仍对应同一空间位置和目标照片朝向。

## 2. 实际审计结果

### 来源与代码

- 从原 inventory / capacity 审计重新产生全部 144-source 声明，与原声明逐字段比较。
  除协议 schema、协议 SHA、初始状态中的 schema 外，**所有字段一致**。
- 直接载入封存 v2 的选择函数，与当前默认九格选择逐项比较：结果完全一致。
- 导航策略、GEM、深度、控制器、构造共用函数和环境文件逐文件核对，没有夹带方法改动。
- 新包不再从脏工作树全量抓取运行时；从校验过的旧 bundle 复制，只覆盖 9 个明确的
  Table-II 协议/方向配置/编排/检查文件。其余 **1,772 个文件逐字节不变**。
- 排除了本轮无关的 `build_hm3d_fresh_birdeye_cases.py` 和
  `verify_repaired_fullmono_local.py` 工作区差异；没有回滚用户文件。
- v3 草稿曾把 B/C 并发也扩到 4，现已收回；保留 v2 已有的 A=4、B/C=2。
- 检查报告中将“源是否保留”与“是否能构造目标”分开，避免把空菜单误写成丢弃来源。

### 真实场景差分检查

没有调用 NavDP、GEM 或 LingBot 推理，没有新增导航成绩。使用已消费的真实历史作为
**构造测试输入**，这些历史不会进入 v3 正式导航总体。

| 检查输入 | 新 Novel 合规格数 | 原样保留的 Revisit 格数 | 核验 |
|---|---:|---:|---|
| 原声明 A source 000，空历史 | 3 | 0 | 与旧版前向候选完全一致 |
| 一条 actual-A 末态 | 0 | 1 | 不用原后向 Novel 补空；Revisit 不变 |
| actual A + native Novel-B | 0 | 3 | 无未见前向目标则留空；Revisit 不变 |
| actual A + native Revisit-B | 2 | 2 | 两类候选与对应旧候选一致 |

差分不是只比较标签：逐项核对 proposal index、位置、照片 yaw、路线距离、初始角度、
完整共视曲线、目标 JPEG SHA、历史 SHA、真实末态位置/yaw 和 runtime 字段。
Novel 的空间提案及视觉检查序列恰好是旧版的前向子序列；三个有历史输入的
Revisit 提案诊断、候选和图像均完全相同。

Habitat 输出既有语义标注加载警告；本检查不使用语义观测。实际 RGB、深度构造与
保存图像均完成并通过旧/新比较，没有为消除该警告更换资产、相机或依赖。

### 测试

本轮 **278 passed**，覆盖前向角度边界、拒绝伪装为 front 的后向角度、Revisit 后向保留、
三格/九格选择、空菜单、C 来源 50/50、真实前缀隔离、CLI 参数、GT/策略边界、
bounded pursuit、后向处理、归档/SPL 复算及冻结 runtime 打包。

这表示实现与控制变量检查通过，不表示新版本已取得导航 SR。

## 3. 构造损耗与解释边界

旧版在导航前已封存的全部 A 菜单中，有 **97/144** 个来源提供合规前向目标，
其余 **47/144** 在原构造预算内没有前向目标。三距离格的选择预览为 **33 / 32 / 32**。
这是前向条件下的可构造性，不是 97 次导航成功或 47 次导航失败。

全部 144 个来源继续登记；不增加提案次数、不转正机器人、不换更容易的场景、
不放宽共视、不根据旧 A 成败选择来源。正式导航结果分开报告声明源数、
构造损耗、实际执行分母、SR/SPL；B/C 同样遵守原条件分母。

**20:42 CST 全量验收完成：144/144 个 A 来源已重新构造并逐项核验。**
新菜单严格等于原菜单的前向子集；所有保留目标的图像 SHA、位置、照片 yaw、
共视曲线及其余查询字段一致，仅 schema 和保存路径不同。
实际合规来源仍为 **97**，构造损耗 **47**，三个距离格为 **33 / 32 / 32**，
与预览一致。验收收据：`full_A_differential_verification.json`，`verified=true`。

“不引入其他变量”指不改变其他实验规则，不是强迫后续实际状态保持原样。
A 的目标变化后，新 A 轨迹、终态和可见历史可能变化，B/C 查询也随之变化。
不能复用旧 A 轨迹冒充新 actual-A，也不能把旧/新原始 SR 差当作固定目标下的纯转向因果效应。
主比较仍是同一查询、同一真实前缀上的 native/GEM 配对。

## 4. 证据与复现入口

本轮目录：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/table2_forward_only_audit_20260911_v1
```

- `source_declaration.json`：144-source v3 声明。
- `static_verification.json`：来源字段、旧选择器、运行时差异与方向供给预览。
- `render/render_verification.json`：四种真实构造输入的差分结果。
- `full_A_construction.log`、`A_menus/`：全量重新渲染过程及候选。
- `full_A_differential_verification.json`：全部 144 个新菜单与旧前向子集的精确比较。
- `bundle_verification.json`：继承运行时的逐文件验证。
- `bundle/FORWARD_OVERLAY.json`：明确覆盖清单和旧/新 SHA。

旧 bundle receipt SHA：
`7a66e6a3e056afed96f6d724a0936c2551ff45e46b78ec9d9a7796639f0363b6`。

新 bundle receipt SHA：
`c801f6a52fc128451afffa5a5066d713f362f354bf5acc11c3225ec60a0f08ca`。

新 source declaration SHA：
`2259b00e01aaf43d3e93a185b368489c74448e83eb5d0de8be06b1731bc151ff`。
封存包已从自身目录独立完成 v3 构造/编排模块 import，不借用工作区模块。

审计入口：`MemNavData/audit_table2_forward_novel.py`；测试入口：
`MemNavData/test_table2_forward_novel.py`。静态检查使用原 MemNav 环境，
真实渲染使用原 Habitat 环境，不安装新依赖。

截至本报告写入：新 v3 未提交 HPC、没有新 SR；没有改论文、Git commit 或 push。
旧 v2 归档与论文结果保留；其待运行 B/C 续接已在用户改为前向 Novel 后停止。
