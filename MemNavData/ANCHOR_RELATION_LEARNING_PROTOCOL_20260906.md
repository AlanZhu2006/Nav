# 目标—历史关系学习：最小实验与数据边界

日期：2026-09-06。状态：实施中；不替换 CEC，不启动长训，不改论文或真机。

## 1. 研究目标及与旧实验的区别

保留 DINO 地址检索、LingBot 连续当前状态、单目深度支路、冻结 NavDP 与 2.5 m
方向接口。学习目标是目标相对历史 anchor 的局部位置，而非候选排序、Novel/Revisit
身份、单独的证书分数、当前状态重建或动作。

8 月 13 日 M2P 协议已讨论过关系读出，不能宣称首次提出。本次可区分的新实测变量是：
固定支持 anchor，直接监督局部位置；当前状态不由 query 网络更新。已有 Pi3X 连续状态
试验支持职责分离，但三臂 4/4、主要绕行改善集中同一场景，不是新正式 SR。

## 2. 最终模块与本轮最小对照不是一回事

最终候选：冻结图像 patch + 历史 LingBot 局部三维几何 → 小型关系 decoder →
目标在历史中的位置；再与连续当前状态解析组合。授权后仍使用原方向接口。

当前本机具有：3,203 张图像的冻结 8×8 DINO patch；480-session 旧完整资格表；
geometry 选中且 GT 支持为正的 126 行。真实 RGB 内容与原行顺序有哈希绑定。
但本机这份表没有逐 anchor 稠密 LingBot depth/point-grid，旧 NPZ 是 patch，不是几何。

因此先做 **V0：无几何输入的固定 anchor 位置 probe**。这是缺少几何的输入对照，
不能代替最终模块；成功不授权闭环，失败不否决几何条件模型。不能以零/常数点图
冒充真实几何，也不为利用现成数据改变既定最终输入。

代码同时实现几何条件输入接口及明确的缺失检查；该接口尚未用真实几何训练。
V0 输出 anchor-base 的 [forward, lateral] 米制位置，仅对应现有表的平面监督，
不是已经完成三维目标 pose 学习或尺度无关部署。完整版本需在 anchor 局部坐标中
一致归一化输入几何和目标标签，再单独验证三维到机体平面的变换。

特别注意：旧 `depth_scale_raw` 是 candidate depth **与 goal depth 合并后的 median**，
不是纯历史尺度。V0 禁止读取它作为模型输入或标签归一化，不能将其当作缓存的历史尺度。

## 3. 运行前固定的数据与选择

- 仅 `router_multiscene_split_20260805.json` 的 train40。
- 使用完整旧 geometry 表中 `label=1` 的已选 anchor，不按 CEC 是否通过二次过滤。
- 这是已知支持 anchor 的条件定位诊断，不是可部署检索性能。
- 按 query/anchor RGB 内容 SHA 去重；重复 session 不重复计入样本。
- 32 train / 8 validation scenes，使用固定盐 `anchor-relation-position-probe-v0-20260906`
  对场景名哈希排序，前 8 为内部验证。先固定划分，不根据位姿误差选择。
- development、Final14、正式 HM3D 不参与训练或选择；内部 validation 不是新的外部确认。
- 验证图像内容、source row、candidate frame < decision frame 和嵌套 target 一致。
- 输入只含两个冻结 patch 张量和二维 patch 坐标；标签、pose、scene、role、DINO score、
  LightGlue/PnP 与 teacher 的错误统计均不能进入 forward。

## 4. V0 训练预算与比较

共享 1024→128 patch 投影；2 层 cross-attention，4 heads，dropout 0.05；
平面位置输出，不训练 proof head。AdamW lr 3e-4，weight decay 0.01，batch 16，
seed 11；固定 1,200 steps，不用验证集 early stopping 或选择最优 checkpoint。

损失：米制位置 Smooth-L1(beta=0.2) + 0.1 × (1−cos 方向误差)。
方向项只在 GT anchor-goal 平面距离 ≥0.25 m 时计算，避免近零基线的角度奇异性。
先独立运行训练子集前 8 个去重 pair 的 600-step 可拟合性 smoke，再运行上述固定训练。

对照：

1. anchor-copy（目标位置等于 anchor，零局部偏移）；
2. 仅训练集估计的平均目标偏移；
3. 同一个已选 anchor 的旧 frozen GCT goal query；
4. 同一个已选 anchor 的旧 finite-PnP（报告有效覆盖，不把缺失预测隐去）；
5. learned visual position；
6. 最终网络的 goal 置换，仅作图像依赖诊断，不作独立导航证据。

旧 GCT/PnP 是同数据参考输出，不是新同机闭环臂。报告 pair 和 scene-macro 位置误差、
近零基线外的 anchor-local 方向误差及有效覆盖。方向误差不是实际当前导航 bearing
误差；定位命中率不是 SR。一个训练/验证划分也不等于完整 scene-OOF。

## 5. 下一阶段与去留

V0 之后的直接任务是小规模提取真实 **causal history-only** LingBot depth/point-grid，
记录独立的 anchor-only 深度归一化尺度；核验 GT 到同一局部坐标/尺度的转换。
有几何与无几何采用同一固定 anchor、划分、预算比较，而非根据 V0 结果改验证集。

之后才引入完整 top-8 候选、针对实际新几何输出的证据学习，以及实际 full-mono
共享 A 历史下 CEC / 修复读出的冻结 Pi3X / 新 learned 三臂闭环。

小样本定位或训练 loss 不授权 6–8 小时长训；SR 不劣需要预定差值及配对区间。
若学习仅记住训练样本而跨场景无收益，先检查几何条件与样本支持，不扩大旧排序网络。

## 6. 范围与入口

- 模型与坐标组合：`MemNavData/anchor_relation_decoder.py`。
- 最小训练：`python -m MemNavData.train_anchor_relation_probe --out-dir NEW_DIRECTORY`。
- 测试：`MemNavData/test_anchor_relation_decoder.py`。
- 使用现有本机 memnav 环境，输出隔离在 `.diagnostics/anchor_relation_learning_20260906/`。
- 本轮不提交 HPC GPU 作业；按共享 SSH 手册做只读资源查询不等于提交实验。
- 不动现有未提交修改，不停止其他工作区 GPU 进程，不 commit / push。
