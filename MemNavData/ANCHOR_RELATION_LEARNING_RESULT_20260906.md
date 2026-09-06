# 后续学习第一步：固定 anchor 的位置学习结果

日期：2026-09-06。结论：**最小训练已完成；视觉-only probe 可拟合，但内部跨场景精度
远未达到几何读出。保留 CEC，不扩大这一版训练，不称已完成 learned relocalizer。**

协议：[ANCHOR_RELATION_LEARNING_PROTOCOL_20260906.md](ANCHOR_RELATION_LEARNING_PROTOCOL_20260906.md)。

后续进展：[真实历史几何输入对照](ANCHOR_RELATION_GEOMETRY_PROGRESS_20260906.md)。
2026-09-06晚已完成原始依赖/GT复算和两段真实几何提取，本机正在提取其余历史；
几何条件训练尚待自动链完成。本文件的V0结果及未通过决定不变。

## 1. 本轮实际完成了什么

- 新建隔离的目标关系 decoder、三维坐标组合函数、训练入口和独立复算入口。
- 固定历史 anchor，不学习 DINO 排序、NULL、Novel/Revisit 分类、proof 或 NavDP。
- 复用本机冻结 DINO patch 和已审计 train40 位置标签，未重新运行导航或生成正式目标。
- 已有正支持 126 行，经 RGB query/anchor 内容去重为 123 个 pair；3 个重复没有重复训练。
- 固定 32/8 场景划分：103 train pairs、20 validation pairs。只使用原 train40。
- 先完成 8-pair 600-step 可拟合性 smoke，再完成固定 1,200-step 训练。
- 两次运行均由独立程序重读 predictions.csv 复算位置误差及 atan2 角误差，verified=true。
- 新模块 7 项测试通过；与旧 CDEC/Pi3X 相关测试合计 13 passed、1 个既有 PyTorch
  nested-tensor/norm-first 提示。没有掩盖 warning。

## 2. 这不是完整模型的替换试验

本轮模型输入只有目标图与 anchor 图的冻结 8×8 patch，以及 patch 的二维坐标。
**没有 LingBot 稠密深度/point-grid 输入，没有 proof，没有当前状态输入，没有闭环。**
其输出是旧标签定义下 anchor-base 的 [forward, lateral] 米制位置，尚不是完整
anchor-normalized 三维目标关系。因此本轮只回答“已有视觉特征换成位置监督后能做到
什么”，不能用来否决几何条件模型，也不能声称取代 SuperPoint/LightGlue/PnP。

代码的 `requires_geometry=True` 分支是已通过形状/梯度测试的待训练接口；它没有真实
几何训练结果。训练脚本只实例化 `requires_geometry=False`，没有填充虚假深度。

## 3. 定量结果

### 3.1 可拟合性与迁移差距

| 设置 | 数据 | mean position error | median position error |
|---|---|---:|---:|
| 8-pair smoke，训练样本本身 | 8 pairs | 0.0178 m | 0.0183 m |
| 固定训练最终模型，训练集 | 103 pairs / 32 scenes | 0.0893 m | 0.0862 m |
| 同一最终模型，内部验证集 | 20 pairs / 8 scenes | 1.3054 m | 1.0230 m |

模型参数 580,866。没有根据验证结果 early stopping、改变划分或挑选 checkpoint。
这些数值表明优化链路能够拟合训练数据；不能据此证明它学到了可迁移的几何算法。

### 3.2 同一内部验证集上的参考

| 读出 | 有效预测 | 平均位置误差 | 位置误差 ≤0.5 m，分母20 |
|---|---:|---:|---:|
| 目标直接取历史 anchor（零偏移） | 20/20 | 1.7923 m | 4/20 |
| 训练集平均位置偏移 | 20/20 | 1.3738 m | 4/20 |
| learned visual position | 20/20 | 1.3054 m | 3/20 |
| 旧 frozen GCT goal query | 20/20 | 0.4189 m | 17/20 |
| 旧 finite-PnP | 19/20 | 0.2898 m（仅有效19） | 16/20 |

finite-PnP 不是 strict CEC 接管率。在共同有效的 19 对上，learned 平均误差为
**1.1825 m**，finite-PnP 为 **0.2898 m**；没有借不同有效分母构造不公平比较。
旧 GCT/PnP 是同一已选 anchor 的已有资格测量，不是本次重新运行的同机闭环。

以位置误差 ≤0.5 m 作描述性计数：

- learned 对 anchor-copy：+0/−1；
- learned 对训练均值：+3/−4；
- learned 对 frozen GCT：+0/−14；
- learned 对 finite-PnP 的共同有效 19 对：+0/−13。

以上均是定位阈值计数，不是导航 paired SR，不套用正式导航的通过门。

### 3.3 方向与目标依赖

20 个验证 pair 中，19 个 GT anchor-goal 距离 ≥0.25 m，适合报告 anchor-local 方向：

- learned：13/19 方向误差 ≤30°；
- 训练均值：12/19；
- frozen GCT：17/19；
- finite-PnP：18/19（该方向总体中的有效情况见 JSON）；
- 将 learned 的目标图在验证 pair 间固定置换后：7/19。

置换结果说明输出确实依赖目标图，不足以证明已经学到准确几何；跨场景置换本身改变了
输入分布，不能当独立因果导航实验。anchor-local 方向也不是移动机器人每步的实际
current-to-goal bearing，尤其不能替代已有 4-query 连续状态闭环的指标。

## 4. 数据审计中新确认的限制

旧 `diag_lingbot_goal_loop_closure.py` 的 `depth_scale_raw` 为 candidate 与 goal 的
正深度合并后的 median。它是有目标参与的描述量，不是可直接复用的 history-only
归一化尺度。本轮完全没有把它、PnP/GCT 预测、误差、role 或 scene ID 输入网络。

这不表示该字段是 GT 泄漏；问题是它不符合计划中的“只缓存历史几何、目标由学生读取”
接口。V1 应使用真正 anchor-only 的尺度，不能为了现成字段而改变模型职责。

## 5. 当前决定与下一步

**不提交这版的 6–8 小时长训。** 已完成的定位监督，比重复训练 selector 更直接；
但 123 个已知支持 pair、单个划分、压缩到 8×8 的视觉特征不足以决定最终模型可行性。
不能将差距单独归因为数据量、DINO 表示或 decoder 容量中的任何一个。

下一项是唯一受控变量：给同一个关系读出加入真实历史几何。

1. 特征生产只读取实际因果历史 RGB，不把 query goal 送入历史深度生产过程。
2. 保存与 DINO pad/patch grid 对齐的历史三维位置和有效 mask；处理 padding、无效深度。
3. 保存 anchor-only 归一化尺度、camera-to-world 约定；GT 标签以同一局部坐标和
   一致的尺度规范表达。GT depth/pose 若用于生成监督，必须与模型输入分离。
4. 早期 anchor 若使用后续历史修正，只能使用 decision 前已观察帧，不能混用未来 prefix。
5. 先检查少量真实 pair 的投影与尺度，再在同一 32/8 划分、相同预算比较有/无几何。
6. 若仍不能接近已有 GCT/PnP，先检查表示与几何预训练，不靠加长本模型训练掩盖差距。

上述几何提取、几何条件训练、完整候选和 full-mono 闭环 **尚未运行**。
新的证据学习必须针对最终实际几何输出，不能继承旧 proof 的阈值。

## 6. 运行与文件

现有环境：`/home/asus/miniconda3/envs/memnav/bin/python`，PyTorch 2.8.0+cu128。

本次优化循环日志记录约 1.86 s / 5.43 s；这一计时在缓存加载后开始，**不包含历史
构建、DINO 提取、数据准备、环境启动或任何在线导航**，不是部署延迟。
PyTorch 峰值 allocated 约 104 / 124 MiB，也不是包含 CUDA context/冻结模型的系统显存。
由于本轮使用现成冻结特征，训练很轻；未来主要成本仍可能是几何特征生产，而非小 head。

输出根：`.diagnostics/anchor_relation_learning_20260906/`

- `overfit8_v0/`：8-pair smoke。
- `scene32_8_v0/`：固定场景划分结果。
- 各含 `manifest.json`、`probe.pt`、`report.json`、`predictions.csv`、
  `training_curve.csv`、`independent_verification.json`。

运行已经结束，没有遗留本轮 GPU 训练进程。共享 HPC 入口核验为 yz11502；仅做只读
路径检查，已退出本轮 PTY，权威 SSH master 保留；没有提交/取消 HPC GPU 任务。
其他工作区进程没有被停止。生产 CEC、论文、真机及原有用户修改均未改动；未 commit/push。
