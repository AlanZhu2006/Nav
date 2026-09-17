# GEM 固定 top-8 多参考定位：本机完整结果

日期：2026-09-07。状态：**定位对照完成，独立复算通过；没有导航 rollout，没有训练，没有提交 HPC。**

## 结论先行

本次朴素多参考联合 PnP **没有显示值得升级主方法的收益**。
同一 top-8 下，原几何第一帧定位达到平面误差 <=0.5 m 的为 14/41，
最优单帧离线 oracle 为 15/41，去重后联合定位为 12/41。
联合相对原单帧是 +0/-2 个 session；两个 loss 是同一目标/相同候选的重复输入，
不能当成两个独立案例。去掉完全相同的输入后为 13/38 -> 12/38，+0/-1。

这不是导航 SR，也不是“当前 CEC 只有 14/41 成功”。单帧数字评价的是
**原排序选出的位姿估计**，包括部署证书会拒绝的求解；联合没有获得控制权限。
主方法和论文表格保持不变。

## 1. 本次确实运行了什么

协议：`GEM_JOINT_REFERENCE_PROTOCOL_20260907.md`。
点顺序控制：`GEM_JOINT_REFERENCE_ORDER_CONTROL_20260907.md`，在任何真实 query 求解前追加。

- train40 内 8 个 scene/history，每个 history 纳入其全部合格 session。
- 41 sessions、28 个不同 query RGB hash、38 组不同的 history/query/top-8 输入。
- 选取规则只依据字典序、输入完整性、每 history 至少四个 session、前缀 <=384 帧。
- 共重放 2,024 张已有因果历史 RGB，补提取 253 个 anchor 的 LingBot 深度/置信度。
- 不重新生成图片，不增加 DINO 候选，不改变匹配器、PnP 参数或任何证书阈值。
- 历史为训练资产中的专家轨迹，**不是实际 NavDP Goal-A rollout**。
- 尺度使用原 first64 相机高度收据，**不是主方法的 first40 协议**。
- GT 仅在求解完成后的评分阶段读取，不用于前缀重放、对应选择、PnP 或多帧筛选。

三个条件：

| 条件 | 输入和职责 |
|---|---|
| 原几何第一帧 | 原几何排序，第一帧 2D–3D PnP；另外记录旧证书结果 |
| 最优单帧 oracle | 同一八帧单独求解后，评分端按位置误差选最佳；不可部署 |
| 多参考联合 | 同一八帧的对应提升到公共历史坐标系，query 点去重后一次 PnP |

所有条件使用同一 query 内参，即几何第一帧的预测内参。相比原运行路径，
第一帧本身的内参不变；其余单帧 oracle 固定相同 query 标定。
重复 query 关键点只保留最高 LightGlue score 的一个 3D 对应，不平均 3D 点，
不通过重复观测累计独立证据。

这仅检验一版明确的经典联合几何基线，不代表对所有多视图模型的否定。

## 2. 全部定位结果

主要误差：在共同的历史 anchor 坐标下，目标平面相对位置误差。
该 anchor 是原几何第一帧；尺度由历史相机高度收据提供。
这是 anchor-relative endpoint bearing，不是当前 decision 时刻 bearing，
更不是到目标的 geodesic 路线首段方向。

| top-8 原支持标签分层 | sessions | 原几何第一帧 <=0.5 m | 最优单帧 oracle <=0.5 m | 联合 <=0.5 m |
|---|---:|---:|---:|---:|
| <=0.10 | 21 | 0 | 0 | 0 |
| (0.10,0.50) | 6 | 3 | 4 | 1 |
| >=0.50 | 14 | 11 | 11 | 11 |
| 全部 | 41 | 14 | 15 | 12 |

分层只用作离线报告。top-8 最大共视不等于全历史支持，也不等于真实 Novel/Revisit 标签。
中间支持的 6 个 session 不构成连续共视曲线；其中也有重复 query。

| 指标 | 原几何第一帧 | 最优单帧 oracle | 联合 |
|---|---:|---:|---:|
| solver 有效输出 | 26/41 | 27/41 | 23/41 |
| 有效输出的中位位置误差 | 0.339 m | 0.258 m | 0.397 m |
| 有效输出的平均位置误差 | 2.413 m | 2.079 m | 1.437 m |
| 有效输出的位置误差 P90 | 7.698 m | 7.698 m | 4.605 m |

**不能拿联合的较低均值宣布改善**：三个条件有效输出集合不同。
原单帧和联合共同有效的 22 个 session 中，联合减单帧误差的中位差为 +0.0232 m。
联合没有把任何“所有单帧都超过 0.5 m”的样本救进 0.5 m。
联合误差低于单帧 oracle 的三条全部来自 <=0.1 支持组，仍未达到 0.5 m。

原单帧证书在 15/41 中接受；其中 13 条位置误差 <=0.5 m。
这是当前诊断人口中的定位检查，不是 certificate 的正式 precision，
因为这里的 0.5 m 位置标准与旧 stable-support 标签不是同一个定义。

## 3. 点顺序控制：不把 RANSAC 抽样差异当信息收益

原单帧保持原匹配顺序；联合输入在去重后按 query 像素坐标排序。
独立复算额外把各单帧对应也按同样顺序排列，其他参数全部不动。

| 条件 | <=0.5 m |
|---|---:|
| 排序控制后的几何第一帧 | 15/41 |
| 排序控制后的最佳单帧 oracle | 15/41 |
| 原联合结果 | 12/41 |

因此当前结果不能支持“多帧超过单帧的信息上限”。
排序控制不是新主方法，不用它的更好数字替换原单帧行，也没有挑选最佳 seed。

## 4. 一个具体退化案例：有更多内点，不等于目标更准

`ARNzJeq3xxb/episode_0000` 的 counterfactual Goal-B 在起点与中途两个 session
使用了同一个目标图、同一组候选和同一个共同评价 anchor=95，因此重复出现相同结果。

| 指标 | 原单帧 | 联合 |
|---|---:|---:|
| 平面位置误差 | 0.237 m | 0.632 m |
| anchor-relative 方向误差 | 0.944° | 4.372° |
| PnP 内点 | 67 / 280 对应 | 97 / 534 去重对应 |

该查询的 top-8 最大共视为 0.3103，真实 anchor–goal 平面距离约 3.674 m。
联合内点来自全部八个参考，但位置更差。联合重投影 RMSE 为 1.473 px；
小的内部重投影残差不能保证正确的物理目标位置。

对同一个 query 关键点在多帧提升出的 3D 点，测量它们到各自坐标中位数的距离：
中位为 0.456 m，P90 为 1.558 m，使用既有高度尺度换算。
这说明本案例中合并的对应并不充分自洽；不是“独立好证据越多越好”的理想情形。

但这个量是**跨帧离散程度，不是已知真值的几何误差**。
来源可能包括错匹配、深度预测、历史位姿或内参误差，本实验没有分离这些因素，
不能直接写成“已经证明 LingBot 漂移是主因”。
两种 bearing 的角度也都较小，因此不能从位置阈值退化外推实际导航会失败。

## 5. 数据与独立复算

评分初次退出：旧本地 RGB 包缺少 `7y3sRwLe3Va/episode_0000/meta/gen_meta.json`，
影响三个 query session；41 个推理均已成功完成，无需重新推理。

没有删除这三条、伪造 metadata 或以模型预测充当标签。评分专用修复：

- 使用旧完整 train40 Pi3X 评测的 `true_goal_center_reporting_only` 字段；
- 旧 summary 的 SHA 绑定了相同 3,840 行候选表和原 JSONL；
- 原生产函数 `_goal_center` 明确把 generator feet position 加上 `[0,0,0.5]`；
- 反向减去该固定偏移后，3,648 条有原始 metadata 的行全部复核一致；
- 只恢复缺少的目标位置，不读取旧模型预测、拟合变换或误差作为本次算法输入；
- 独立 verifier 38 条直接用原始 meta，3 条重读绑定的 GT 归档；全部保留。

`independent_verification.json`：

- `verified=true`；
- 41 个联合结果从保存的 2D–3D 对应重新求解，pose 最大差为 0；
- 原始 parquet/GT 重算位置误差一致；
- 重复 query 关键点只计一次；
- 原单帧、oracle、联合汇总全部复算一致。

原 score 入口要求原始 query metadata 齐全，因此本次使用独立的
`score_gem_joint_reference.py` 完成评分修复；冻结的推理脚本未修改。

## 6. 本机成本与测试

- 8 段几何提取累计 379.5 秒，不含模型加载；253 个 anchor 压缩缓存约 419.6 MiB。
- 两张与旧缓存重叠的深度 witness（17DRP，125/211）逐像素完全一致。
- 一次 top-8 匹配的中位计时约 0.103 秒，联合 PnP 中位约 0.0226 秒。
- 上述不是端到端导航延迟：不含原 DINO 寻址、历史写入、NavDP、网络和执行；
  匹配缓存也存在，且同卡有原真机服务驻留。不能将其写成部署实时性结果。
- 新增几何测试与原 PnP 测试：13 passed；语法检查及 `git diff --check` 通过。
- 未停止/修改原真机服务，实验 GPU 进程已退出。

## 7. 决策

保留当前 GEM/CEC，不把这版联合 PnP 上线，不进行它的 HPC 长闭环或长训。
本次不支持“多参考联合必然更好”，也没有发现值得靠放松证书补救的新能力。

若继续研究多帧读出，下一项应先隔离对应错误与跨帧几何不一致，
而不是加 K、累加内点、换阈值或者仅延长当前小模型训练。
这不否定预训练多视图定位模型，但其收益尚未在本项目得到验证。

## 8. 复现入口与文件

```bash
/home/asus/miniconda3/envs/memnav/bin/python -m pytest \
  MemNavData/test_lingbot_pnp_localization.py MemNavData/test_gem_joint_reference.py -q

# 新运行需使用一个尚不存在的 --out，不能覆盖本轮封存目录。
/home/asus/miniconda3/envs/memnav/bin/python -m MemNavData.run_gem_joint_reference prepare --out NEW_OUTPUT
/home/asus/miniconda3/envs/memnav/bin/python -m MemNavData.run_gem_joint_reference extract --out NEW_OUTPUT
/home/asus/miniconda3/envs/memnav/bin/python -m MemNavData.run_gem_joint_reference solve --out NEW_OUTPUT
/home/asus/miniconda3/envs/memnav/bin/python -m MemNavData.score_gem_joint_reference --out NEW_OUTPUT
/home/asus/miniconda3/envs/memnav/bin/python -m MemNavData.verify_gem_joint_reference --out NEW_OUTPUT
```

本轮输出：`.diagnostics/gem_joint_reference_20260907/`

- `manifest.json`：固定人口和历史输入。
- `geometry_receipt.json`：单目历史几何来源与耗时。
- `sessions/000.json` 至 `040.json` 与同名 NPZ：全部八候选/联合解及实际对应。
- `solve_receipt.json`：推理完成记录与冻结源码 SHA。
- `gt_label_recovery.json`：三个缺少 metadata 的评分修复及原标签核对。
- `scored_sessions.json`、`summary.json`：全部评分。
- `independent_verification.json`：复算与点顺序控制。

summary SHA256：`f6c98accc5397e3b597754c984de23c63924e82209edc7ca3eb5aa2c7799b63e`。

本轮没有修改论文、旧实验数字、生产 controller、任何证书阈值或冻结代码默认值；
没有 commit/push。
