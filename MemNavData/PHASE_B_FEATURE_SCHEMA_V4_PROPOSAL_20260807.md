# Phase-B 特征 schema v4 提案：收进已计算但被丢弃的 LingBot 几何信号

日期：2026-08-07

状态：提案。**不改动正在运行的 collector（job 15474001）**；v4 只改
`phase_b_feature_schema.py` + join/trainer 读取列。生效前置条件见 §4。

## 1. 动机

全仓审计（2026-08-07）确认：`diag_lingbot_goal_loop_closure.py` 逐候选计算
了方向性 cloud overlap、overlap 分布统计和多 anchor pose-consensus
dispersion，但 v3 `FEATURE_NAMES`（19-D）只收了对称的
`cloud_overlap_f1_center` 一个 overlap 标量，dispersion 完全未收。

已有离线证据支持这些信号有判别力：

- 93-row development：cloud-overlap ROC-AUC `0.743` > DINO `0.610`，
  融合 `0.776`（`NOVEL_MEMORY_RESIDUAL_V2_20260807.md` §4.2）；
- 离线 AUC 表中 `lingbot_pose_consistency` 作为独立条目被评估
  （`diag_lingbot_goal_loop_closure.py` 约 line 1255），却不在部署特征里；
- DINO-only 学习模型 candidate recall@1 差 max-DINO 约 15 pp，缺口被诊断
  为"输入特征不足"（`LATEST_TRAINING_RESULTS_20260807.md` §16.2）。

## 2. v4 新增列（全部来自 collector 既有输出，零重采集）

2026-08-08 按 collector 实际落盘核实修订：v3 实际为 **20 维**（此前审计报告
中的 19 为算术笔误）；`dino_cosine_median` 只是 summary 聚合、不是逐行列，
从提案中移除；dispersion 在邻域 <2 时为 NaN，新增显式 presence flag。

| # | 列名 | 来源 | 为什么有判别力 |
|---|---|---|---|
| 21 | `cloud_overlap_candidate_to_goal_center` | `hypotheses_json` 的 offset-0 条目（非平铺列，join 时提取） | 方向性视角关系："我看得到它"≠"它看得到我"；对称 F1 抹掉了这一信息 |
| 22 | `cloud_overlap_goal_to_candidate_center` | 同上 | 两方向不对称本身是视角差的指纹 |
| 23 | `cloud_overlap_f1_median` | 平铺列，已存在 | 对孤立好分的稳健性（真匹配是一片，alias 是一根） |
| 24 | `goal_pose_translation_dispersion_norm` | 平铺列，已存在 | 最接近回环一致性检查的量；alias 的多 anchor 定位互相矛盾 |
| 25 | `goal_pose_rotation_dispersion_deg` | 平铺列，已存在 | 同上，旋转通道 |
| 26 | `goal_pose_dispersion_present` | 派生 flag：`n_hypotheses >= 2` | dispersion NaN 时置 0 并把两列归零（mask=0 ⇒ 字段归零，同 V2 candidate schema 约定；不伪造共识） |

不新增派生分数列（如 overlap 非对称差）：两方向原始值已含该信息，交给模型。
使用 `_norm`（scale 归一）而非 `_raw` 的 dispersion，避免把逐场景 metric
scale 漂移（17DRP ~0.90 vs 1LX ~0.79）泄进特征。

**2026-08-08 对真实 train artifact（job 15474001，`train_top2_gap16`，
1098 rows × 67 cols）的核查修订**：该采集配置为每候选单 center 假设
（`n_hypotheses == 1` 于全部行），因此本 artifact 上：

- #21/#22 方向性 overlap **可用**（`hypotheses_json` offset-0 提取，已核实
  两字段存在且非退化）——这是 v4 在当前数据上的全部即时增量；
- #23 `cloud_overlap_f1_median` 恒等于 `f1_center`、#24/#25 dispersion 全为
  NaN、#26 flag 恒 0——四列在本 artifact 上是**常数/退化列**，会被
  finite/non-constant 审计正确拒绝，不得入列。

因此拆成两步：**v4a（立即可用，20→22 维）**只加两列方向性 overlap；
dispersion 三件套推迟到未来一次带邻域假设（neighbor offsets ≥2）的采集，
届时才是"零边际成本"——在那之前它不再是免费项，须按 GPU 预算单独排期。
`phase_b_feature_schema.py` 的 `FEATURE_NAMES_V4` 在实施时按 v4a 缩减。

## 3. ABI 变更（已在 `phase_b_feature_schema.py` 落地为双版本常量）

```text
CHECKPOINT_SCHEMA_VERSION: 3 -> 4（ACTIVE_FEATURE_SCHEMA_NUMBER 翻转时生效）
FEATURE_SCHEMA_VERSION:
  lingbot_native_phase_b_features_v4_directional_overlap_pose_consensus
FEATURE_DIMENSION: 20 -> 26（v4 为 v3 的纯后缀扩展，前 20 维逐位一致）
FEATURE_NAMES_SHA256: feature_schema(4) 自动计算
```

已验证：模块默认 ACTIVE=3 时与 HEAD 提交的 v3 ABI **逐位一致**（20 维、
digest 相同）；`feature_schema(3)/feature_schema(4)` 显式访问器供 v3-vs-v4
对照训练使用，两条 lane 都不得依赖模块级默认。

`validate_checkpoint_metadata` 的 fail-closed 语义保证旧 16/17/19-D
checkpoint 不能与 v4 部署路径混用。

## 4. 生效前置条件（按序）

1. `15474001` + `15474003` 完成，artifact 通过既有审计；
2. **确认 join artifact 的逐候选行保留了 §2 六列的原始值**（collector 在
   算，但需核实 join 是否透传）。若透传：v4 = schema bump + 重跑 join 列
   选择 + 重训，零 GPU 采集；若未透传：只需重跑 CPU join，仍零 GPU 采集；
3. 六列过既有 finite/non-constant 特征审计；
4. 校准注意：overlap/confidence 是模型分数不是概率
   （`DEPTH_CONFIDENCE_SEMANTICS` 同款警告适用），进 GLP 似然前必须
   isotonic/温度校准；
5. v4 模型的 Go/No-Go 与 GLP Stage 2 合并：同协议 development candidate
   top-1 必须超过 max-DINO `82.35%`，否则不进闭环。

## 5. 明确不做

- 不中断/修改正在运行的 collector；
- 不加尚无采集来源的列（yaw 协方差 head 不存在，保持 honest-absent）；
- 不把 `_raw` dispersion（未归一）入列；
- 不在 v4 里同时引入 dense patch 特征——那是独立的 B 项
  （patch 似然），等 v4 结果出来再定。
