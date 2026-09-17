# Raw fixed 的独立匹配验证：本机组件实验

2026-09-10。用户同意继续后启动；沿用此前 MASt3R 设计，不修改 GEM 默认方法。

## 唯一问题

固定 raw 实际选中的历史帧和 LingBot 目标位姿，新的双视图匹配证据能否区分可靠与
不可靠的历史对应，并保留低共视下已有的有效方向？

本轮运行的是 MASt3R 双视图匹配，不是完整 MASt3R-SLAM，也不是联合位姿优化。
不训练模型、不启动 NavDP/Habitat、不产生新导航 SR、不提交 HPC GPU 任务。
HPC 只按共享 SSH 手册读取已封存 RGB/JSON；不下载场景或 GT 深度。

## 数据与输入

- 使用已完成修复版共视实验的完整 159 queries：131 支持查询、28 原 Novel 对照。
- 从原三臂配对归档读取 raw 第一决策实际 anchor，保留其原 LingBot bearing。
- MASt3R 与重新运行的 SuperPoint/LightGlue 使用完全相同的原始目标 JPEG/anchor JPEG。
- 模型入口只读 RGB pair manifest；role、共视、GT、旧 SR 分开放在 evaluation_only.json。
- 保留 frame≥39 原分档，同时报告 frame≥8 可检索域分档与具体 raw anchor 的共视。
- 不以五档总体差异冒充严格的共视因果效应；同 history/scene 的样本相关。
- 这是已消费数据上的开发诊断；不能称为新 held-out 确认。

## 匹配实现与读数

MASt3R 使用官方预训练 metric checkpoint、官方 512 预处理和 reciprocal-NN 例程。
保存对应、描述子置信度、匹配数、空间分布，以及与 LightGlue 使用相同设置的
Fundamental-MAGSAC 残差读出。不计算替代 PnP，不修改 raw 位姿，不按结果重排 anchor。

注意：reciprocal-NN 读出不是 MASt3R-SLAM 的 pointmap-to-ray valid_match；不能
直接套用其 min_match_frac=0.3 并称为官方 SLAM 重定位复现。第一遍仅比较连续证据，
不从本批结果选最佳阈值，也不把 confidence 当作已校准的 Revisit 概率。

评分分开报告：

1. 原支持/Novel 类别上的匹配证据分布；边界 Novel 的真实共视保留。
2. 各共视档 raw 直线 endpoint bearing 误差与匹配证据的关系。
3. 匹配可信但 raw 位姿错误的例子，不能称为已认证的正确方向。
4. 首次模型加载、冷/热调用、匹配和几何读出的延迟与显存。

无新策略授权规则，因此不报告反事实导航 SR；旧成功标签只作错误分析。

## 资源和复现

运行目录：`.diagnostics/raw_match_verification_20260910_IJGnxv/`。
官方 MASt3R checkout：`.diagnostics/dependencies/MASt3R_official_20260910/`。
Git HEAD：`f5209afc300cec36239a7ac992263f36847bbba0`；DUSt3R 与 CroCo 使用其记录的 submodule commit。
独立 venv 只读复用本机 memnav 的 torch；新增依赖只装在该 venv，不更新主环境。
预训练模型用于本项目非商业研究，保留官方 LICENSE/CHECKPOINTS_NOTICE；不转发权重。
原真机 8888/18888 常驻服务不调用、不重启、不停止。

所有成果写入独立目录；论文、主方法、已有实验结果及用户未提交修改保留。
