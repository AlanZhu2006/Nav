# 低共视覆盖条件消融：HPC 提交记录

2026-09-10。主协议：`COVERAGE_ABLATION_HPC_PROTOCOL_20260910.md`。

## 本机触发依据（已完成，不是 HPC 新 SR）

2 histories / 2 scenes、2 Revisit + 2 Novel、4 arms =16 rollout，独立复核通过。
原CEC Revisit1/2、Novel1/2；去面积CEC Revisit2/2、Novel1/2；raw Revisit2/2、Novel0/2。
新规则对原CEC、raw各+1/−0，exact McNemar均p=1.0；这里只建立值得扩样的机制证据。
两个Novel的新旧CEC都逐动作精确等同native。本机私有服务已正常退出，真机服务未改。

## 已提交

| job | 内容 | 初次状态 |
|---|---|---|
| 17297864 | task10、71：Novel/低共视四臂集成检查 | PENDING / Priority |
| 17297865 | 其余157个查询，四臂同节点配对 | PENDING / afterok:17297864 |
| 17297866 | 全159结果完整性、配对统计汇总 | PENDING / afterany:17297864:17297865 |

GPU明确指定`a100_tandon`，account=`torch_pr_769_tandon_advanced`、QOS=`gpu48`。
每元素1GPU/10CPU/96GB/1小时；集成检查最多2并发，其余最多4并发。
CPU汇总`cpu_short`、2CPU/8GB/30分钟。正式池包含集成检查两项，共159查询/636rollout，
没有重复计入额外gate样本。gate仅按运行/配对/审计是否通过放行，不按导航SR筛选。

提交后`scontrol`独立回读通过：A100、gpu48、1GPU/10CPU/96GB/1小时以及两个数组
`10,71%2`、`0-9,11-70,72-158%4`均符合协议。集成任务随后显示`QOSGrpGRES`，
即项目GPU额度等待；不是启动报错、环境故障或HPC失联。未为绕过额度改分区或取消他人任务。

## 预检已完成

- 按HPC共享SSH手册使用`alantorch`默认master，并核实身份为`yz11502`。
- 新source-only bundle封存，未覆盖旧bundle/数据/结果/共享conda。
- 本机210项执行器/输入/证书/统计回归测试通过；12项为已有matplotlib弃用警告。
- HPC exact container下40项相关测试通过；2项为只读bundle不能写pytest cache的警告，
  无依赖或推理错误，未为了清理警告修改共享环境。
- 全159目标、在线A、目标RGB/depth注释的原始SHA和种子检查通过；目标depth仅验证数据，
  不作为模型输入。新旧CEC使用同一当前运行栈，不把旧HPC导航作为本轮配对对照。
- 8个Novel/Revisit×四臂CLI通过；Habitat/MemNav/NavDP实际环境导入与源码来源检查通过。
- `safe_sbatch` lint/test-only通过。提交后另外回读资源与依赖，不能只看脚本默认值。
- myquota：scratch约1.36TB/5TB、3,525,907/5,000,000文件（70%）。逐帧证据写node-local，
  以完整任务归档持久化，避免再次产生大量scratch散文件。

## 分析范围

CEC实际候选域frame≥8的五档分别14/26/23/32/36查询，另28原Novel。
主效应为低共视40查询的新CEC对原CEC配对SR差；高共视91查询检查损失；Novel报告相对
native的救回/破坏、接管和逐动作回退。统计按scene聚类，同时保留原q39定义敏感性分析。
旧Novel中一个q≈0.1167边界例保留原身份，并额外报告排除此例的27查询对照。

这是已消费总体上的单因素消融，不是新的blind确认。A来自旧实际mono执行，query使用
修复执行器；不称为新修复端到端A+query，也不提前更换默认方法或论文结论。

## 精确路径和版本

远端运行根目录：

```text
/scratch/yz11502/Research/Nav-axis-uturn-results/coverage_ablation_20260910/run_1df157b0b3d26b4f
```

该目录含`plan.json`、`preflight_success.json`、`submission.json`、
`submission_resource_verification.json`；结果在`evaluation/task_NNN/`，最终汇总为
`evaluation/paired_summary.json`。完整SR必须等待所有159项完成且归档/独立审计通过。

新runtime bundle：

```text
/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/coverage_ablation_1df157b0b3d26b4f
```

- source receipt SHA256：`1df157b0b3d26b4f0795b034f3e503c00058985d671d7d75f78e8c0820c2b7ca`
- source tar SHA256：`091ab72abc1ff95a8f84615ba3cb14021b279b4aaa6c4b662908127e05817691`
- plan SHA256：`818be9c95f4da587b0043f43e50a3dfb11880acbd044483a00f5f435797e8ea9`
- 原population SHA256：`182f3a6d2519d5b2c178b88345db4d0bb678088dd88487cfecaad9814c6fdaaf`

外部纯提交/预检脚本（不改变推理，未放进runtime bundle）：

- `prepare_coverage_ablation_hpc.sh` SHA256：`2ac9c6b31806c35639a01eb6a272679098a468bed6ff3979cda3ad733b6ff6a0`
- `submit_coverage_ablation_hpc.sh` SHA256：`a919602bd23969133bdfcd0ad0cad752e6ec6c51891006b9f90690611fbf08ce`

本机版本镜像和回执：

```text
/home/asus/Research/Nav-graph-blind/.diagnostics/coverage_hpc_20260910_H6Heyg
```

本轮没有修改论文或真机，没有commit/push。
