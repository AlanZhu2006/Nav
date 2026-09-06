# HM3D authority-spectrum：仅基础设施的 exact repair

日期：2026-09-06。继承 2026-09-04 冻结四臂协议，不改样本、模型、阈值、
深度来源、随机种子、执行步数或同一 history 内的配对顺序。

## 原任务与修复范围

- 原 task bundle：`hm3d_table1_authority_spectrum_9bb6fc2fcc16303b`。
- 原 server overlay：`hm3d_table1_navdp_authority_transaction_718661db1733d5de`。
- 原 runtime closure：`hm3d_table1_navdp_authority_transaction_repair_51ee9a4ca063c7f1`。
- 原正式 array：`16929030`；20 个 history 完成，8 个基础设施失败。
- 精确补跑索引：`2,3,6,7,10,13,18,21`。每个索引重新执行全部四臂，
  不拼接失败 history 内已经运行的部分臂。
- 20 个完成目录保持原样；新 run root 以只读来源链接复用，绑定原 completion SHA。
  index 2 的旧 partial 也留在原路径，不覆盖、不删除。

## 仅有的执行修改

1. 每个 array element 的 runtime 目录加入 `history_${SLURM_ARRAY_TASK_ID}`，
   消除同场景多 history 的目录碰撞。使用现有 wrapper 的 `RUNTIME_ATTEMPT`，
   不替换现有 wrapper 或 evaluator。
2. NavDP 的相同 JPEG 缓存命中若属于不同 transaction，将其视为 cache miss，
   重新获取调用者 token 绑定的深度。保持原来的 SHA/frame/token 验证；
   不把错误深度复用为有效输入。该修复已存在于研究 worktree，
   这里只把相同逻辑移入遗漏它的旧 overlay，不带入其他未提交改动。
3. 按共享 SSH/HPC 手册使用 `h100_tandon,a100_tandon`，每元素 1 GPU、1 小时，
   最大并发 4；不再申请 `h200_public`。

## 执行顺序

- 对新冻结 overlay 使用真实远端 Python 做缓存回归与 import 路径验证；
- 运行同场景双 history 目录测试、四臂契约测试及 shell lint；
- 先正式补跑 index 2；它完成后再释放其余 7 个 history；
- 完成后用原完整分母的 summary + independent verifier 汇总 28 histories；
  analysis 使用 `afterany`，缺失仍然报不完整，不生成部分正式 SR。

补跑结果仍是 retrospective authority ablation，不升级为新的 held-out confirmation。
提交 receipt 另行记录新 bundle、job ID 与 20 个复用来源，不改旧 receipt。
