# main 归档验证记录（2026-09-07）

## 范围

按作者要求，先提交 Final14 Table III 精确 SPL 补跑，再整理研究仓库并合并至
`AlanZhu2006/Nav/main`。不推送论文仓库、不操作真机、不删除原始数据。
仓库入口：[最新总账](STATUS_20260907_MAIN_SYNC.md)、[代码地图](../docs/REPOSITORY_GUIDE.md)、
[实验索引](../docs/EXPERIMENT_INDEX.md)。

## 已完成的验证

- 发布前现有文件已保存：`before.patch`、`before_working_files.tar.gz`。
- 待提交盘点（新增本验证文档之前）：349文件、3,208,737 bytes；全部为文本。
- Python AST：172文件通过；JSON：33文件通过；Shell/Slurm：68文件通过 `bash -n`。
- 未发现单个>1MB文件；高置信度private-key/GitHub-token/access-key/带凭据URL扫描无命中。
  这是指定模式的检查，不声称能证明任意内容绝不敏感。
- 新入口5个Markdown的相对文件链接检查通过；旧生成器README副本逐字保留。
- `memnav`：**400 passed**；12条既有 Matplotlib/Pyparsing deprecation warnings未隐藏。
- `habitat`：**8 passed**（5条unittest + 3条无fixture纯函数断言），无需安装pytest。
- 生产SPL overlay：本机8项、HPC原容器8项通过；HPC全部21条source输入预检通过。
- 代码/JSON/Shell diff whitespace检查通过；历史Markdown保留双空格硬换行和原始字节，
  未为消除Git提示改写冻结文档。没有以单元测试替代新的GPU闭环成功率。

## 历史哈希例外：已退役SE2诊断

22个已有`.sha256` sidecar中21个与当前对应文件一致，1个历史例外为
`LONG_RANGE_SE2_PROGRESS_PROJECTION_DEV_20260902.md`。该文件已改成明确撤回部署主张、
取消两个queued gate后的审计说明，但仍保留最初协议的sidecar：

- 原封存hash：`9c8d5b2da5300d34f2b042a53cc6968ec3e990dddc3cea10c2614d4d83e0d8f7`。
- 当前审计文档hash：`148ed093de5de77acdf835a21f5c6139d54b129b69699cf967e45257d2ceb81b`。

本轮保留两者，不把sidecar重算成当前hash来伪造原协议一致性。旧SE2 prepare/submit
脚本不能以当前审计说明充当原冻结协议；该分支已退役，不在此次新任务和论文方法内。
这不影响新的Final14补跑，其13文件source receipt已在本机和HPC实际核验通过。

## 测试中修正了什么

初次把renderer测试放入memnav环境，三项collection因缺`quaternion`失败；随后使用
项目已有habitat解释器完成。没有修改依赖或把失败测试标成skip。

分环境后出现7项失败，均为`MemNavAgent.__new__`测试替身缺少新增字段。生产构造器
已定义这些字段；本轮只补齐3个测试文件中的cache/model/shadow初值，未加入新的
生产fallback、未修改算法阈值。修正后400项全过。

## 身份与Git边界

- 目标remote：`fork`，URL为`git@github-alan:AlanZhu2006/Nav.git`。
- SSH现场身份核验：`Hi AlanZhu2006!`。
- 本次提交使用作者/提交者`AlanZhu2006 <AlanZhu2006@users.noreply.github.com>`。
- 不使用本机另一个账号的`gh`登录进行写操作。
- 开始时研究分支HEAD为`2a309b6`；远端main为`1aac77e`，已含该旧研究HEAD及其他
  独立提交。整合必须保留main的既有训练/诊断代码，禁止reset/force push覆盖。
- 使用隔离整合工作树从最新main非破坏merge，再核对合并树及相关测试。

论文Abstract SHA仍为`3b389cd59ff8aec60d7fd0afbd11d7b01ddbd41ab32ccbe5abe37f877f4c9e2e`；
Introduction SHA仍为`e0de4f23ae6122c3dbbc8f4de420ef79113667dd0b78707715a3f77cab5d00d2`。
它们不在本轮Git暂存范围。

## 保留而未清除的内容

原始结果、失败任务目录、source bundles、checkpoints、dataset、视频、旧本地paper副本、
`.diagnostics/`、其他工作树均保持原位。清理主要是导航入口、历史说明、忽略规则及
版本化归档，不通过删掉负结果或移动冻结路径制造“干净主线”。

验证产物：`.diagnostics/git_main_sync_20260907_XQXM5q/`。正式补跑收据见
[FINAL14_TABLE3_EXACT_SPL_REPLAY_SUBMISSION](FINAL14_TABLE3_EXACT_SPL_REPLAY_SUBMISSION_20260907.json)。
