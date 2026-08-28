# HPC 提交加固三件套(2026-08-21)

背景:一周内约 20 次失败提交,三分之二属于"bundle 手工清单 + 公共分区"两个
可根治源头。三个工具把这两类问题移到提交前几秒钟内暴露。

## 1. `bundle_selftest.sh` — 上传前按节点条件自检 bundle

```bash
# entries.txt 每行: <解释器> import <模块>  或  <解释器> run <脚本 + 参数>
bash MemNavData/bundle_selftest.sh <staging_root> entries.txt
```

staging 目录在检查期间被临时置为只读(复刻 immutable bundle 条件),在
`PYTHONPATH=仅bundle + PYTHONNOUSERSITE=1 + PYTHONDONTWRITEBYTECODE=1` 的干净
子进程里逐条 import/运行,结束后检查是否有 `__pycache__` 泄写。任何一条失败
即非零退出。**rsync 之后、sbatch 之前,用远端真实解释器在登录节点再跑一遍**
——那一遍才能抓住 habitat env 缺 `requests` 这类解释器环境差异。

能提前抓住的历史事故:lifelong 的 depth_anything 缺包(8 task 团灭)、
factorial 的 py39/pycache、Gate D A2 的只读 pycache、组合实验的
LightGlue/kornia 缺失。

## 2. `slurm_safe_submit.sh` — 分区安全与模板 lint

```bash
source MemNavData/slurm_safe_submit.sh
safe_sbatch --lint-fatal --array=0-19%4 --export=ALL,... template.sbatch
```

`safe_sbatch` 在命令行强制 `--partition=h100_tandon,a100_tandon`(命令行
覆盖模板内旧的 #SBATCH 行);`lint_sbatch_template` 检出模板里的
h200_public / l40s_public 请求和缺失的 PYTHONDONTWRITEBYTECODE。现存模板中
11 个含 h200_public——不回改冻结模板,由 lint 在下次复用时拦截。

背景事实:本账号落到 gh 节点的任务要么被 root 以 QOSGrpGRES 强杀
(lifelong task 5、fresh 46/47/53、eval task 6),要么触发 LingBot 运行时
退化(Gate D task 11,~19 s/帧 后 SIGABRT);所有稳定完成的正式运行都在
h100_tandon / a100_tandon。若想解锁 H200 容量,先单独发一个 15 分钟
LingBot microbenchmark 到 h200_tandon(非 public)确认是否环境可修。

## 3. `eval_2leg_habitat.py --contract_dry_run` — 参数契约干跑

```bash
$HAB_PY eval_2leg_habitat.py --contract_dry_run <整套正式参数>
```

跑完全部参数校验(route/adapter/seed/threshold 组合)后在触碰 Habitat、
服务器、输出目录之前退出。已在登录节点用真实 habitat 解释器验证:合法组合
输出一行 OK、exit 0;非法组合(如 certified route 缺 verified_bearing_v1)
30 秒内 exit 1——这条错误在 2026-08-18 曾消耗一整轮 GPU 排队。submit 脚本
应对每个臂的完整参数串各干跑一次。

## 推荐的提交流程

```text
本地 bundle_selftest → rsync → 登录节点 bundle_selftest(远端解释器)
  → 登录节点 contract_dry_run(每臂)→ source slurm_safe_submit.sh
  → safe_sbatch --lint-fatal → 摘要作业用 afterany + 输出完整性自检
```

尚未做(下一批):通用缺失索引修复工具 `repair_missing_array_indices.sh`、
按场景打包 task 摊薄模型加载开销、h200_tandon 诊断作业。

## 4. 共享 SSH socket 必须同时验证“文件”和“master”

2026-08-24 再次确认：`ControlPath` 是 socket 文件并不表示连接仍然可用。master
进程异常退出后，孤儿 socket 仍可留在磁盘上；此时 `ssh -S ...`、`rsync` 和
`scp` 会静默等待，看起来很像 HPC 或 Slurm 卡住。

提交前必须同时做：

```bash
socket=/home/asus/.ssh/cm-h3life-20260824
test -S "$socket"
timeout 15 ssh -O check -S "$socket" alantorch
timeout 15 ssh -tt -o BatchMode=yes -o ControlMaster=no \
  -S "$socket" alantorch 'id -un; hostname'
```

任一步超时都不能继续提交，也不能据此声称“HPC 连不上”。应保留旧 socket 作
事故证据，换一个任务专用 `ControlPath` 建立新的 master。所有远端命令、rsync
和 scp 必须显式使用同一个已验证 socket，并分别设置有限超时。交互 shell 若仍
占用该 master，也要先确认服务端允许新 channel；不要一边保留交互 channel，一边
默认 rsync 一定可复用。

### 4.1 2026-08-24：先区分沙箱拒绝与 socket 失效

在受限 Codex execution profile 中，连接现有 Unix control socket 可能直接返回：

```text
Control socket connect(...): Operation not permitted
```

这表示本地执行沙箱禁止访问 socket，并不证明 master 已死，更不能据此触发新的
Microsoft device login。应先在允许访问该 socket 的执行 profile 中重跑同一个
`ssh -O check`。本次默认共享 master 随后被验证为：PID `1613846`、用户
`yz11502`、节点 `torch-login-b-3`；用户此前所说“共享 SSH 正常”是准确的。

另一个独立陷阱是把强制 PTY 的 SSH 放入 command substitution 或 pipeline：

```bash
identity=$(ssh -tt ... alantorch 'id -un')
```

此时 SSH 可能因试图读取提交端控制终端收到 `SIGTTIN`，进程状态变成 `T`，脚本看似
卡死。脚本化 PTY 调用必须关闭 stdin：

```bash
identity=$(ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
  -S "$socket" alantorch 'id -un' | tr -d '\r')
```

诊断时同时检查 `ps ... stat`；`T` 是本地 job-control 停止，不是远端 checksum、Slurm
或认证故障。本次 full-mono lifelong 提交在第一条 `id -un` 处发现并修复这一问题，
当时尚未上传科学输出或创建任何 Slurm job。

## 5. Array 出现确定性基础设施失败时：补 fragment，不重跑 population

2026-08-24 的 full-mono lifelong build 暴露了三个 Slurm/数据边界问题。

第一，父 population 的“零历史 scene”有两种物理表示：有的带空 `online_a/manifest`，
有的只有 hash-sealed per-scene completion，明确记录 source generation incomplete、
materialized=0，因此合法地没有 `online_a/`。下游不能一概用
`require(online_root.is_dir())`；只有在同时复核父 population receipt、scene completion
SHA、零 success/零 materialized/零 retained、且 outcomes-read=false 后，才能生成一个
零行 attrition fragment。绝不能创建假 history 或据此放宽构造阈值。

第二，本站 Slurm 对 pending job 的 `scontrol update TimeLimit=...` 和
`scontrol update Dependency=...` 都可能只返回 `Unspecified error`。不要假定原地修改
成功；必须立即用 `scontrol show job` 复读。dependency 无法修改时，保留原 array 和
产物，提交 replacement downstream DAG，并把旧 DAG 的精确 job IDs 记录进 receipt。

第三，replacement DAG 若在取消旧 pending arrays 前整条提交，可能碰到
`MaxSubmitPU`/“Job dependency problem”：旧 prefix/eval array 虽永远不会运行，仍占
submitted-task 配额。安全顺序是：

```text
读取失败索引与原始日志
  → 仅由冻结输入确定 exact repair manifest
  → 新 immutable repair bundle + 独立输出（或原输出中的缺失目录）
  → repair raw receipts 逐文件重算 SHA
  → replacement seal 与首个 downstream node 成功创建
  → 精确取消仍为 dependency-pending、从未运行的旧 downstream IDs
  → 补交其余 replacement DAG
  → receipt 记录原 array、repair job、旧 IDs 和新 IDs
```

本次实例中，原 build `16265026` 的零历史修复索引为
`11,15,34,40,44`，CPU repair `16266646` 为 5/5 completed；随后 replacement DAG
从 seal `16266719` 接续。这个流程不读取 query outcome，也不把 infrastructure failure
计成导航失败。

## 6. Array 上界必须来自 sealed population，而不是 source 上界

2026-08-25 的 HM3D lifelong power expansion 暴露了另一类确定性浪费：上游最多可以构造
260 个 candidate，不代表最终 sealed population 就有 260 条。正式 population 实际只有
8 条，但旧 deferred launcher 仍提交 `0-259`；索引 8 以后虽然会在节点上识别为
`outside_sealed_population` 并退出，却仍要逐个等待 GPU allocation，并阻塞依赖于整个
array 的 finalizer。

修复后的约束是：

1. launcher 必须先验证 `SEALED` 和 `population.json.sha256`；
2. 从 `population.json["accepted"]` 读取精确、非零长度；
3. collection 使用 source population 的精确长度；
4. evaluation 使用 factual-C success population 的精确长度；
5. 长度超过冻结最大值 260 时 fail closed；
6. submission receipt 记录 `sealed_population_count` 和实际 array 范围。

当前已经提交的 job `16318975` 来自旧 immutable bundle，不能在原地用新源码改写；其
多余 pending indices 属于基础设施空任务，不是导航 episode。后续新 bundle 必须使用
`slurm_hm3d_fullmono_shared_c_deferred.sbatch` 的 v2 精确数组合约。

## 7. 远端复合预检必须自身 fail-fast

2026-08-28 的 ViNT--CEC mechanism submit 暴露了一个 shell 边界：本地 submit 脚本虽然
有 `set -euo pipefail`，但通过 SSH 发送的多行复合命令是在另一个远端 shell 中执行，
不会继承本地 shell 选项。一次 Habitat `requests` 导入自检失败后，远端后续 checksum
和发布命令仍继续，导致只提交了一个不带完整 submission receipt 的 gate。

所有包含两条以上命令、且后面可能触发 bundle publish 或 `sbatch` 的远端命令必须以
下面一行开头：

```bash
set -euo pipefail
```

另外，环境自检必须复刻正式 runner 的完整、显式 `PYTHONPATH`。若依赖来自解释器自带
的 vendored 路径（本次为 Habitat 环境中 `pip/_vendor/requests`），应先解析真实
`site-packages`、验证目标文件可读，再通过 `SELFTEST_EXTRA_PYTHONPATH` 传给
`bundle_selftest.sh`；不能因为登录节点的默认 import path 不同而放宽自检。

本次事故、旧 gate 和唯一权威 retry DAG 的逐项收据见
`VINT_CEC_BEARING_ALIGNMENT_SUBMISSION_INCIDENT_20260828.md`。

## 8. 不依赖交互 shell 的解释器别名

2026-08-28 的最终本地预检发现，项目的 Habitat Python 3.9 存在于
`/home/asus/miniconda3/envs/habitat/bin/python3.9`，但裸命令 `python3.9` 不在登录
shell 的 `PATH`。这不是代码错误，但如果批处理脚本依赖交互式 conda 初始化，会在
不同节点或非交互 SSH 中表现为 `command not found`。

因此所有正式脚本和 bundle self-test 必须：

1. 使用 receipt 固定的解释器绝对路径；
2. 在提交前检查该路径 `-x`；
3. 对需要 Habitat 语法兼容性的入口，显式用该解释器执行 `py_compile`；
4. 不把 `conda activate`、用户 `.bashrc` 或裸 `python`/`python3.9` 当成运行时依赖。

本轮 modified/untracked 源码已用显式 Habitat Python 3.9 完成 21 个文件的
`py_compile`；9 个 shell 入口均通过 `bash -n`。

## 9. 共享多 GPU 节点上的 TCP 端口不是 job-private

2026-08-28 的 Final14 zero-depth array 中，index 19 与同节点另一进程竞争
`35756`，NavDP server 在 evaluator 启动前以 `Address already in use` 退出。其余
20 个单元完成，失败单元没有 query outcome。单纯用 PID/Slurm job ID 取模再做一次
`ss` 检查仍存在 TOCTOU race；GPU allocation 也不隔离宿主机 TCP namespace。

后续多进程 cell 使用 `slurm_port_pair.sh`：

- job-keyed 候选顺序；
- `/tmp` node-local `flock`，父 shell 全程持有；
- 对连续 2-port 或 5-port block 逐端口检查 listener；
- 最多 128 个候选，失败则 fail closed；
- server 退出后再释放锁。

旧 partial 只读归档，未删除；exact repair 只补 index 19。repair job `16502265`
和 replacement verifier `16502270` 均为 `COMPLETED 0:0`。已知仍使用旧端口算法但尚未
运行的 authority/lifelong bundle 被废弃并以 portsafe immutable bundle 重提。

## 10. Namespace package 会掩盖不完整的 source bundle

2026-08-28 的 portsafe authority smoke `16502418` 暴露了一个不同于“缺少第三方包”
的问题。启动脚本的 `PYTHONPATH` 同时包含新 overlay 和旧 base bundle；两边的
`MemNavData` 都是可合并的 namespace package。新 overlay 漏装
`monocular_depth_runtime.py` 时，普通 import 仍然成功，但实际解析到旧 bundle。
结果是新 `memnav_server.py` 在第一次 retrieval transaction 才发现旧模块没有
`bind_monocular_depth_transaction`。38 个不覆盖该延迟路径的启动测试均通过，说明
“import 成功”本身不是依赖闭包证明。

正式 bundle 现在执行以下约束：

1. 所有 overlay 修改过或直接调用的运行时模块及其测试必须显式进入 files manifest；
2. 对会被 base bundle 同名模块掩盖的关键 import，启动前检查 `module.__file__` 的
   resolved path 必须等于当前 immutable bundle 中的预期文件；
3. 同一个 provenance check 在本地 staging、HPC 登录节点和 GPU runner 内都执行；
4. delayed endpoint 所需 API 至少有一个启动前单元测试覆盖，不能只测 server import；
5. provenance 不符时必须在模型加载和 episode outcome 写入前 fail closed。

修复后的 authority bundle receipt 前缀为 `18fe24537b840871`；replacement DAG 是
`16503212 -> 16503217 -> 16503241`。失败 smoke 没有产生 arm outcome，其下游由
`afterok` 自动取消。

## 11. 执行成功不等于收据字段完整

provenance 修复后的 smoke `16503212` 完整执行两臂后，在 post-run audit 发现
authority policy 字段为 `None`。HTTP response 与内部 proposal-attempt authority
均正确，缺失发生在 generic evaluator 的显式 `plans.append(...)` 序列化白名单。
因此模型、route 和 trajectory 都运行成功，但无法证明唯一干预变量，正式数组仍必须
阻止。

预防规则：

1. 新增决策变量时，必须同时测试 endpoint response、episode plan receipt 和独立
   verifier 三层 schema；
2. 审计关键字段用一个 dependency-free helper 显式复制，不在多个长字典中手写；
3. receipt helper 必须能在最小 Habitat 解释器中导入，不能为了两行 JSON 复制引入
   OpenCV、PyTorch 或其他 server-only 依赖；
4. smoke 必须跑到 completion/verifier 前置审计，而不是只以 server 健康或 episode
   rollout 退出码作为通过条件。

修复只新增 `certified_relocalization_authority` 和
`certified_relocalization_authority_policy` 两个诊断字段，不改变控制。当前 replacement
DAG 是 `16504303 -> 16504304 -> 16504307`。
