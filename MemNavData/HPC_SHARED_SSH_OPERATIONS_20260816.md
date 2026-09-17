# HPC 共享 SSH、Slurm、数据传输与故障排查手册

最后更新：2026-09-18（Asia/Shanghai）

适用范围：NYU Torch HPC 上的 GEM / MemNav / Certified Episodic Compass / Pi3X
开发、数据传输、冻结评测与结果审计。

本文只整理**基础设施与可复现性问题**。方法效果、SR、统计显著性和论文结论以最新
`STATUS_*.md`、冻结 protocol、summary 与 independent verifier 为准。任何
infrastructure failure 都不能被解释为方法成功或失败。

## 0. 最重要的六条规则

1. 本项目唯一正确的共享入口是 `ssh alantorch` 对应的 `yz11502`；普通交互登录能用
   时，不要另造 socket，也不要反复触发 Microsoft device login。
2. SSH 可用、Slurm controller 可用、GPU 可立即调度是三件不同的事，必须分别检查。
3. 每个配对 episode 的所有 arm 必须在同一 array element、同一节点和同一运行栈中
   完成；不同 episode 才允许分配到不同的合格 GPU。
4. `#SBATCH --time` 是**每个 array element**的上限，不是整个 array 的总时间；必须
   用完整 gate 的实测时长决定，不能沿用没有依据的 8 小时模板。
5. 小型源码 bundle/receipt 走 SCP、SFTP 或 rsync；大型数据集、模型、SquashFS
   overlay 和批量结果优先走 **Globus**。
6. 失败输出不删除、不覆盖。先保留日志和 partial output，再只重跑 exact failed
   index；summary 和 independent verifier 必须重新绑定到完整成功的依赖链。

### 0.1 默认执行顺序：先验证完整任务，再批量提交

新实验、新依赖或更换运行栈时，按下表推进。这里的 **gate** 指一项完整验证任务：
使用正式配置，走完配对运行、结果落盘和独立校验，而不只是成功 import 或启动服务。
已验证且未改变的部分复用原收据；改动影响到哪里，就重做对应检查。

| 顺序 | 做什么 | 进入下一步的条件 |
|---|---|---|
| 1. 连接与路径 | 用 `ssh alantorch` 确认账户；检查数据、配额、临时目录和输出位置 | 目标计算节点能读输入、写结果；见 §1、§6、§11–12 |
| 2. 依赖准备 | 固定解释器、源码、权重、容器与扩展库，完整打包并核验 | 实际运行环境加载的是预期文件；见 §5.4 |
| 3. 低成本预检 | 先检查全部输入、参数、CPU 逻辑、请求解析和输出字段 | 缺文件、错误参数与序列化问题在占用长时间 GPU 前暴露 |
| 4. GPU 完整验证 | 在拟用 GPU 上跑完整配对，覆盖长历史、目标切换和实际常驻状态 | 所有分支与独立校验通过，记录耗时、内存、显存和落盘量 |
| 5. 小批量运行 | 先低并发运行少量固定索引，确认服务隔离、端口和 I/O | 未出现新的运行故障，再逐步增加并发 |
| 6. 正式提交 | 显式传资源参数，读回 Slurm 状态，连接汇总与校验任务 | 索引、运行栈、时限和依赖都与计划相符；见 §2–3 |
| 7. 收尾与补跑 | 校验完整结果，保留失败记录，只补失败配对 | 结果可从持久存储复核；见 §9 |

优先测试容易拖到后期才出错的环节：第二臂模型加载、长历史显存峰值、最后一个目标、
真实图像请求、结果汇总和归档。验证样本按覆盖范围和资源需求选择，不按成功结果挑选。
正式配置下已完成并校验的 gate 样本只计一次；后续数组排除这些索引。
缩短步数或改过配置的工程测试单独保存，不并入正式结果。

连接配置、配额和可用分区会变化。下面保留的 socket、作业号与资源数值是项目记录；
实际使用时从 `ssh -G` 和集群状态读回，不直接照抄历史值。

## 1. 已确认的共享 SSH 配置

最近记录的 `ssh -G alantorch` 配置为：

- 用户：`yz11502`；
- 主机：`login.torch.hpc.nyu.edu`；
- `ControlMaster auto`；
- 默认 control socket：
  `/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3`；
- `ControlPersist 86400`。

这个默认 socket 是项目的权威共享连接。另一个解析为 `yz11445` 的活动 socket 属于
不同账户，禁止复用、关闭或替换。

### 1.1 常用 SSH 命令

```bash
# 查看最终生效的用户、主机、socket 和复用设置
ssh -G alantorch | grep -E '^(user|hostname|controlmaster|controlpath|controlpersist) '

# 检查默认 master 是否仍活着；ControlPath 应从上一条命令读取
ssh -O check -o ControlPath=/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3 alantorch

# 普通交互登录
ssh alantorch

# 最小身份检查
ssh alantorch 'id -un; hostname; date'

# no-PTY mux 卡住时，复现用户真实使用的 PTY 路径
ssh -tt -o BatchMode=yes -o ControlMaster=no \
  -S /home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3 \
  alantorch
```

登录后至少确认：

```bash
id -un       # 必须是 yz11502
hostname     # 应是 torch-login-*
date
pwd
```

### 1.2 2026-08-16 的 mux 误判

当时默认 master 能通过 `ssh -O check`，但下面的 no-PTY channel 卡在
`mux_client_request_session`：

```bash
ssh -o BatchMode=yes alantorch 'id -un; hostname'
```

这曾被错误解释为 MFA 过期或 HPC 宕机。实际情况是：用户普通登录正常，显式 PTY
立即进入 `torch-login-b-3`，SFTP/SCP 也能通过相同 master 工作。已复制的冻结协议在
远端复现 SHA-256：
`a019a49248950a537b14c651b7a812ba7ccb421504901f8a8de030d63ae3a230`。

因此正确结论只是：**一个 multiplexed no-PTY command channel 卡住**。底层原因没有
被证明，禁止进一步猜测为认证失效、网络中断或 Slurm 故障。

### 1.3 SSH 决策流程

1. 先运行 `ssh -G alantorch`，不要依赖记忆中的 socket 路径。
2. 再运行 `ssh -O check`。
3. no-PTY 超时时，先用显式 PTY 验证真实登录路径。
4. PTY 成功而 no-PTY 卡住时，把状态改变命令留在已认证 PTY 内执行；文件可继续通过
   同一默认 socket 的 SCP/SFTP 传输。
5. 不得仅凭一次 timeout 要求用户重新认证、宣布 HPC down 或把任务标记 blocked。
6. 登录节点曾缺少 `rg`；远端脚本启用 `set -e` 前必须先检查命令是否存在，必要时用
   `grep`、`find`、`sed` 等 POSIX/GNU 工具。

### 1.4 2026-08-21：复用 master 做临时只读 localhost tunnel

再次观察到：默认 master 的 `ssh -O check` 正常、已认证 PTY 正常，但新开的
`ssh alantorch hostname` / no-PTY session 仍可能无输出等待。此时不应绕过 master、
反复 device login 或宣称 HPC 失联。`~C` 在 multiplexed slave 中会明确返回
`escape not available to multiplexed sessions`；应通过 master control command 管理
转发。

当 SCP/SFTP 的新 channel 也无法及时建立、且只需拉取少量已解析文件时，验证过的
最小只读路径是：

```bash
# 1. 在已经认证的远端 PTY 中，仅绑定登录节点 localhost
python -m http.server REMOTE_PORT --bind 127.0.0.1 \
  --directory EXACT_READ_ONLY_ROOT >/tmp/PROJECT_http.log 2>&1 &
# 立即记录返回的 PID

# 2. 工作站通过权威 master 增加 localhost-only forwarding
ssh -S /home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3 \
  -O forward -L LOCAL_PORT:127.0.0.1:REMOTE_PORT alantorch

# 3. 只下载已解析的显式文件，并按 sealed receipt 逐项核验 SHA-256
curl --fail --silent --show-error \
  http://127.0.0.1:LOCAL_PORT/RELATIVE_FILE -o LOCAL_FILE
sha256sum LOCAL_FILE

# 4. 完成后立刻在远端 PTY kill 精确 PID，并撤销 forwarding
ssh -S /home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3 \
  -O cancel -L LOCAL_PORT:127.0.0.1:REMOTE_PORT alantorch
```

安全边界：server 和 tunnel 两端都只绑定 `127.0.0.1`；不能用 `0.0.0.0`；不能把
大数据集走这条路径；不能用目录级模糊下载；结束时必须 kill 精确 PID、cancel 精确
forward，并用原 benchmark/manifest 中的哈希验证每个文件。本次用该方式拉取
`621,086` bytes 的 NNR 最小镜像，benchmark、A/B trace、metadata、parquet 和三个
goal assets 全部逐项哈希一致。

### 1.5 2026-08-28：socket 可响应仍不代表账户正确

`ssh -S SOME_SOCKET alantorch ...` 会直接附着到 `SOME_SOCKET` 对应的 master；命令行
上的 alias 不会把这个 master 重新认证成 `ssh -G alantorch` 所声明的用户。因此，
“遍历 `~/.ssh/cm-*` 并选择第一个能运行命令的 socket”是错误且危险的做法。

本次同时存在两个可通过 `ssh -O check` 的 socket：默认 `yz11502` master 的新 no-PTY
channel 卡住，而较新的另一个 socket 可以立即执行命令，但远端 `id -un` 实际是
`yz11445`。后者能遍历 `/scratch/yz11502` 的部分目录，却不能读取 owner-only 的冻结
bundle，于是表现为数百个 `sha256sum: Permission denied`。这不是 bundle 损坏，也不是
Slurm 或文件系统故障；准备链在上传和 `sbatch` 前已 fail closed。

自动化必须同时验证连通性和身份：

```bash
expected_user=yz11502
socket=/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3
actual_user=$(timeout 15 ssh -n -T -o BatchMode=yes \
  -o ControlMaster=no -S "$socket" alantorch 'id -un' 2>/dev/null || true)
test "$actual_user" = "$expected_user"
```

禁止把另一个账户的 responsive socket 当作 fallback，禁止关闭或修改它，也禁止在身份
未核对时执行远端写入、rsync 或 sbatch。`ssh -O check` 只能证明本地 master 进程接受
control 命令，不能证明它属于目标账户，也不能证明新 session channel 可用。

## 2. Slurm 常用命令

项目账户：`torch_pr_769_tandon_advanced`。当前 GPU workflow 使用 QOS：`gpu48`。

### 2.1 查看分区、GPU 和队列

```bash
# 查看本项目已经验证过的三个 GPU 分区
sinfo -p h100_tandon,h200_public,a100_tandon \
  -o '%P %a %l %D %t %G'

# 查看自己的任务
squeue -u yz11502 \
  -o '%.18i %.12P %.22j %.10T %.10M %.10l %.6D %R'

# 查看同一项目账户正在占用/等待的资源
squeue -A torch_pr_769_tandon_advanced \
  -o '%.18i %.10u %.12P %.22j %.8T %.10M %.4D %b %R'

# 调度优先级与预计启动时间；预计时间会动态变化
sprio -j JOB_ID -l
squeue --start -j JOB_ID -o '%.18i %.10T %.19S %R'
```

### 2.2 查看历史状态、真实提交命令和节点

`squeue` 只适合活跃任务。任务完成后，`squeue -j JOB_ID` 可能返回
`Invalid job id specified`；这不表示 controller down，应改查 `sacct`。

```bash
sacct -j JOB_ID \
  --format=JobID,JobName%24,Partition%18,State,Elapsed,Timelimit,ExitCode,NodeList%16,Start,End -P

# 追溯完整 sbatch 提交行和工作目录
sacct -j JOB_ID -X \
  --format=JobID,JobName,SubmitLine%240,WorkDir%160 -P

# 查看活动任务的完整 Slurm 元数据
scontrol show job -dd JOB_ID

# array 显示 ID 与内部 JobId 不同时，查看映射
scontrol show job ARRAY_JOB_ID_ARRAY_INDEX | tr ' ' '\n' \
  | grep -E '^(JobId|ArrayJobId|ArrayTaskId|JobState|TimeLimit)='
```

### 2.3 查看日志与当前 arm

当前 SBATCH 通常使用 `%x_%A_%a`：job name、array job id、array index。

```bash
tail -n 200 /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/JOBNAME_ARRAYID_INDEX.out
tail -n 200 /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/JOBNAME_ARRAYID_INDEX.err

# 查最近完成的 episode receipt
find RUN_ROOT/evaluation -name completion.json \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort | tail

# 查常见硬错误；warning 不能脱离 exit status 单独定性
grep -RInE 'ABORT|Traceback|SIGABRT|CUDA error|RuntimeError|core dumped' \
  RUN_ROOT/logs RUN_ROOT/evaluation
```

### 2.4 标准提交模板

CLI 参数会覆盖 SBATCH 文件中的同名默认值。提交前必须显式核对 partition、account、
QOS、GPU、CPU、memory、time 和 array concurrency。`FROZEN_INDEX_SPEC` 必须由冻结
manifest 导出；不能假设索引连续，也不能把下面的占位符原样提交。例如 Final14 的
正式索引是 `0-20,42-62`，不是 `0-41`。分区只填写已通过完整 gate 的集合；其余资源
数值也须按本次实测替换。

```bash
FROZEN_INDEX_SPEC='REPLACE_WITH_EXACT_FROZEN_INDICES'
VALIDATED_PARTITIONS='REPLACE_WITH_PARTITIONS_THAT_PASSED_FULL_GATE'

eval_id=$(sbatch --parsable \
  --partition="${VALIDATED_PARTITIONS}" \
  --account=torch_pr_769_tandon_advanced \
  --qos=gpu48 \
  --gres=gpu:1 \
  --cpus-per-task=12 \
  --mem=96G \
  --time=01:00:00 \
  --array="${FROZEN_INDEX_SPEC}%4" \
  --export=ALL,... \
  MemNavData/slurm_EXPERIMENT_eval.sbatch)

summary_id=$(sbatch --parsable \
  --dependency=afterok:"$eval_id" --kill-on-invalid-dep=yes \
  --export=ALL,... MemNavData/slurm_EXPERIMENT_summary.sbatch)

verify_id=$(sbatch --parsable \
  --dependency=afterok:"$summary_id" --kill-on-invalid-dep=yes \
  --export=ALL,... MemNavData/slurm_EXPERIMENT_verify.sbatch)
```

`--array=...%4` 只表示最多四个元素并发，不是预留四张 GPU。实际可能只有一张或零张。

`sbatch --test-only` 检查提交参数并估计调度时间，不执行依赖、渲染或导航测试。
正式提交后用 `scontrol show job -dd JOB_ID` 核对 `Partition`、`TimeLimit`、`ReqTRES`、
`Dependency` 和数组并发上限，保存实际提交命令。尤其留意 `safe_sbatch` 等封装器注入
的 CLI 默认值。参数优先级与 `--test-only` 的含义见 [Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)。

`afterok` 只依据上游退出码；包装脚本必须把运行、归档和校验失败正确传回。
依赖整个数组时，须所有成员成功才会放行，见 [Slurm 数组依赖说明](https://slurm.schedmd.com/job_array.html)。
下游任务应紧接上游提交，避免短任务完成并从控制器清除后才绑定依赖；已有旧结果的
处理流程见 §19。提交命令超时先查作业是否已创建，不直接再执行一遍。

### 2.5 取消、保留和 exact retry

取消是状态改变操作，必须先解析准确目标。禁止用模糊 glob 或错误账户。

```bash
# 取消单个任务
scancel JOB_ID

# 只取消某个 array 中尚未启动的元素，保留 running/completed
scancel --state=PENDING ARRAY_JOB_ID

# 取消已经失效的 summary/verifier dependency
scancel SUMMARY_JOB_ID VERIFY_JOB_ID
```

如果 runtime abort 留下 partial directory：

1. 将该 episode、对应 server logs 和 buffer 整体移动到
   `RUN_ROOT/failed_attempts/<job_and_reason>/`；
2. 不删除、不覆盖，不把 partial metric 纳入结果；
3. 以相同 bundle、manifest、seed、arm 和 output contract 重跑 exact index；
4. summary 依赖同时绑定 retained running task、replacement array 和 exact retry；
5. verifier 仍只依赖新 summary。

重跑单位是该索引的完整配对，不是只补最后失败的一臂。已有若干动作或一个成功分支，
不等于可以恢复其余运行；除非已验证的 checkpoint 同时保存模拟器、模型/几何状态、
随机数状态和目标进度，否则从该配对起点重跑。HTTP 请求可能已推进流式状态，响应
丢失时也不能盲目重发同一观测。

## 3. 分区、并发与申请时长

### 3.1 分区选择

- 对允许跨合格硬件运行、episode 内严格配对的评测，只列入本次运行栈通过完整配对
  gate 的分区；有多个合格分区时，让 Slurm 选择最早合法资源。
- 若结论与 GPU 型号、数值稳定性或性能有关，必须固定一个分区，并在 receipt 中记录
  GPU name、UUID、driver 和显存。
- 不得为了缩短排队临时加入仅通过 import、短渲染或 server-ready 检查的分区。
- 同一个 episode 的五个 arm 不可拆到不同 GPU；跨机器 CUDA 非确定性曾让同一 Goal-A
  路径出现 `7.81 m` 与 `2.70 m` 的差异。

常见 pending reason：

| Slurm reason | 含义 | 正确处理 |
|---|---|---|
| `Priority` | 有更高优先级任务 | 查看 `sprio`/`squeue --start`，不是代码错误 |
| `QOSGrpGRES` | 项目/QOS 的 GPU 总量受限 | 查看 account 队列，等待额度释放 |
| `QOSMaxGRESPerUser` | 单用户 GPU 数达到上限 | 不要重复提交；等待自己的任务完成 |
| `Dependency` | 上游未成功完成 | 查上游 `sacct`，不要把它当排队 |
| `Resources` | 当前没有满足请求的节点 | 核对分区与资源规格 |
| `Invalid job id specified` | job 已不在 active queue 或 ID 错误 | 用 `sacct` 查历史，不代表 controller down |

### 3.2 申请时长规则

`#SBATCH --time` 作用于每个 array element。过大的上限会让本可 backfill 的短任务看起来
像长任务。

1. gate 的计时包括输入准备、server 启动、全部 paired arms、结果校验与归档。
2. 根据合格 GPU 上的连续运行、最大冻结 step budget、长历史和多臂常驻状态估算
   上界；不能只用快速成功样本。已有任务曾采用约 2–3 倍实测余量，具体仍须覆盖
   本任务的准备和收尾开销，并以小批量实际耗时复核。
3. 同时记录 requested time 与 observed time；总 wall-clock 还受排队和实际并发影响。
4. 提交后核查真正生效的时限：

   ```bash
   squeue -j JOB_ID -o '%.18i %.10T %.10M %.10l %R'
   ```

### 3.3 Final14 的 8 小时到 1 小时修正

- 五臂 gate 实测 `18:12`；随后成功元素约 `11--16` 分钟。
- 旧 SBATCH 却为每个元素申请 `08:00:00`。
- “约 12.6 小时”只是 42 个约 18 分钟任务单卡串行的算术总量，不是申请了 12 小时。
- 生产时限改为 `01:00:00`，仍有三倍以上 H100 实测余量，并明显改善短槽调度。

Torch Slurm 拒绝了用户态原地修改：

```text
scontrol update ... TimeLimit=01:00:00
Unspecified error
```

因此采用的安全流程是：保留 completed/running，只取消 pending，以完全相同参数和
explicit `--time=01:00:00` 重提 exact pending indices，再重建 summary/verifier
dependency。当前审计 ID：

- replacement array：`15903404`；
- unrelated Habitat-aborted index 4 exact retry：`15903546`；
- replacement summary/verifier：`15903547` / `15903548`。

这是 infrastructure-only correction，没有改变 population、方法、模型、阈值、seed、
arm 或成功判据。

## 4. 文件传输：SSH 与 Globus 如何分工

### 4.1 小文件：SCP/SFTP/rsync

适合 source bundle、脚本、manifest、receipt 和少量日志。

```bash
scp LOCAL_FILE alantorch:/scratch/yz11502/Research/DESTINATION/

# 显式复用 control socket 时，scp 要用 -o ControlPath；scp 的 -S
# 表示“连接程序”，不是 socket 路径。
scp -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath=/home/asus/.ssh/cm-e3ce4155ccb925413d599e13706baddf79fddff3 \
  LOCAL_FILE alantorch:/scratch/yz11502/Research/DESTINATION/

rsync -av --partial --info=progress2 \
  LOCAL_DIR/ alantorch:/scratch/yz11502/Research/DESTINATION/

# 远端核验
ssh alantorch 'stat -c "%s %n" /scratch/yz11502/Research/DESTINATION/FILE; sha256sum /scratch/yz11502/Research/DESTINATION/FILE'
```

不要用 `rsync --delete`，除非用户明确授权且目标已被精确解析。

### 4.2 大文件：Globus

适合大型数据集、多 GB checkpoint、scene archives、SquashFS overlay 和批量结果镜像。
优先 collection-to-collection 直传，不要让数据绕经共享工作站或 SSH mux。

当前环境事实：工作站已有 `/home/asus/.local/bin/globus`；Torch 登录节点当前没有
`globus` CLI。因此通常应在工作站 CLI 或 Globus Web App 发起传输，HPC 端只做目标
路径和 checksum 审计。

```bash
# 工作站：登录、确认身份、查找 collection
globus login
globus whoami
globus version
globus endpoint search 'COLLECTION_NAME'

# 查看目标路径
globus ls COLLECTION_ID:/path/

# 递归传输；checksum sync 用于可安全恢复的重复提交
globus transfer \
  SOURCE_COLLECTION_ID:/source/path/ \
  DEST_COLLECTION_ID:/destination/path/ \
  --recursive \
  --sync-level checksum \
  --verify-checksum \
  --label 'memnav-dataset-or-overlay'

# 保存 transfer 返回的 task ID 后查询/等待
globus task show TASK_ID
globus task wait TASK_ID
```

禁止在没有明确授权时使用 `--delete-destination-extra`。Globus 显示 succeeded 只说明
传输任务完成，不替代项目自己的 SHA-256 与数据合约检查。

每次大文件传输必须记录：

- source/destination collection ID 与路径；
- Globus task ID；
- 预期文件数和总 bytes；
- destination 上的 checksum manifest identity；
- 数据集许可/TOS 已满足；
- 核验后是否设置为只读。

## 5. 不可变 bundle 与 checksum

### 5.1 常用核验命令

```bash
# receipt 本身的身份
sha256sum SOURCE_ROOT/SOURCE_BUNDLE.sha256

# 在正确根目录验证 receipt 内全部相对路径
(cd SOURCE_ROOT && sha256sum -c SOURCE_BUNDLE.sha256)

# 检查是否仍有可写文件
find SOURCE_ROOT -type f -perm /222 -print

# 嵌套部署 receipt 也必须在 relocation 后独立执行
(cd SOURCE_ROOT/pi3x_deployment && sha256sum -c OUTPUTS.sha256)
```

外层 `SOURCE_BUNDLE.sha256` 只能证明嵌套 receipt 的字节没变，不能证明嵌套 receipt
搬家后仍可执行。Pi3X Attempt 4 就因为 `OUTPUTS.sha256` 写入 5090 生产机绝对路径而在
HPC relocation 后失败；后续 receipt 必须只含相对路径，并运行 portable verifier。

### 5.2 打包闭包

- `required=(...)` 不能只列入口脚本；必须递归检查 repository-local imports。
- 提交脚本、子进程 launcher、环境脚本和配置也必须进入 bundle 身份；只改入口脚本
  却不更新 receipt，会让实际执行与记录的版本脱节。
- Final14 Attempt 1 漏掉冻结的 `strict_graph_blind_20260806.json`。
- Attempt 3 虽包含 `materialize_paper_online_a_scene.py`，却漏掉其本地依赖
  `materialize_online_a_traces.py`。
- 远端 exact container import smoke 必须发生在提交科学任务之前。
- 新 run root 必须不存在；旧 attempt 不复用、不改写。

### 5.3 只读 bundle 与 Python bytecode

对只读 bundle 直接运行 `py_compile` 会试图创建 `__pycache__`，Replica/GOAT 均因此在
零 episode 阶段失败。可选做法：

```bash
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache_${SLURM_JOB_ID}"
python -m py_compile PATH/TO/MODULE.py
```

或对严格只读 preflight 使用 AST syntax audit，不产生 bytecode。禁止为了让
`py_compile` 通过而临时把冻结 bundle 改回可写。

### 5.4 依赖先构建，再冻结，再到目标环境验证

**先确认每个进程实际使用什么。** Habitat、模型服务和汇总工具可能使用不同 Python。
记录各自的绝对解释器路径、`sys.executable`、关键包版本与 `module.__file__`，并核对
容器、CUDA/PyTorch、NumPy 和本地扩展。包名相同、import 成功，不代表加载了正确副本。
9 月 17 日的[整合验证](../docs/VALIDATION_20260917.md)就发现另一份 LingBot checkout
的 attention 实现与冻结版本不同；应匹配原依赖，不能放宽哈希检查。

**在正式数组前完成安装与构建。** 缺少的依赖放入任务专用环境、overlay 或依赖目录，
不临时升级多人共用的 conda 环境。权重、配置、tokenizer 等资源提前准备并核验；
会延迟编译的 CUDA 扩展，要在匹配目标 GPU/CUDA 的构建或验证作业中执行一次真实调用。
记录构建命令与产物版本，避免每个数组成员临时下载或同时写同一可变缓存。
登录节点只做站点允许的轻量准备，GPU 编译和测试使用调度资源。

**打包包含运行全程需要的内容。** 不仅检查服务启动，还检查目标切换、图像审计、
结果序列化、压缩与汇总涉及的依赖。
[Replica 缺包记录](REPLICA_ATTEMPT2_DEPENDENCY_INCIDENT_20260814.json)中，漏打包
`depth_anything` 导致历史生成后才发现模型无法启动；§13 的 OpenCV 事故则来自
Habitat Python 与模型 Python 的环境差异。不要用另一个环境的整份 `site-packages`
补洞。完成构建后生成 checksum，传输到独立目录，在最终路径重新核验，再冻结供作业读取。

**用目标运行栈检查真实输入与输出。** 所有计划输入先检查路径、读取和协议条件，
再在正式 container/overlay/解释器中做实际模型前向、渲染与请求往返。
CPU 预检应覆盖真实图像解析和输出字段；GPU gate 再把全部分支跑到独立 verifier。
[multipart 事故](TABLE1_MULTIPART_REPAIR_STATUS_20260911.md)只在特定 JPEG 分块边界
触发，而[收据事故](HPC_HARDENING_20260821.md)是在运行完成后才发现关键字段没有写入。
因此 ready、退出码为零与结果可复核是三个分别需要检查的条件。

## 6. Singularity、EGL 与数据 overlay

Habitat client 与渲染必须在已验证的 container + `--nv` + read-only overlay 中运行：

```bash
singularity exec --nv \
  --overlay /scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf:ro \
  -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  COMMAND ...
```

提交前不仅检查 overlay 文件存在，还要检查：

- `stat` bytes 与冻结值一致；
- container 内 episode root 完全一致；
- exact manifest 引用的 parquet 能在该 container/overlay 中逐个打开；
- collection、construction 和 evaluation 三阶段使用同一挂载合约。

已发生过两类真实错误：

1. Final14 Attempt 2 使用 host 上只有 37 scenes、零 Final14 episode 的 extracted root，
   没有挂 PT1 SquashFS；
2. Attempt 5 的 collection/manifest 已正确使用 overlay，但 evaluation SBATCH 漏掉
   `--overlay`，第一 arm 在读取 parquet 前失败。

GOAT EGL 事故的根因也不是 GPU family：wrapper 覆盖 `LD_LIBRARY_PATH`，把 Singularity
`--nv` 注入的 `/.singularity.d/libs/libEGL.so.1` 换成普通容器 EGL。不要整体替换
`LD_LIBRARY_PATH`；必须保留 `--nv` 注入路径，并在 exact runtime 内做 EGL smoke。

## 7. Server、端口和 CUDA 生命周期

### 7.1 端口

“先探测空闲端口，加载模型后再 bind”存在 TOCTOU。Paper Attempt 6 中端口 `39788`
在这段窗口被占用，出现 `Address already in use`。

正式任务应使用 `retrying_server_launcher.py`：

- 在受限端口范围内重试；
- 写出实际端口文件和 launcher receipt；
- readiness 成功后才启动 evaluator；
- `trap` 中只清理本任务记录的 PID；
- 不因端口冲突修改方法或数据。

### 7.2 CUDA HTTP handoff

Pi3X/Habitat 共驻时曾出现：bare renderer 约 `5--12 ms`，但完整异步 lifecycle 中 render
约 `19.18 s`。模型常驻和显存容量测试都正常；强同步后恢复到约 `5.9 ms`。根因是
Flask response 返回时 CUDA work 仍在队列中，随后 Habitat EGL/CUDA 被迫承担等待。

当前 formal server 使用：

```text
--synchronize_cuda_http_handoff
```

它在 HTTP response 交接前调用 `torch.cuda.synchronize(agent.device)`。这属于运行时
正确性修复，不是 Pi3X 方法增益。排查类似问题时必须分别测：

1. bare Habitat renderer；
2. 模型仅常驻、无请求；
3. 完整 memory/plan lifecycle；
4. 强同步对照。

不能仅看到“Pi3X 加载后变慢”就得出显存不足或模型几何失败。

## 8. 开发中已遇到的问题总表

| 现象 | 已确认原因 | 正确处理 | 禁止的误判/操作 |
|---|---|---|---|
| no-PTY SSH 卡住 | mux command channel 问题 | 先验证默认 socket 与显式 PTY | 直接宣布 MFA/HPC 失效 |
| `scp -S /path/to/socket` 报 `Permission denied` | `scp -S` 把路径当作连接程序执行 | 使用 `-o ControlPath=/path/to/socket`；若 jobs 已提交，只补传同一 receipt | 因 receipt 上传失败重复提交科学 DAG |
| 登录到了错误用户 | 复用了 `yz11445` socket | `ssh -G` + `id -un` 双检 | 杀掉别人的 master |
| 一次命令暂时无输出 | checksum/SSH 命令仍在运行，调用方返回了 session ID | 继续 poll 同一 session，并先查 `squeue`/`sacct` | 立即重复提交；曾因此生成等价 `15892898`/`15892919`，后者已在启动前取消 |
| `Invalid job id specified` | job 已离开 active queue | 用 `sacct` 查历史 | 说 Slurm controller down |
| 长时间 pending | `Priority`/QOS/user GPU limit | 报告 exact reason 与 start estimate | 当成 evaluator 卡死 |
| 18 分钟任务申请 8 小时 | 沿用保守模板 | gate 后按实测改为 1 小时 | 用 requested time 推算真实 compute |
| `scontrol ... TimeLimit` 报 `Unspecified error` | 集群拒绝用户态原地修改该 array | 仅取消 pending，exact resubmit | 取消 completed/running 或重开 population |
| server `Address already in use` | port precheck/bind TOCTOU | retrying launcher + receipt | 改方法或忽略失败 task |
| source episode 为零 | host root 与 overlay root 混淆 | exact container + overlay preflight | 替换 scene 或放宽冻结条件 |
| evaluation 找不到 parquet | eval SBATCH 漏挂 overlay | 三阶段统一 mount contract | 认定数据不存在 |
| bundle import 缺文件 | 只打包入口，未递归闭包 | import closure audit + 新 immutable bundle | 原地补文件到冻结 bundle |
| nested checksum 搬家失败 | receipt 含生产机绝对路径 | relative receipt + relocated verification | 只信外层 bundle hash |
| empty scene builder 报错 | builder 错把合法零 history 当异常 | 输出 audited empty fragment | 借用别的 scene episode |
| readonly bundle 语法检查失败 | `py_compile` 写 `__pycache__` | 临时 pycache root 或 AST audit | 把 bundle chmod 回可写 |
| EGL context 创建失败 | 覆盖 `LD_LIBRARY_PATH` 丢失 `--nv` NVIDIA EGL | 保留注入库路径，exact EGL smoke | 归因于某个 GPU family |
| renderer 从毫秒变 19 秒 | 异步 CUDA HTTP handoff | response 前显式 synchronize | 归因于 Pi3X 能力或显存不足 |
| Habitat `core dumped` 且 depth cast warning | 原生 simulator/runtime abort；该次无完整 arm summary | 保留 partial/log/buffer，exact-index retry | 把 partial outcome 计入 SR |
| summary 因上游失败被取消 | `afterok` 正常 fail-closed | 修复 exact failed task 后重建 dependency | 绕过依赖直接汇总不完整分母 |
| summary 假设每 scene 2 条而实际 4 条 | hard-coded population size | 从 frozen manifest 读取 `episodes_per_scene` | 手工删记录迎合旧代码 |
| diagnostic 缺 `requests` 等包 | 使用了错误 interpreter/环境 | 绝对 interpreter 路径 + dependency import smoke | 临时在正式环境无记录安装 |
| `config.json not found` 等 warning | 不一定是 fatal；Pi3X server 后续可正常 ready | 结合 exit code、ready receipt 和 arm completion 判断 | 看到 warning 就宣布方法失败 |

## 9. 标准的提交前与结果后检查表

### 9.1 提交前

- [ ] `ssh -G`、`ssh -O check`、`id -un`、`hostname` 已核验；
- [ ] source bundle 与所有 nested receipts 在目标路径通过 SHA-256；
- [ ] 完整依赖与构建产物已准备，实际解释器和包路径正确；shell syntax、CPU 预检已通过；
- [ ] exact container、`--nv`、overlay、bind mounts 和 parquet 可读性已通过；
- [ ] partition/account/QOS/GPU/CPU/memory/time/array cap 已逐项确认；
- [ ] 完整配对 gate 连同归档、独立校验已通过；耗时及 RAM/VRAM 峰值已记录；
- [ ] 长历史、目标切换与多臂常驻需求已覆盖；新运行栈的小批量运行无新增故障；
- [ ] 用户字节/文件数配额、临时空间和计算节点输出可写性已检查；时限包含收尾余量；
- [ ] run root 不存在，输出路径不会覆盖旧 attempt；
- [ ] summary 和 independent verifier 使用 `afterok` 串联；
- [ ] 大文件已决定走 Globus，小 bundle 才走 SSH copy。

### 9.2 运行中

- [ ] 同时报 `completed / running / pending / failed` 数量；
- [ ] pending 报 exact Slurm reason；
- [ ] 记录节点与 GPU identity；
- [ ] 查看当前 arm，但不根据 partial outcomes 调方法或筛 population；
- [ ] 查看动作/请求进度、最新落盘时间和资源增长，不能只凭 GPU 利用率判断健康；
- [ ] 超过 60 秒的命令继续 poll 原 session，不重复提交；
- [ ] 发现 partial failure 时先保全现场。

### 9.3 完成后

- [ ] 所有预期 `completion.json` 数量与 manifest 一致；
- [ ] 每个 completion SHA 与 `.sha256` 一致；
- [ ] array 所有 scientific elements exit `0:0`；
- [ ] summary 成功且分母、paired fields、cluster 统计正确；
- [ ] independent verifier 返回 `verified=true`；
- [ ] 从持久存储重新读取并校验归档后，才清理本作业的临时数据；
- [ ] runtime incidents 单独记录，不混入方法结果；
- [ ] 最后才报告 aggregate SR、paired gain/loss、McNemar 和 cluster CI。

### 9.4 给收尾留时间，中断后保留可补跑的记录

申请的 walltime 应覆盖 **准备 + 运行 + 校验/归档 + 余量**。应用层截止时间要早于
Slurm 硬上限，并预留实测的收尾时间。例如现有
[Table I wrapper](run_table1_repaired_hpc.sh)在一小时作业内为 evaluator 设置
48 分钟超时，随后归档并保留原退出码；前面的准备也占总时限，不能据此假定总有
12 分钟可收尾。更长历史、更大输出和更慢 GPU 都要重新估算。

每项任务使用独立的临时目录、端口和输出目录。启动 Python 前设置可写且有空间的
`TMPDIR`、`LIBFFI_TMPDIR` 与 bytecode/cache 路径。检查用户配额和文件数，也检查
节点临时盘；§12 的 `MemoryError` 实际来自满的 `/tmp`，扩大 RAM 申请没有帮助。
归档目的地要从实际计算节点验证；§15 中 `/archive` 只能由登录节点写入，计算作业
须先落到可写的持久 scratch，再完成归档。

新 runner 的落盘策略应先经过小规模验证：完整配对完成后及时保存输出与哈希，
确认必需文件齐全后才发布完成标记。失败路径也保存日志、配置、进度和非零退出码；
节点临时盘不是备份。大数组尤其要避免“算完全部任务才第一次复制结果”。
已有 runner 的实现不完全相同：
[bearing wrapper](run_gem_reviewer_paper_controls_hpc.sh)直接调用 evaluator，
没有 Table I wrapper 的外层超时与归档步骤，使用前应核对整条调用链的收尾行为。

为新的收尾逻辑安排一个隔离工程测试，验证子进程失败或提前退出时能否保存现场、
释放本任务服务并正确失败；不要用正式运行测试取消。强制终止不一定会执行清理，
因此定期落盘与提前截止都需要，不能只依赖退出时的 `trap`。

出现故障先看 Slurm 状态、最后进度和日志，再决定补跑范围。`SIGABRT`、OOM、
超时、输入错误和校验失败分别处理；完整配对未结束就不发布结果。
[连续运行故障记录](REPAIRED_HM3D_CONSTRUCTION_RUNTIME_AUDIT_20260909.md)说明，
import、短渲染与模型 ready 全部通过，也不能排除几十个动作后的故障。
先在原配置下定位或验证兼容运行栈，再按 §2.5 补跑；增加时限不能修复崩溃，
换 GPU 也不能修复请求解析错误。

## 10. 本项目的标准路径

- immutable bundles：
  `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/`；
- run roots：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/<experiment>/<RUN_TAG>`；
- Slurm logs：
  `/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/`；
- Habitat interpreter：
  `/scratch/lg154/conda-envs/habitat/bin/python`；
- model/server interpreter：
  `/scratch/lg154/conda-envs/memnav/bin/python`；
- standard Singularity image：
  `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`。

HPC 上不要 `conda activate`，统一使用绝对 interpreter 路径。状态改变命令优先通过
已验证的共享 PTY；read-only 查询可通过正常工作的 no-PTY channel。本文应作为以后
所有 HPC 工作的强制 preflight 与故障分类入口。

## 11. 用户文件数配额与全盘空间（2026-09-06 核验）

Torch 上先运行站点命令 `myquota`。`quota -s` 可能因 NFS 权限返回
`Operation not permitted`；这不表示无法检查配额，也不代表配额无限。

本轮 yz11502 实测：scratch 1.37TB/5TB（27.39%），文件数
4,897,908/5,000,000（站点显示97%）；与此同时 `df -h /scratch` 显示全盘仍有603TB。
所以 `Disk quota exceeded` 的检查必须同时看 **用户字节配额和文件数配额**，
不能拿 `df` 的全盘剩余容量排除配额问题。以上只是2026-09-06快照，不是长期常数。

逐帧JPEG、重复runtime、源代码/依赖快照都应估算文件数预算。只对已确认的实验目录
做 `du --inodes -s EXACT_RUN_ROOT`；该命令包括目录，链接复用的run也不代表被引用
源目录的全部占用。避免无目的地递归扫描整个scratch。

准备新的长任务时，优先评估节点临时空间录制、结束后按query/arm归档持久化的可行性，
并保留内部文件SHA与可恢复索引；这是待验证的存储方案，不可未经verifier兼容检查
直接改变冻结产物布局。不要自动删除旧失败/成功记录来换空间；先确定恢复与保留范围。

## 12. 2026-09-08：`MemoryError` 也可能来自满的临时文件系统

在 `torch-login-a-0`，Habitat Python 3.9 导入 `quaternion → numba → llvmlite`，
在创建 `_ObjectCacheNotifyFunc` 时出现 `MemoryError`，host 和标准 container 都复现。
独立的 `ctypes.CFUNCTYPE(None)(lambda: None)` 已能复现，模型和 GPU 尚未加载。

同一时刻：`free -h` 显示约 83 GiB 可用，`ulimit -v` 为 unlimited，
但 `df -h /tmp` 为 2.0G/2.0G、100%。为本任务创建独立临时目录，并仅对子进程设置
`TMPDIR` 和 `LIBFFI_TMPDIR` 后，同一个解释器的 callback、quaternion、Habitat 导入都通过。
不能把这个异常称为模型显存不足，也不需要重装共享 conda。

诊断顺序：

1. 检查 `free -h`、`ulimit -v`、`df -h /tmp /dev/shm`。
2. 用实际解释器执行无模型的最小 callback 测试，再导入 Habitat。
3. 登录节点的少量预检可以在本任务精确 scratch 目录下 `mktemp -d`，
   将上述两个变量传给子进程；不要清理他人 `/tmp` 文件。
4. GPU 作业在任何 Python/FFmpeg 预检之前创建自己的 node-local 临时目录，
   设置 `TMPDIR`/`LIBFFI_TMPDIR`；逐帧 buffer 仍放节点临时盘，不迁回高文件数的 scratch。

这条只说明已验证的基础设施案例，不表示所有 `MemoryError` 都由 `/tmp` 占满造成。

## 13. 2026-09-08：新图像审计不能假设 Habitat 已安装 OpenCV

旧 evaluator 主要用 PIL/ImageIO；本轮深度栅格审计和视频入口用到 `cv2`。
实际 Habitat Python 3.9 没有 OpenCV，MemNav Python 3.10 有已运行的 headless 4.9.0。
因此旧任务完成不代表新增图像入口的依赖已经齐全。

本轮只把现有 `cv2` 及 `opencv_python_headless.libs` 复制到任务专属只读依赖目录，
绑定 SHA，并在原 Habitat Python 3.9 / NumPy 1.26.4 下验证 abi3 导入和 resize。
没有安装、升级或修改任何共享 conda 环境，也没有把另一解释器的整个 site-packages
混入 Habitat。新依赖由 `REPAIRED_HAB_EXTRA` 显式传递给所有 Habitat 子进程。

依赖路径：`/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_hab_cv2_20260908`。
该目录不通用于任意 Python/NumPy 版本；其他组合仍需实际解释器验证。

## 14. 2026-09-09：连续运行故障不能用 import / 短渲染预检排除

修复版 HM3D `17196996_1` 通过全部导入、CLI、双场景渲染与模型启动后，
仍在 H100 gh008 出现约 19 秒 memory HTTP 完成间隔，56 步后 evaluator SIGABRT。
同一 bundle 的 shard 0 在 H100 gh003 正常完成，不能据此把所有 H100 归为故障节点。

本项目已有相似故障转 A100 补齐的实际记账：`17089916` 九项和 `17089989_16`
全部 COMPLETED。本次 exact shard retry `17201733_1` 因此显式使用
`safe_sbatch --partition=a100_tandon --array=1`，原代码/参数不变，仅增加
`PYTHONFAULTHANDLER=1`。不要只复用通用分区默认值而遗漏已有事故的运行规避记录。
也不要把该规避直接称为根因修复；状态以新终局和 verifier 为准。

若需进一步定位，记录连续 render、memory HTTP、planning HTTP 分项时间及 abort 堆栈；
HTTP 日志间隔不等于模型 forward 延迟。短渲染成功或依赖预检通过不等于 GPU/渲染
持续交接已稳定。仅补失败项，保留原失败输出，不靠加大时限或改导航参数掩盖异常。

本次 A100 重试最终 ga015 / 4:59 / COMPLETED，两条 A 均成功，verifier=true，
不再有持续 19 秒间隔。记录为本次运行规避成功；H100 根因仍未经过同卡受控验证。

## 15. 2026-09-09：大量 runtime buffer 的可恢复清理

scratch 达到 4,909,485/5,000,000 文件，而字节仅约 1.38/5 TB。只处理已解析的旧
runtime buffer，不删除数据集、模型、在线历史、冻结 bundle 或 metrics/trace/verifier。
对每个确切 buffer：归档全部文件及 SHA → 完整回读 → 核对原文件未变化 → 删除散文件。
原位置留下 `buffer.ARCHIVED_20260909.json` 恢复指针。

`/archive/yz11502` 在本次登录节点可写，但 CPU 试任务 `17243007_4` 的访问检查失败，
尚不能假设所有计算节点都挂载/允许访问它。该失败在删除前发生，没有数据损失。
已验证的两步流程是：计算节点先写 scratch `verified_archives/`；登录节点再转存
`/archive/yz11502/nav_runtime_buffers_20260909/`，核对完整包 SHA 后移除 scratch 重复包。
前一步主要释放文件数，后一步释放 scratch 字节；不要把“已压缩”写成“已转存”。

具体清单、恢复办法与作业状态见 `HPC_STORAGE_CLEANUP_20260909.md`。归档不是永久
备份承诺，仍受站点保留政策约束。新 eval 应在 node-local 写逐帧证据并按完整任务归档，
避免再次积累数百万散文件。

## 16. 2026-09-09：safe_sbatch 默认值也可能覆盖脚本分区

本次脚本明写 `#SBATCH --partition=a100_tandon`，但未传 CLI 分区时，
`safe_sbatch` 实际提交为 `a100_tandon,h100_tandon`。test-only 输出某个 A100
预计节点，并不证明允许分区只有 A100。必须提交后读取 `scontrol show job` 的
`Partition=`、`TimeLimit=`、`ReqTRES=` 和依赖。

`17253413/17253414` 在 PENDING、0 秒且无节点分配时发现此问题；原地修改分区
返回 `Unspecified error`。仅撤回这两个未启动数组，显式传入
`--partition=a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48`
及原 GPU/CPU/memory/time 参数后，重交为 `17253441/17253442`。
实际回读仅 A100，源码、目标、seed 不变；撤回作业没有导航输出。

之后不可仅依赖 SBATCH 文件默认值或 test-only 的预计节点判断运行分区。

### 数组并发更新：错误输出之后也要核对实际状态

同日对17240715调整`ArrayTaskThrottle`时，`scontrol`对已展开元素返回
`Unspecified error`，但父数组和待启动范围已实际变为`%2`、随后`%1`；已有运行元素不中断。
因此既不能仅凭错误文本认定“完全没改”，也不能假设“全部成功”：
读取父数组`ArrayTaskThrottle=`、`ArrayTaskId=`以及运行/等待元素再判断。
这与上面的分区修改不同：分区当时实际没有改变。降低并发只是让其他任务有调度机会，
不是GPU预留；临时调整须记录恢复点，不修改冻结目标、种子或方法参数。

## 17. 2026-09-11：已认证 PTY 正常，SCP 新通道仍超时

本次默认 `yz11502` master 正常；显式 PTY 在 `Last login` 后暂时没有提示符。
只向**自己新开的 PTY**发送一次 Ctrl-C 后，正常得到 `id -un=yz11502` 和
`torch-login-a-0`，`squeue`、`myquota` 都可用。不关闭 master，不切到其他账户 socket。
随后两次有时限的 SCP 均超时，远端目标文件不存在。不能由此宣布 HPC/MFA 失效。

对本次 3.65 MB 源码包和小型 plan，使用相同 master 的 localhost-only **反向转发**：

1. 工作站只读 HTTP server 绑定 `127.0.0.1:46150`，根目录限定为本次打包目录，
   不使用 Research、用户目录或含凭据的目录。
2. 在共享 master 添加 `-O forward -R 127.0.0.1:46151:127.0.0.1:46150`。
3. 已核实身份的 HPC PTY 用 `curl --fail --max-time 45 http://127.0.0.1:46151/精确文件名`
   下载到事先核实不存在的新目标。两端的包/plan SHA-256 完全一致后才解包预检。
4. 结束后通过相同 master `-O cancel -R 127.0.0.1:46151:127.0.0.1:46150` 撤销转发，
   并停止自己启动、记录了精确 PID 的本地 HTTP server。不得停止其他服务或共享 master。

这与第 1.4 节的只读下载方向相反，但仍只有 localhost 可访问，流量经过已有 SSH。
只用于小源码/收据的传输恢复，不替代大型数据集和模型的 Globus 路径。

## 18. 2026-09-11：JPEG SHA 不一致也可能来自 multipart 分块解析

Table I 两个任务换 A100 节点后仍在相同位置失败。通过原归档复现发现：
请求体长度均为 65,539 bytes，Werkzeug 3.1.8 的 multipart 解析会在 64 KiB
分块附近把分隔符 CR 加到 JPEG 尾部。158 的异常 SHA 精确等于“原目标 JPEG + CR”，
110 的历史 JPEG 则直接保存了额外 CR。两个案例解码 RGB 不变，但字节检查正确拒绝。

处理方式：本机 CPU 构造边界复现 → HPC 原输入复现错误 SHA → 进程内保留完整 CRLF
解析边界 → 精确字节复验 → 只补原失败配对单元。不是关闭 SHA 校验、strip 文件，
或已经推进模型状态后重复 POST。无需安装 conda 依赖或改共享包。

修复入口：`multipart_crlf_repair.py`；实际归档复现：`audit_table1_multipart_repair.py`；
状态：`TABLE1_MULTIPART_REPAIR_STATUS_20260911.md`。新的预检除 import 和 CLI 外，
应包含跨 64 KiB 的单图/双图上传；普通小图 smoke 不覆盖这一故障。

## 19. 2026-09-11：动态后处理依赖与已结束作业

本次只读查询 `scontrol show config` 得到 **`MinJobAge = 64 sec`**。
构造、汇总这类短 CPU 作业可能已经正常完成，但后续 CPU 调度器尚未获得资源；
等它启动时，上游编号可能已不在 `squeue/scontrol` 的活跃记录中。

动态衔接 B/C 和最终复算时：

- 先用 `sacct -X -nP -j JOB_ID --format=State,ExitCode` 核实真实状态；
- 上游仍 pending/running 时，继续使用 `afterok` 依赖；
- 上游已 `COMPLETED / 0:0` 时，无需再等待已完成事件，可在核对所属 plan/阶段后提交后处理；
- 失败、取消或查不到可靠状态时，不推断为成功，也不放行最终报告。

该处理仅作用于后处理调度，不重跑导航，不改 frozen manifest 或补造缺失结果。
实现：`table2_final_postprocess.py`；后处理只申请 CPU，并从原始归档独立复算 SR/SPL。

## 20. 2026-09-11：NavMesh 重载造成离线碰撞复算不一致

Table II 的 `17357681_21` 已完成导航，但 verifier 复放一帧碰撞时相差约 2.4 cm。
本机与 HPC CPU 复核确认：同一份逐字节一致的 NavMesh，现场重建时有 2 个连通区域，
保存后重新加载有 3 个。以原始 GLB/设置重建后，164 个执行落点全部精确复现。

遇到这种日志时，先核对控制请求与碰撞环境，不要直接重跑 GPU、改成功标签或放宽容差。
允许的补充复核必须证明重建网格与原归档逐字节一致，并通过完整轨迹/深度/目标审计；
原失败归档移至明确的 `failed_attempts` 目录保留。它不改变导航 controller 或 physics 合约。

本次 CPU 作业 `17365241` 用时 22 秒、`COMPLETED / 0:0`；另一次独立原始归档复算通过。
修复说明：`TABLE2_COLLISION_REVERIFICATION_20260911.md`。
代码：`reverify_table2_collision_task.py`、`slurm_table2_collision_reverify.sbatch`。

## 21. 2026-09-11：调整现有数组并发，不重提样本

Table II A 数组的资源等待除了分区 `QOSGrpGRES`，也出现自身的 `JobArrayTaskLimit`。
本次先实查 `gpu48` 单用户上限为 16 GPU、`a100_tandon` 分区 QOS 总上限为 60 GPU，
再对已确认属于本任务的数组 `17357681` 使用：

```bash
scontrol update JobId=17357681 ArrayTaskThrottle=4
scontrol show job 17357681
squeue -u yz11502
```

原并发 2 改为 4，随后确实看到四个元素同时运行。这个操作不新增数组，不重跑样本；
每项 A100 / 1 GPU / 1 h、样本、seed 和同机配对规则均保持不变。
原协议若写明并发数量，必须单独记录资源调度补充，不能悄悄改写旧协议或提交收据。

此次更新对已结束成员输出错误提示，但根数组读取到 `ArrayTaskThrottle=4`，
随后 `%4` 及四项 running 证实修改已生效。以实际状态核验，不能仅凭提示重复提交。
并发上限提高不保证 QOS 配额立即可用，也不能通过加入未经验证的分区绕过排队。
详见 `TABLE2_RESOURCE_SCHEDULE_ADDENDUM_20260911.md`。

## 22. 2026-09-11：连续双臂的显存不能沿用单臂估计

逐段共同发题要求两臂都保留活跃状态，不能再把“同一张卡依次 reset 跑两臂”的显存
视为本任务需求。本机 Table II 首例 native A 完成后，其 LingBot live allocation 为
14.34 GB；保留它并执行 GEM A 时，约 48 GiB 卡在另一臂深度头申请 66 MiB 时 OOM。
其他工作区当时仍占显存，未停止它们。

私有 `private_cuda_trim` 只在该服务空闲时释放未使用的 allocator 块；原日志中 reserved
从 24.21 GB 降至 14.75 GB，allocated 不变。这不能释放仍有引用的 KV、尺度或历史状态。
因此不能将此 OOM 误判为 conda 依赖错误、靠 reset 活跃 memory 规避，或用单臂成功当作
双臂持续运行的通过证据。已经结束的失败 episode 则可在完整写入收据后释放其私有服务。

对应首例按既有稳定运行记录选 A100 80 GB、1 h，在一个作业/节点/GPU 上完成两臂；
正式规模仍待实测。四个端口使用 `claim_slurm_tcp_port_block ... 4`，不硬占别的端口。

本次 SSH 仍是默认 master 和原已认证 PTY；SCP 新通道再次超时，3.73 MB 源码包通过
仅 localhost 的反向转发完成传输并核对 SHA。临时 HTTP server 和转发已立即关闭。
若新 PTY 需 Ctrl-C 恢复提示符，应先单独发送 Ctrl-C、等提示符，再发送 `id -un`；
不要把控制字符和身份命令连在一次输入里，以免命令首字符被吞掉。
