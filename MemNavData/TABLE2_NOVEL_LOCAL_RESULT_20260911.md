# Table II：Novel 构造与切换的本机检查结果

2026-09-11。检查完成；没有新增导航 SR，没有修改运行中的模型或提交 HPC。

## 1. 本轮完成了什么

1. 新增独立的 Novel 候选模块 `table2_novel_sampling.py`：A/B/C 共用同层、2–9 m、
   0.30 m clearance 和独立目标 yaw 规则。A 起始 yaw 不再来自最短路；
   B/C 保留真实末态朝向。前/侧/后缺额不自动换方向。
2. 目标视图选择只接受完整因果历史的共视曲线；早期不可检索帧也计入，
   不能把「检索池中没有」当作 Novel。具体渲染/共视回调尚未接入正式采集器。
3. 实现 C 前缀的 50/50 选择函数：按既定顺序等额取 native Novel-B / Revisit-B，
   依据 B 成功及 C 可构造性选择，不读取 C 成绩，不拿 GEM-B 替换失败的 native-B。
4. 运行目标切换、NavDP 短期 FIFO、物理朝向适配、修复运行配置和新增构造测试。

最终一次测试：**73 passed，0 failed，0 skipped**。
其中 60 项既有检查、13 项新增检查。没有加载完整模型进行推理。

新增状态测试直接调用当前 `MemNavAgent` 方法，核对：

- 已接受 Revisit 切换到 Novel 后，旧 embedding/anchor/认证/候选缓存被清理；
- RGB 历史对象、LingBot 对象、真实字段 `_first40_scale_receipt` 及历史深度缓存保留；
- 目标切换不推进 NumPy / PyTorch CPU 随机状态；
- 新查询的空合法候选集不复用上一目标的候选。

这是 CPU 状态与接口测试，不是网络精度、CUDA 采样等价或完整闭环验证。
没有证明所有输入回执问题都已解决，也没有改变 NavDP 的默认 FIFO carry 规则。

## 2. 空间供给检查及新发现

本机固定使用已消费的 `gxdoqLR6rwA`、`pLe4wQe7qrG` 两条源任务。
两条 A 初始物理位置均纳入；旧 A 失败只阻止构造其后续实际前缀，不能删掉该 A 起点。

每个起点最多提议 5000 次，每方向保留至多 16 个空间候选：

| 起点配置 | 前向保留 | 侧向保留 | 后向保留 | 说明 |
|---|---:|---:|---:|---|
| gxdoq，A 新独立 yaw | 16 | 16 | 16 | 新 A 尚未执行 |
| pLe，A 新独立 yaw | 12 | 16 | 0 | 新 A 尚未执行 |
| pLe，旧实际 A 末态 → B | 0 | 2 | 16 | 保留实际位置与朝向 |
| pLe，旧 native Novel-B 末态 → C | 16 | 0 | 16 | 保留实际位置与朝向 |

共保留 126 个空间候选。此处没有渲染、没有测视觉支持，**不是 126 个合规 Novel 查询**。
重复运行后，全部起点、候选、计数与排除原因相同；输入文件哈希及当前构造代码绑定通过。

最重要的观察：pLe 的真实 A 末态，5000 次提议中通过空间约束的方向计数为
**前向 0、侧向 2、后向 1107**。表中后向只保留 16，是候选上限，不是供给只有 16。
这只是固定提议预算内的结果，不是证明场景里绝对不存在前向合法目标。

它说明：后续目标偏后向不一定仅来自随机朝向设置，还可能由真实到达位置、
末端朝向及可达路线共同限制。统一照片和 yaw 规则后，也不能保证每个历史都有相同方向供给。
若实现中在前向构造失败后转用后向任务，总体会重新偏向后向；新入口没有这种隐式替换。

不能从这一条末态推断旧总体下降全部由此造成。本轮也没有测试 NavDP 能否解决这些候选。

## 3. C 的 50/50 是否可以在这两个旧本机源上立即凑齐

不能。

- gxdoq 的旧实际 A 失败，因此没有合法后续参考前缀。
- pLe 的旧 native Novel-B 成功，可以检查其 C 空间供给。
- pLe 的旧 native Revisit-B 失败，不能为构造 C 而传送、换目标或改用成功的 GEM-B。
- C 的完整共视条件尚未计算，所以成功 B 数也不等于可构造 C 对数。

这只是两条旧本机源的供给限制，不是新正式总体构造失败。
不要为凑比例把不同来源、不同方法的实际历史混在一起。

## 4. 对“第二次 Novel 不会再下降”的回答

目前仍不能作出这一保证。

- 已发现的目标状态遗留风险在本轮 CPU 测试中未复现。
- 已有新的共同候选入口，但旧 `collect_repaired_fullmono_a.py` 仍使用原目标图载体；
  本轮没有替换它的默认采集规则，也没有执行新的 A。
- 新发现说明任务方向供给存在真实限制，需要跨起点检查，不能为抬 SR 把机器人预先转向目标。
- A 没有前序任务历史，不代表目标在首帧不可见；后续 Novel 的全历史排除可能同时限制起始可见性。
  该差异需要在下一轮实际目标视图构造中测量，本轮没有测得数值，不新增未经验证的过滤条件。

主比较仍应是同一 B/C 查询上的 native 与 GEM；A/B/C 的绝对 SR 不要求相等。
统一采样不能替代原生接口检查，接口通过也不能替代真实导航。

## 5. 下一步

1. 将新目标视图规则接入独立的小测采集入口，保持旧协议和既有结果不变。
2. 用新渲染目标实际执行 mono A，再依据这条新 A 构造 B；不能接用本轮旧 A/B 末态冒充连续结果。
3. 同时记录初始可见支持、方向、距离以及失败类型。先核验请求/切换与构造，再报告小测 SR；
   不把 SR 高作为代码通过门，不因单例失败换目标。
4. 明确共同可构造范围、C 等额供给后才冻结正式 Table II。

本轮 GPU 正由其他工作区的训练使用，因此只执行 CPU PathFinder 和测试；
没有停止训练或真机驻留服务，没有启动新的后台模型任务。
Table I 已记录的两项输入/消费回执异常仍属未关闭事项，不由本轮测试代替。

## 6. 文件与复现

输出根目录：

`/home/asus/Research/Nav-graph-blind/.diagnostics/table2_novel_local_20260911_uUGKl4/`

- `all_contract_tests.xml`：最终 73 项测试。
- `geometry_pilot_v2.json`：当前代码的空间供给、输入哈希、候选及排除账。
- `geometry_pilot.json`：首次空间运行；与 v2 的候选和排除账相同，旧记录保留。
- `existing_contract_tests.xml`、`new_construction_tests.xml`：早期分次检查，最终计数以合并测试为准。

仅复现空间检查（输出必须使用一个不存在的新文件路径）：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  PYTHONPATH=/home/asus/Research/Nav-graph-blind:/home/asus/Research/Nav-graph-blind/MemNavData:/home/asus/miniconda3/envs/habitat/lib/python3.9/site-packages/pip/_vendor \
  /home/asus/miniconda3/envs/habitat/bin/python -m MemNavData.table2_novel_sampling \
  --old-local-root .diagnostics/repaired_fullmono_local_20260908/e2e_v1 \
  --out /absolute/path/to/a/new/geometry_check.json
```

设计：[Table II mixed-role 草案](TABLE2_MIXED_ROLE_REDESIGN_DRAFT_20260911.md)。
范围：[本机预检](TABLE2_NOVEL_LOCAL_PREFLIGHT_20260911.md)。
