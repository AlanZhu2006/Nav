# Table II 双角色分支：HPC 小批链路协议

日期：2026-09-11。仅做有限的跨节点链路验证及构造供给诊断，**不是论文正式扩样**。
不修改原 Table II、不提交机器人动作、不训练或改变 NavDP/GEM。

## 固定范围

- 使用既有 HM3D `core_source_plan.json` 的原始顺序：前 4 个场景，每场景前 2 个源载体，共 8 个 A 起点。
- 源载体只提供初始位置与相机参数；A 初始朝向独立采样，目标照片重新按共同规则构造。
- 不读取旧 A/B/C SR 来选源，不使用 expert 轨迹作 memory，也不把旧目标照片作为新 A 目标。
- 不保证 8 个 A 均成功，不保证足够的 C 前缀。源池耗尽后报告缺额，不自动追加场景。
- 主比较仅 native/GEM。A 最多 8 次导航；B 最多 16 个配对查询/32 次导航；
  C 最多 32 个配对查询/64 次导航。实际数量由 A、native-B 成功与构造供给决定。

## 冻结运行栈

沿用本机已完成的 `bounded_standard / rgb_v1 / source_rgb / heading_on`：
冻结 NavDP；单目 LingBot 深度；canonical GEM reference depth；最多 600 ticks；
exec_horizon=8；GT 平面位置 <1 m 计成功；保存最后动作后的坐标用于实际位移 SPL。
这不是官方 MPC。旧 snap 仅作诊断；实际碰撞采用 Habitat try_step。
角色标签不进入 server；没有认证接管时逐动作与 native 对齐。不是自主视觉 STOP 验证。

## 分阶段设计

1. **A 数组**：每任务一个新实际 A；成功后只从实际帧构造 B-Novel/B-Revisit。
2. **B 数组**：每任务一个 B 目标的 native/GEM 两臂，使用同一 GPU 上同一对常驻 server。
   两种 B 从同一个 A 末态分别重放启动，不能共享彼此的新增历史。
   成功 native-B 可生成自己的 A+B 前缀及 C 双角色候选；GEM-B 从不充当前缀采集器。
3. **全局 CPU 选择**：汇总所有 B 数组的合规来源，按原源顺序从两种 B 各取相同数量。
   **不要求同一 A 的两种 native-B 都成功**；任何 C 导航之前写入不可覆盖的 c_population.json。
4. **C 数组**：每任务一个已封存 C 目标的 native/GEM 两臂，原样恢复相应 A、native-B 前缀。
5. **总结**：逐阶段分母、A/B 构造损耗、B→C 来源、支持来源、SR/SPL 与成对增损。
   所有源任务失败均保留。基础设施失败不当成导航失败。按 scene 聚类，不冒充自主 A→B→C joint。

小测沿用本机构造：Novel 完整历史最大共视 <0.10；Revisit 合规历史共视 [0.55,0.90]；
最终目标测地距离 2–9 m；Revisit 是扰动重渲染视图，不是复制 JPEG。
Novel 使用共同的独立目标照片 yaw 规则；Revisit 仍须遵守扰动/共视条件。
目标选择不参考 DINO、certificate 或查询 SR；不强制同一个历史 N/R 同方向。

本轮输出距离、方向、当前可见性与历史供给，用于后续冻结正式总体。
**不声称各阶段、各角色的距离/方向已经匹配；不以角色间原始 SR 差作为因果结论。**

## 任务、存储和依赖

- 唯一身份为 scene/episode；B/C 再含角色、前序 B 类型和 arm。不能只按 scene 分组。
- 统一从绑定的 plan 读取 sources，不调用本机固定两场景 sources()。
- 每任务 host mktemp；在容器内绑定同一实验专属虚拟根目录。
  上游完整归档恢复到原虚拟路径，内部 JSON/RGB/深度/SHA 不改写。
- 只把少量 manifest、summary、verifier 和每任务单个压缩归档留在 scratch。
  A 与成功 native-B 的真实历史必须可恢复；不能只保留 SR。
- 复用既有 container、memnav/habitat 解释器及 cv2 只读依赖；不修改共享 conda。
- GPU 任务显式 A100、1 GPU、10 CPU、96 GB、01:00:00；并发上限 2。
  初始 A index 0 先行，其余 A 依赖接口成功；不根据该 A 是否导航成功决定放行。
  每阶段最多 48 分钟运行，剩余时间用于保全归档。超时保存失败，不更改导航预算。
- 完整 pilot 的实际耗时用于决定下一次正式规模；一小时是每个元素的上限，不是总耗时预测。
- B 依赖全部 A 成功退出，C 来源选择依赖全部 B；无合规 C 时报告 0，禁止拼接旧前缀。
- 独立 verifier、归档回读、精确 CLI/import 与 source/checkpoint 预检必须通过。

## 放行标准

本机 CPU 测试检查重复场景多 episode、两分支隔离、全局 50/50、归档恢复、错误路径拒绝、
完整配对与空前缀。HPC 预检覆盖新 Table II 入口及实际源 parquet/相机参数。
实际 GPU gate 检验完整渲染—模型—物理动作—记录链，不用 SR 设通过阈值。
只有本轮供给和运行审计完成，才另行冻结正式距离/朝向配额及样本规模。
