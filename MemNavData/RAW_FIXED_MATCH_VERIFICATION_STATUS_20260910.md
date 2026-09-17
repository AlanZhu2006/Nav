# Raw fixed 独立匹配验证：本机进度

2026-09-10 22:29（Asia/Shanghai）快照。不是最终 MASt3R 结果。

## 已完成

- 按现有共享 SSH 手册核对 `alantorch -> yz11502`，复用原 master。
- 从完整159-query封存三臂总体提取159个 raw 第一决策图像对，不按成败筛选。
- 每对含实际 raw anchor JPEG、原目标 JPEG；对应归档成员与原采集收据哈希校验通过。
- 仅取回约16 MB小包，不取场景/GT深度/整条RGB buffer，不修改远端源数据。
- 本机320个成员回读校验通过；包SHA：
  `c3f6e5ea32d1e0186c832483fad5e349aace1ce0b9ec61e20ce08fb88851edd0`。
- SuperPoint/LightGlue已在全部159个固定 raw 图像对上运行完毕。
- MASt3R官方代码、DUSt3R/CroCo子模块已获取，独立venv导入通过；仅在venv新增roma1.5.4。
- 六项坐标/输入/数值单元测试通过。原GEM、NavDP、真机服务和论文未修改。

## 正在进行

官方 MASt3R checkpoint 下载中，总字节数由源服务器头确认为2,754,910,614。
本机监督命令已经启动：下载完整后先执行前两个固定pair的接口smoke，成功才运行全部159pair。
之后生成 `comparison_v1.json`。任何阶段失败会停止并保留日志，不用别的模型代替。

官方 CUDA RoPE 扩展尚未编译，当前使用其自带 PyTorch 实现；实际延迟须按此实现标注，
不能拿官方完整 SLAM 的FPS作为本项目延迟。首轮不修改推理精度。

目录：
`/home/asus/Research/Nav-graph-blind/.diagnostics/raw_match_verification_20260910_IJGnxv/`

- `lightglue_v1/summary.json`：已完成的匹配证据，不是新的导航结果。
- `mast3r_supervisor_v1.log`：下载完成后的smoke、正式组件读出与错误信息。
- `mast3r_v1/summary.json`：仅在全部MASt3R图像对完成后存在。
- `comparison_v1.json`：完整两种matcher的分层读出，尚未生成。

目前新增导航rollout=0，新SR=无。完整MASt3R-SLAM、raw授权控制器和联合位姿优化均未运行。
没有提交HPC GPU任务，没有commit/push。

## 同轮 Table II 口径核对

当前论文 `tables/continual_meeting.tex` 第一、二行的0.618/0.200是SPL；
SR分别为131/196=66.8%、54/183=29.5%。SPL已离线纠正，SR未变。
这两行仍属旧输入/执行链；没有被新Table-I、共视实验或45-history查询结果替代。
183个B候选覆盖67个成功A历史，并不是196个A任务的一次一对一后续评测。
因此不能仅比较两行得到“第二段造成策略能力退化”或无条件joint SR。
