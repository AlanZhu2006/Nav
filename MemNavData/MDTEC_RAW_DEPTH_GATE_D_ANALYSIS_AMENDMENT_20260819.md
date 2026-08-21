# Gate D 统计实现补充 — 2026-08-19

状态：**前瞻冻结；正式 Gate D 尚未运行。** 父协议 SHA-256 为
`8deb4e13bd169bb39c7696a777656c3e4912f985f48ee364111b4a9a76cf9413`。

父协议已经冻结 paired McNemar 与 scene-cluster percentile bootstrap，但未写明伪随机
实现参数。本补充只固定实现，不改变人口、三臂、阈值或判据：

- paired unit：`(scene, episode)`；
- cluster unit：scene，抽中一个 scene 时保留其两条 episode；
- percentile bootstrap：100,000 次；
- seed：`2026081901`；
- 区间：2.5% 与 97.5% 分位数；
- primary：`raw_first40 - metric_teacher`；
- secondary：`zero_depth - metric_teacher`、`raw_first40 - zero_depth`。

冻结前仅运行了一条 `max_steps=80` 的 transport smoke，用于确认三臂切换和 receipt wire
contract；该 smoke 不计入任何统计量，也没有用于修改本补充。
