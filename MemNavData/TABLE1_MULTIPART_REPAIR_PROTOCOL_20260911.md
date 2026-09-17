# Table I：JPEG multipart 边界修复与两项补跑

## 故障与证据

原数组 210 个 controller/history 单元中，208 项完整成功；110（NavDP）和
158（ViNT）在 Revisit-native 最后一个臂中止。原样补跑在不同节点复现。
不是缺依赖、超时或已证明的 GPU 故障。

- 110：服务端历史 `ep_0004/348.jpg` 为 65,191 bytes，结尾为 `ff d9 0d`。
- 158：冻结目标为 49,039 bytes；异常 SHA 精确等于原 JPEG 加一个 `\r` 的 SHA。
- Werkzeug 3.1.8 的 multipart 分块解析可独立复现该错误：在分隔符跨 64 KiB
  分块时，partial-boundary 搜索保留 LF 却提前提交 CR，导致文件多出一个字节。
- 相同请求、相同 JPEG 的 CPU 回放用于核验错误 SHA 是否与失败归档一致；不需要
  模型推理，不用结果选择修改。

## 最小修复

`multipart_crlf_repair.py` 仅在三个私有评测服务入口安装进程内解析修复：
partial-boundary 索引落到 CRLF 中的 LF 时，连同前一个 CR 一起保留。

不改共享 conda 环境；不升级依赖；不删除 JPEG 字节；不修改 SHA 校验；不以解码
像素近似相同代替字节一致性；不自动重试已推进模型状态的单次 HTTP POST。

只变更三个 wrapper 的启动挂接，原冻结 bundle 的其余已有文件逐字节保留。
新增修复、测试、归档复现和提交工具另行记录 SHA。checkpoint、GEM 证书参数、
bounded controller、后向转身、600 步、horizon 8、1 m 评分、目标、种子和臂顺序不变。

## 补跑与统计

110 和 158 各重跑完整四臂。保留原始两次失败归档，以及其余 208 项成功结果；
新输出写到独立 `multipart_retry_20260911_*` 目录。只有 210 项都通过原 independent
verifier 和完整归档校验后，原汇总器才输出完整 SR/SPL。不把失败单元记作 SR=0。

已完成样本未因本次修复更改结果。当前两项故障的像素未改变，但这不能替代所有旧
请求的逐字节审计；本次证据支持“修复传输并完成缺失单元”，不声称重跑了全部 840 条。

资源沿用手册：A100，单项 1 GPU / 10 CPU / 96 GB / 1 h，最多并行 2 项；
CPU 汇总 2 CPU / 8 GB / 30 min。正式提交前须在 HPC 实际 Python 环境复现两个故障
并核验修复后的精确字节；作业 ID、最终 source SHA 与结果另记状态文档。
