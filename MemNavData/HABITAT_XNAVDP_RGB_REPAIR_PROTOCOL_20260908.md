# X-NavDP RGB 适配修复：本机配对协议（2026-09-08）

## 本次只回答什么

原四历史 X 结果使用了 BGR actor 输入，而发布的 Isaac 客户端—服务端合成链路给 actor 的是 RGB。
先隔离这一可证实的实现错误，不同时换真实目标距离、调 MPC 或增加失败回退。
这是已消费样本的适配诊断，不是新的正式泛化 SR，也不是完整 Isaac 复现。

## 冻结对照

- 源历史、顺序、权重、种子、初始落地与上一轮 `xnavdp_stack_v2` 相同。
- 四历史：gxdoqLR6rwA、pLe4wQe7qrG、yqstnuAEVhm、mJXqzFtmKg4。
- 旧臂 `cec_x_legacy / xmpc`：保留原 BGR 输入，复现被审计的条件。
- 修复臂 `cec_x / xmpc`：恢复 RGB 通道。**当前帧与历史回放一起修复**。
- 偶数历史旧臂先跑，奇数历史修复臂先跑；每臂 reset，使用同一组私有服务。
- CEC、LingBot 接收的原始 JPEG 字节不动，深度 frame/token 绑定不动。
- 相同固定 2.5 m bearing、wheeled posttrain actor、8 candidates、官方 Q/RTC、X MPC。
- 相同 0.376 m/s、π/4 rad/s、0.1 s 命令、8 命令重规划、600 命令预算。
- 不进行 NavMesh 运动投影；倾斜 ≥5° 仍标记代理无效，不能删掉后计算成功率。
- 到达仍只用 evaluator 的 GT 平面距离 <1 m，不代表自主 STOP。
- 保留已有 CEC reject→native 合约；不增加 X 失败→其他 controller 的替换。

## 前置与输出

1. 实际官方客户端函数（拦截 HTTP）与服务端图像解码语句的离线测试通过。
2. 19 项单元测试通过；既有真实 acados 前/后/零参考原语收据保持有效。
3. 修复臂八命令 smoke 通过后，启动四历史配对。
4. 独立复算初始 RGB、深度、证书、实际位移、到达与命令限幅。
5. 新增 actor 通道／张量哈希证据，覆盖 history replay 和 live query；不能只查 wire SHA。
6. 旧臂与 v2 的逐动作一致性另行报告，不能假定跨 reset/CUDA 位级相同。
7. 保留全部失败和无效物理轨迹。不得依据该小样本宣布 X 优于或劣于原 NavDP。

结果目录计划：`.diagnostics/habitat_physics_executor_20260908/xnavdp_rgb_pair_v1`。
之前 v2 结果和录像保持不变；本次不改生产策略、HPC、真机或论文数字。
