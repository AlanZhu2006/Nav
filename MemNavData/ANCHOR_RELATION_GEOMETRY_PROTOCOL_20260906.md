# 固定 anchor 定位：真实历史几何输入对照

日期：2026-09-06。此文件在几何条件训练前写入。生产 CEC、论文、真机不改变。

## 问题与范围

前一版为纯视觉位置 probe：103 train pairs / 32 scenes，20 validation pairs / 8 scenes，
训练均值误差 0.089 m，验证 1.305 m。它没有得到 LingBot 历史深度，不能据此断言
学习不如几何定位。新对照只检验真实历史 XYZ 是否有助于同一小读出跨场景定位。

仍使用原去重 123 pairs、相同 anchor 和 32/8 场景划分；这是原 train40 内开发，
不是 fresh confirmation。不增加 Novel/Revisit 标签输入，不学习检索、proof 或控制。

## 特征来源

1. 原 PT1 expert RGB 历史，不是 actual-online NavDP rollout。每一 anchor 的深度从
   0..anchor 完整 RGB 前缀重放：首 8 帧一个初始化块，其后逐帧、window=32、
   frozen LingBot long 权重、BF16 aggregator、原 FP32 depth head。
2. 沿用旧 full-replay 几何读出的因果深度语义；不以压缩 KV + 短窗口假定等价。
   同一历史多个 anchor 一次顺序提取，在该帧完成时立即物化；不以更晚帧更新早期深度。
3. K 从同一原预计算历史相机的预测 FoV 解码，不使用 GT 内参或 GT pose 生成 XYZ。
4. 米制换算复用已审计、目标未参与的前 64 帧相机高度尺度。所有 query decision >=90，
   故该尺度当时已可得。它是旧 train40 诊断合约，不冒充生产 first40。
5. 只传输所需 RGB 前缀、少量历史 pose 预测和尺度收据；不搬运全部 PT1 或多 GB KV。
   GT parquet/gen_meta 独立放在 supervision_only，仅用于独立核对旧训练标签。

## 空间与尺度

冻结 DINO 特征仍为 pad-to-518、37×37 patches 后平均池化到 8×8。历史 XYZ 首先按
实际 pad 像素坐标提升到相机 [right, down, forward]，在 14×14 patch 内汇总，再按
同一 37→8 adaptive pooling 支持域汇总；不把 padding 当真实几何。

保存 valid fraction 与 mask，主要由真实图像区域和有限正深度定义；8×8 单元中有效
区域至少占 0.5 才作为读出 token。两臂使用同一 mask，不将 mask 差异当几何增益。

令 d_a 为 anchor 有效图像区域的 median raw depth，s_h 为该历史前 64 帧的 m/raw
尺度，则 X_normalized=X_raw/d_a，输出单位对应 L_a=s_h*d_a 米。两臂都预测同一
无量纲 [forward, left] 偏移，然后乘 L_a 与原米制 GT 计算相同 loss。这不直接等价
于上一版米制回归；新有/无 XYZ 才是主要受控比较。

GT 只生成监督/评分，既不作为 forward 输入，也不用于拟合额外尺度。使用已知真实
目标位姿的投影检查属于标签审计，不是可部署定位结果。

## 训练与比较

- 同一小型两层 cross-attention，宽128，4 heads，dropout0.05。
- 两臂共享基础参数初始值、batch 顺序和随机种子；XYZ 臂仅多几何编码层。
- 相同米制 SmoothL1(beta0.2) + 0.1 cosine direction loss，方向项 GT range>=0.25m。
- AdamW lr3e-4、wd0.01、batch16、固定1200steps；种子11/23/37全部报告，不挑最好值。
- 先少量真实 pair 提取/投影测试，再8-pair过拟合检查，随后固定全量比较。
- 参考：anchor-copy、训练均值、旧 frozen GCT / finite-PnP；有限 PnP 缺失单独报告。
- 比较位置误差、scene-macro、<=0.5m计数与方向长尾；这些均不是导航 SR。
- goal 或 XYZ 置换只检查输入依赖，分布改变意味着不能算正式因果泛化证据。

## 去留

若完整提取不能保持历史、尺度和坐标一致，则先修数据而非训练假输入。
若新几何臂仍显著落后参考，不用加长这版训练掩盖差距，优先重新审视表示/几何预训练。
只有多个训练种子、多个验证场景有一致定位改善才考虑训练数据扩充；开放集 proof 和
正式 full-mono 闭环尚属后续。本轮不提交长训/导航评测，不改变任何正式通过决定。
