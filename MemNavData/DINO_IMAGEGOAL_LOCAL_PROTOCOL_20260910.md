# DINO 历史图直接作为 ImageGoal：本机接口对照

2026-09-10。此协议在新对照运行前固定。不是 RANa 全模型复现，不是新场景确认。
动机来源：RANa 的检索目标图替换方式，https://arxiv.org/html/2504.03524v2 §5.2。

## 1. 问题与范围

检验“把 DINO 检索到的历史 RGB 直接交给冻结 ImageGoal controller”是否已足够，
并与原始目标、raw fixed 几何方向和严格 GEM 比较。只检查接口并取得小样本机制结果；
不凭本机 N=4 替换主方法或宣称超过 RANa。

复用已消费的 history2/vBMLrTe4uLA、history12/SiKqEZx7Ejt，各自 c10_30 和 natural_novel。
2 histories / 2 scenes、2 Revisit + 2 Novel，四臂16次完整导航。
历史是旧版 actual-mono A，query 是修复执行链；不是新版 A+query 端到端结果。
固定原有目标图和评分位置，不因检索结果重新生成目标。

## 2. 四臂

| arm | 策略实际输入 | 记忆干预 |
|---|---|---|
| native | 当前RGB、共享单目深度、原始ImageGoal | 无 |
| dino_imagegoal | 当前RGB、共享单目深度、检索历史ImageGoal | 不加PointGoal、不调用几何认证 |
| raw_fixed | 当前RGB、共享单目深度、原始ImageGoal、既有raw固定2.5m PointGoal | 保留既有raw规则 |
| cec | 当前RGB、共享单目深度、原始ImageGoal、严格认证后的2.5m PointGoal | 默认strict certificate，未改面积条件 |

四查询采用循环平衡的四臂顺序；每臂独立reset并逐字节重放同一actual-A，
同一私有服务/模型/设备，逐规划冻结noise seed。NavDP和LingBot权重不变。
600 ticks、exec_horizon=8，bounded_standard/rgb_v1/source_rgb/heading_on。
Image-only臂没有PointGoal，因此不会触发显式朝向适配，不能伪称各臂动作完全相同。

## 3. 新臂的因果输入边界

- 首个query决策的native memory append替换为一次既有retrieval_probe_step，**不是额外append**。
- 原始Goal只用于DINO查询，不写入在线LingBot序列。
- 使用CEC的原始DINO shortlist第一项，域为frame>=8到query开始前，top8/gap4。
  不使用LightGlue排序、PnP、覆盖面积、learned gate、相对位姿或GT角色。
- 既有probe内部会计算未使用的旧retrieval/gate字段；它们不参与本臂选择，
  不因此把本臂称为新训练的retriever。
- 选中的原始历史JPEG来自刚刚重放过的causal RGB buffer，核对其SHA，固定到query结束。
- 后续观察正常更新短期FIFO和LingBot；不把新query帧加入本次候选集合。
- 评分始终针对**原始目标位置**，不是检索anchor位置；GT位置不进入策略请求。
- 原始raw fixed保留旧候选域amargin39/exclude_recent32；因此四臂比较整体记忆接口，
  不是声称只改变一个数学变量的bearing因果分解。新image臂的候选域与CEC一致。

## 4. 成功、报告与验证

成功沿用evaluator平面距离<1m；不是自主视觉STOP。SPL按实际每步位移与最终落点重算。
分别报告Novel/Revisit，保留全部16条，不根据中途成功率删任务或追加有利样本。
独立检查首帧、历史、原始目标评分、RGB/深度输入、实际执行和无接管CEC的exact native。
新臂额外核对：一次检索append、固定anchor SHA、每次真正进入NavDP的goal JPEG、
无PointGoal/证书/目标位姿请求。基础设施异常不计作导航失败。

本机使用21910/21911私有服务，其他真机/训练服务不动。运行前显存与磁盘检查；
既有16臂测试约814MB，本次本地结果不复制大模型或场景。低于1GB可用空间停止新rollout，
不删除既有结果。当前不新提交HPC，不修改论文、默认GEM或真机。
