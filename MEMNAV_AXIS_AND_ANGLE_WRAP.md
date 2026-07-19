# MemNav 坐标轴（axis/extrinsic）与角度 wrap 问题完整说明

本文记录 MemNav 数据生成、NavDP 标签加载和 LingBot revisit pose 之间的两个独立问题：

1. **axis/extrinsic 问题**：生成器把相机安装旋转写进了 `action`，却把
   `camera_extrinsic` 写成单位阵；NavDP 因而把相机光学坐标误当机器人 base 坐标，
   真实 forward 位移落入随后被丢弃的第三个空间坐标。
2. **角度 branch-cut 问题**：`atan2` 输出跨过 `+pi/-pi` 边界时，旧代码直接相减，
   把真实约 `2 deg` 的转动构造成约 `358 deg` 的动作标签。

这两个问题互相独立：axis 修复决定使用哪个平面和哪个方向；angle wrap 修复角度作为
圆周变量时的差值。只修其中一个，另一个仍然存在。

## 0. 代码版本和目录边界

- 原始母目录（只用于对照）：`/home/asus/Research/Nav`
- 本文及修复所在工作树：`/home/asus/Research/Nav-axis-fix`
- 原始对照提交：`8c37868`
- axis 修复提交：`8e235f8` (`fix: align generated MemNav poses with NavDP frame`)
- MemNav action angle-wrap 修复提交：`cbd3433`
- 本文创建时工作树提交：`96fdd2a`

### 0.1 必须先区分“仓库里的问题”和“上游 NavDP 的问题”

本文所说的“原始 Nav”是我们的合并工作树 `/home/asus/Research/Nav`，不等于所有代码都
由上游 InternNav/NavDP 官方编写。Git 历史和官方仓库对照后的准确归因如下：

| 问题 | 问题由谁引入 | 上游状态 | 准确结论 |
|---|---|---|---|
| identity `camera_extrinsic` 导致 forward 落入 z | 我们新增的 `MemNavData/generate_twoleg.py` | 官方 loader 按非 identity mount 的数据契约工作 | **我们的 MemNav 数据接入 bug，不是官方 NavDP loader bug** |
| MemNav 旧 `_R_CONV/_SCALE` | 我们新增的 `memnav_policy.py` | 上游没有这条 MemNav 分支 | **我们针对错误标签做出的下游经验补偿** |
| `atan2` heading 相减不 wrap | 相同模式原本就在上游 NavDP dataset 中；我们的 MemNav override 也沿用了它 | 官方当前 InternNav 仍有 raw subtraction | **上游代码的圆周边界健壮性 bug，在 U-turn 密集的 MemNav 上更容易暴露** |
| dense-KV/no-keyframe | 我们的 LingBot feature precompute 没有遵循当前官方长序列 keyframe 语义 | LingBot 官方对 `>320` views 提供自动 keyframe 策略 | **集成/预计算配置问题，不是 NavDP action loader bug** |

代码来源证据：

- `generate_twoleg.py` 的 Git 历史从 2026-07-07 的我们自己的数据生成提交开始；
- `memnav_dataset_lerobot.py` 和 `memnav_policy.py` 是我们在 2026-06-30 后新增的 MemNav；
- `navdp_lerobot_dataset.py` 的 `relative_pose()`、`xyz_to_xyt()` 和 raw action subtraction
  可在 [InternRobotics/InternNav 官方仓库](https://github.com/InternRobotics/InternNav/blob/main/internnav/dataset/navdp_lerobot_dataset.py)
  中找到。

因此，可以说这些问题“存在于我们当前 Nav 工作树/训练链中”，但不能把 axis 问题描述成
“官方 NavDP loader 把坐标轴写错了”。准确说法是：**我们的 MemNav producer 没有满足
官方 loader 已存在的 extrinsic 契约。**

本文不会把四维齐次坐标中的 `w` 称为“第三维”。下文所有三维平移均为：

```text
[x, y, z]
```

只有写成齐次点时才是：

```text
[x, y, z, w]，点的 w 通常为 1，方向向量的 w 通常为 0。
```

因此，“forward 落入第三维”里的第三维是数组索引 `2` 对应的 **z**，不是 w。

---

## 1. 涉及的坐标系

### 1.1 Habitat 世界坐标

生成器使用 Habitat 的 Y-up 世界坐标：

```text
Habitat world = [x_h, y_h, z_h]
y_h：竖直向上
x_h、z_h：地面平面
```

生成器中的相机采用 OpenGL 风格局部轴：

```text
+x_camera：向右
+y_camera：向上
-z_camera：向前
```

因此零 yaw 时，机器人/相机向前 1 m 的 Habitat 位移是：

```text
d_hab = [0, 0, -1]
```

### 1.2 parquet 中的 Z-up data 世界坐标

文件：[`MemNavData/generate_twoleg.py`](MemNavData/generate_twoleg.py)，`M_W` 定义附近。

```python
# Habitat(Y-up) -> stored data(Z-up) rotation:
# (x,y,z)_hab -> (x,-z,y)_data
M_W = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0],
], float)
```

即：

```text
x_data =  x_hab
y_data = -z_hab
z_data =  y_hab
```

所以零 yaw 时向前 1 m：

```text
d_data = M_W @ [0, 0, -1] = [0, 1, 0]
```

`M_W` 这个世界基变换本身是正确的。错误是 generator 和 loader 对
`camera_extrinsic` 的契约不一致。

### 1.3 NavDP 的局部动作坐标

文件：[`InternNav/internnav/dataset/navdp_dataset_lerobot.py`](InternNav/internnav/dataset/navdp_dataset_lerobot.py)，
函数 `relative_pose()`。

NavDP 在得到 raw local translation 后执行：

```python
T_frame = np.array([T_frame[1], -T_frame[0], T_frame[2]])
```

因此 NavDP 的局部动作含义是：

```text
x_nav =  raw_y   # forward
y_nav = -raw_x   # lateral/left
z_nav =  raw_z   # 非平面维，后续动作标签不使用
```

在当前 LingBot OpenCV-like 局部坐标中，等价的平面映射是：

```text
NavDP planar [forward, left] = [LingBot z, -LingBot x]
```

当前工作树中的显式实现位于
[`InternNav/internnav/model/basemodel/memnav/revisit_pose.py`](InternNav/internnav/model/basemodel/memnav/revisit_pose.py)，
`GaugeInvariantRevisitPose.forward()`：

```python
# LingBot local camera is OpenCV-like (+z forward, +x right).
# NavDP's planar action convention is +x forward, +y left.
planar = torch.stack((t_rel[..., 2], -t_rel[..., 0]), dim=-1)
```

---

## 2. axis/extrinsic 的数据契约

NavDP 期望 `action_R` 是相机的世界旋转，并可分解为：

```text
R_action = R_base @ R_mount
```

其中：

- `R_base`：机器人 base 到世界的旋转；
- `R_mount`：相机相对机器人 base 的固定安装旋转；
- `R_action`：parquet 的 `action` 中保存的 camera-to-world rotation。

因此 loader 用下面的式子恢复机器人 base：

```text
R_base = R_action @ inverse(R_mount)
```

原始 NavDP 代码正是这样做的：

文件：[`InternNav/internnav/dataset/navdp_dataset_lerobot.py`](InternNav/internnav/dataset/navdp_dataset_lerobot.py)，
函数 `relative_pose()`（原母目录约第 260--280 行）。

```python
def relative_pose(self, R_base, T_base, R_world, T_world, base_extrinsic):
    R_base = np.matmul(
        R_base,
        np.linalg.inv(base_extrinsic[0:3, 0:3]),
    )
    ...
```

变量名 `R_base` 在函数入口处容易误导：传入时其实是 parquet `action` 的旋转；执行
`@ inv(base_extrinsic_R)` 后才成为 NavDP 语义下的 base rotation。

---

## 3. 我们的原始 MemNav 接入中 axis bug 的完整代码链

### 3.1 generator 同时做了两件互相矛盾的事

原始文件：`/home/asus/Research/Nav/MemNavData/generate_twoleg.py`

原始 `save_traj()`（约第 357--365 行）：

```python
# poses -> stored Z-up camera-to-world; extrinsic = identity mount
ext = np.eye(4)
rows = []
for i, Tw in enumerate(poses_hab):
    Td = np.eye(4)
    Td[:3, :3] = M_W @ Tw[:3, :3]
    Td[:3, 3] = M_W @ Tw[:3, 3]
    rows.append({
        "index": i,
        "observation.camera_intrinsic": K.astype(np.float32).tolist(),
        "observation.camera_extrinsic": ext.astype(np.float32).tolist(),
        "action": Td.astype(np.float32).tolist(),
    })
```

这里：

```text
R_action = M_W @ R_hab_camera
R_mount_recorded = I
```

注释里的 “identity mount (we bake full pose)” 是我们 MemNav generator 对上游数据契约
的误解。即使完整 camera pose 已写入 `action`，NavDP 仍需要独立的固定
`camera_extrinsic`，因为它要从 camera pose 中恢复 base pose 后再形成平面动作标签。

这不是 `relative_pose()` 本身的公式错误。对本机 109 个原始 InternData-N1 parquet 的
只读检查结果是：

```text
identity mount rotation：0 / 109
非 identity mount rotation：109 / 109
其中与 M_W 数值一致：45 / 109
```

不同官方轨迹可以有不同的真实相机安装矩阵，但都没有假装成 identity；这进一步说明
loader 本来就是依赖真实 mount 工作的。我们的合成数据却把 mount 写成 identity，才破坏
了这个契约。

正确关系应为：

```text
R_mount = M_W
```

对零 Habitat yaw：

```text
R_action = M_W
R_base   = M_W @ inverse(M_W) = I
```

### 3.2 loader 原样读取 identity extrinsic

原始文件：
`/home/asus/Research/Nav/InternNav/internnav/dataset/navdp_dataset_lerobot.py`

函数 `process_data_parquet()`（约第 170--178 行）：

```python
camera_extrinsic = np.vstack(
    np.array(df['observation.camera_extrinsic'].tolist()[0])
).reshape(4, 4)

camera_trajectory = np.array(
    [np.stack(frame) for frame in df['action']],
    dtype=np.float64,
).reshape(-1, 4, 4)

return camera_intrinsic, camera_extrinsic, camera_trajectory, trajectory_length
```

在原始 MemNav loader 中，这个 identity matrix 没有经过任何兼容修正，直接传给
`_build_actions()`。

原始文件：
`/home/asus/Research/Nav/InternNav/internnav/dataset/memnav_dataset_lerobot.py`

原母目录约第 415--439 行：

```python
(
    _camera_intrinsic,
    base_extrinsic,
    extrinsics,
    traj_len_parquet,
) = self.process_data_parquet(ti)

...
seg = extrinsics[k : goal_step + 1].copy()
pred_actions, goal_rel_pose = self._build_actions(
    seg, base_extrinsic, pred_digit
)
```

### 3.3 identity 让 loader 无法移除 camera mount

原始 `relative_pose()` 核心代码：

```python
R_base = R_action @ np.linalg.inv(base_extrinsic[:3, :3])
```

旧数据中 `base_extrinsic_R = I`，所以：

```text
R_base_wrong = R_action @ inverse(I) = R_action
```

也就是把 camera optical rotation 当成机器人 base rotation。

随后：

```python
homo_RT = np.eye(4)
homo_RT[0:3, 0:3] = R_base
homo_RT[0:3, 3] = T_base

T_frame = np.dot(
    np.linalg.inv(homo_RT),
    np.concatenate((T_world, np.ones((T_world.shape[0], 1))), axis=-1).T,
).T[:, 0:3]

T_frame = T_frame[:, [1, 0, 2]]
T_frame[:, 1] = -T_frame[:, 1]
```

### 3.4 零 yaw、前进 1 m 的逐步数值推导

真实 Habitat forward：

```text
d_hab = [0, 0, -1]
```

转换到 parquet 的 data world：

```text
d_data = M_W @ d_hab = [0, 1, 0]
```

零 yaw 时：

```text
R_action = M_W
```

#### 旧 identity extrinsic

```text
R_base_wrong = M_W @ inverse(I) = M_W
raw_local     = transpose(M_W) @ [0,1,0]
              = [0,0,-1]
NavDP reorder = [raw_y, -raw_x, raw_z]
              = [0,0,-1]
```

#### 修复后的 `extrinsic_R = M_W`

```text
R_base_fixed  = M_W @ inverse(M_W) = I
raw_local     = I @ [0,1,0]
              = [0,1,0]
NavDP reorder = [raw_y, -raw_x, raw_z]
              = [1,0,0]
```

所以：

```text
旧标签：[forward=0, lateral=0, z=-1]
新标签：[forward=1, lateral=0, z= 0]
```

### 3.5 第三个空间坐标确实随后被丢弃

原始文件：
`/home/asus/Research/Nav/InternNav/internnav/dataset/navdp_dataset_lerobot.py`

函数 `xyz_to_xyt()`（原母目录约第 304--312 行）：

```python
def xyz_to_xyt(self, xyz_actions, init_vector):
    xyt_actions = []
    for i in range(0, xyz_actions.shape[0] - 1):
        current_vector = xyz_actions[i + 1] - xyz_actions[i]
        dot_product = np.dot(init_vector[0:2], current_vector[0:2])
        cross_product = np.cross(init_vector[0:2], current_vector[0:2])
        theta = np.arctan2(cross_product, dot_product)
        xyt_actions.append([
            xyz_actions[i][0],
            xyz_actions[i][1],
            theta,
        ])
    return np.array(xyt_actions)
```

这里所有位置和方向计算都只消费 `0:2`：

```text
[0, 0, -1] -> planar [0, 0]
```

因此丢失的不是齐次 w，也不是 LingBot 没预测出移动，而是 loader 把 forward 放到了
不属于 NavDP 地面平面的 z 槽位，`xyz_to_xyt()` 又按设计只保留平面前两维。

### 3.6 错误传播到训练目标

原始 MemNav `_build_actions()`：

```python
target_local_points, _, _, _, action_indexes = self.process_actions(
    extrinsics, base_extrinsic, 0, L - 1, pred_digit=pred_digit
)
init_vector = target_local_points[1] - target_local_points[0]
target_xyt = self.xyz_to_xyt(target_local_points, init_vector)
pred_xyt = target_xyt[action_indexes]

goal_rel_pose = target_xyt[-1].astype(np.float32).copy()
pred_actions = (pred_xyt[1:] - pred_xyt[:-1]) * 4.0
```

受影响内容：

| 对象 | 影响 |
|---|---|
| `pred_actions[..., 0]` | forward GT 丢失或严重失真 |
| `pred_actions[..., 1]` | lateral GT 与相机轴混合 |
| `pred_actions[..., 2]` | 由错误平面位移计算的 path-tangent delta |
| `goal_rel_pose[:2]` | aux/revisit 目标位置错误 |
| `goal_rel_pose[2]` | 由错误平面计算的路径方向不可靠 |
| bearing/direction metric | 会出现整轴、约 90/180 度异常 |

不直接受这个 axis bug 影响的主要是基于图像共视/DINO 构造的 retrieval positive/negative
mask。retrieval 找到帧之后的 pose/action 分支才会受到影响。

### 3.7 原 policy 中的经验校准只是下游症状

原始文件：
`/home/asus/Research/Nav/InternNav/internnav/model/basemodel/memnav/memnav_policy.py`

原始 `RevisitMerge` 使用：

```python
_R_CONV = (
    ( 0.0, -1.0,  0.0),
    (-1.0,  0.0,  0.0),
    ( 0.0,  0.0, -1.0),
)
_SCALE = 1.0 / 0.541
```

这组矩阵和固定 scale 是用旧标签拟合的补偿。已经被标签投影丢掉的水平自由度无法靠
全局线性 head 恢复，因此它不是根因修复。正确做法是先修 generator/loader 契约，再让
下游只使用明确的 LingBot `[z, -x]` 平面映射。

---

## 4. axis 修复在当前个人工作树中的位置

### 4.1 新生成数据写入正确 mount

当前文件：[`MemNavData/generate_twoleg.py`](MemNavData/generate_twoleg.py)，
`save_traj()` 约第 357--370 行：

```python
# Poses are stored as Z-up camera-to-world. NavDP expects action_R =
# base_R @ camera_mount_R and removes the mount before making planar labels.
# At zero Habitat yaw, action_R is M_W, so the corresponding mount is M_W,
# not identity. Its translation is the camera height in the Z-up data frame.
ext = np.eye(4)
ext[:3, :3] = M_W
ext[:3, 3] = M_W @ np.array([
    0.0,
    float(meta.get("camera_height_m", 0.5)),
    0.0,
])
```

旋转部分 `ext[:3,:3] = M_W` 是修复平面轴的关键。相机高度也被写入完整 extrinsic；
当前 `relative_pose()` 形成动作标签时主要使用 mount rotation，恒定高度对相对平面位移
通常会抵消，但正确 metadata 不应继续伪装成 identity。

### 4.2 统一的 pose convention helper

当前文件：
[`InternNav/internnav/dataset/memnav_pose_conventions.py`](InternNav/internnav/dataset/memnav_pose_conventions.py)

```python
HABITAT_TO_DATA_ROTATION = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
```

`generated_camera_extrinsic()` 构造新数据所需 mount；
`resolve_memnav_base_extrinsic()` 对旧数据做严格、受 marker 限制的兼容升级。

### 4.3 旧 pt1/pt2 数据的运行时兼容

旧 parquet 已经写成 identity。为了不重新渲染 RGB/depth，也不修改原始数据，当前
MemNav loader 在读入后执行：

文件：
[`InternNav/internnav/dataset/memnav_dataset_lerobot.py`](InternNav/internnav/dataset/memnav_dataset_lerobot.py)，
当前约第 606--614 行。

```python
(
    _camera_intrinsic,
    base_extrinsic,
    extrinsics,
    traj_len_parquet,
) = self.process_data_parquet(ti)

base_extrinsic = resolve_memnav_base_extrinsic(
    base_extrinsic,
    s.get('frame_convention'),
)
```

helper 的行为是：

1. 只有 `frame_convention` 明确以
   `positions+parquet in data(Zup,M_W)` 开头时才处理；
2. 如果已是 `M_W`，原样通过；
3. 如果是 legacy identity，替换 rotation 为 `M_W`；
4. 如果是未知非 identity rotation，直接报错；
5. unrelated InternData/NavDP 数据不改。

这避免了“为了修 MemNav，意外改坏其他真实机器人数据”的风险。

### 4.4 LingBot revisit 平面映射

当前文件：
[`InternNav/internnav/model/basemodel/memnav/revisit_pose.py`](InternNav/internnav/model/basemodel/memnav/revisit_pose.py)，
当前约第 122--126 行：

```python
planar = torch.stack((t_rel[..., 2], -t_rel[..., 0]), dim=-1)
```

明确语义：

```text
forward =  LingBot local z
left    = -LingBot local x
```

该映射用于 LingBot relative pose 的 action conditioning。它与 parquet action label 的
mount 修复是两条相互配套但位置不同的路径：

```text
Habitat GT/parquet -> NavDP loader -> diffusion action GT
LingBot pose9      -> revisit pose encoder -> diffusion conditioning
```

二者最终必须落在同一个 `+x forward, +y left` 的 NavDP 平面上。

---

## 5. `-179 deg` 与 `+179 deg`：问题不是角度值，而是普通减法

### 5.1 为什么两个数看起来差 358 度，实际只差 2 度

角度是模 `2*pi` 的圆周变量：

```text
+179 deg == -181 deg
-179 deg == +181 deg
```

从 `+179 deg` 变到 `-179 deg`：

```text
普通减法：-179 - (+179) = -358 deg
最短转动：+2 deg
```

反方向：

```text
-179 -> +179
普通减法：+358 deg
最短转动：-2 deg
```

标准 wrap：

```python
wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
```

输出区间是半开区间：

```text
[-pi, pi) == [-180 deg, 180 deg)
```

因此 `+pi` 统一表示成 `-pi`，避免同一朝向有两个边界表示。

另一种等价且数值清楚的写法是：

```python
wrapped = np.arctan2(np.sin(angle), np.cos(angle))
```

### 5.2 原始角度来自哪里

原始文件：
`/home/asus/Research/Nav/InternNav/internnav/dataset/navdp_dataset_lerobot.py`

`xyz_to_xyt()` 用每段轨迹方向相对初始方向计算 `theta`：

```python
current_vector = xyz_actions[i + 1] - xyz_actions[i]
dot_product = np.dot(init_vector[0:2], current_vector[0:2])
cross_product = np.cross(init_vector[0:2], current_vector[0:2])
theta = np.arctan2(cross_product, dot_product)
```

`atan2` 的输出天然位于 `[-pi, pi]` 附近。轨迹方向连续经过反向边界时，数值表示会出现：

```text
... 177 deg, 179 deg, -179 deg, -177 deg ...
```

物理轨迹连续，数字表示却在边界从 `+pi` 跳到 `-pi`。

### 5.3 原始 MemNav 的具体 bug 路径

原始文件：
`/home/asus/Research/Nav/InternNav/internnav/dataset/memnav_dataset_lerobot.py`

原母目录约第 367--374 行：

```python
init_vector = target_local_points[1] - target_local_points[0]
target_xyt = self.xyz_to_xyt(target_local_points, init_vector)
pred_xyt = target_xyt[action_indexes]

goal_rel_pose = target_xyt[-1].astype(np.float32).copy()
pred_actions = (pred_xyt[1:] - pred_xyt[:-1]) * 4.0
```

关键错误是：

```python
pred_xyt[1:, 2] - pred_xyt[:-1, 2]
```

对两个 `atan2` 输出做普通减法，却没有把差值 wrap 回 `[-pi,pi)`。

### 5.4 数值后果，以及为什么 x4 会放大异常

假设相邻采样 heading：

```text
theta_i   = +179 deg = +3.12414 rad
theta_i+1 = -179 deg = -3.12414 rad
```

旧代码：

```text
raw delta = -3.12414 - 3.12414
          = -6.24828 rad
          = -358 deg
```

正确 wrap：

```text
wrap(-6.24828) = +0.03491 rad = +2 deg
```

NavDP 随后把整个 action delta 乘以 4：

```text
旧 theta label：-6.24828 * 4 = -24.9931 rad，约 -1432 deg（缩放标签值）
新 theta label：+0.03491 * 4 = +0.1396 rad，约 +8 deg（缩放标签值）
```

这里的 `+8 deg` 是网络标签空间中的 x4 值；还原物理动作后仍是 `+2 deg`。旧 outlier
会让 diffusion noise target 的 theta 维出现极大幅值，产生以下训练现象：

- 少数跨边界样本支配 theta/action loss；
- loss spike 或高方差；
- 网络倾向把 theta 回归到接近均值以规避极端目标；
- 看似只差几度的动作被当作几乎整圈旋转；
- 长回转、U-turn 样本比直行样本更容易触发。

### 5.5 当前 MemNav 修复代码

统一函数位于：
[`InternNav/internnav/dataset/memnav_pose_conventions.py`](InternNav/internnav/dataset/memnav_pose_conventions.py)

```python
def wrap_radians(angle):
    """Wrap scalar/array angles to the half-open interval ``[-pi, pi)``."""
    angle = np.asarray(angle)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
```

应用位置：
[`InternNav/internnav/dataset/memnav_dataset_lerobot.py`](InternNav/internnav/dataset/memnav_dataset_lerobot.py)，
当前 `_build_actions()` 约第 540--564 行。

```python
goal_rel_pose = target_xyt[-1].astype(np.float32).copy()

# 使用真实最后一个 endpoint 的 x/y；这是与 angle wrap 相邻但独立的 endpoint 修复。
goal_rel_pose[:2] = target_local_points[-1, :2]
goal_rel_pose[2] = wrap_radians(goal_rel_pose[2])

pred_actions = pred_xyt[1:] - pred_xyt[:-1]

# 必须 wrap “差值”，而不只是分别 wrap 两端。
pred_actions[:, 2] = wrap_radians(pred_actions[:, 2])

# wrap 完再进入 NavDP 的统一 action scaling。
pred_actions *= 4.0
```

执行顺序非常重要：

```text
先相减 -> wrap delta -> 再乘 4
```

错误顺序示例：

```text
先乘 4 -> 再 wrap
```

这会把标签归一化尺度与物理角度周期混为一谈，不能保证得到正确物理 delta。

### 5.6 wrap 单个角度不等于 wrap 角度误差

即使 prediction 和 GT 都分别位于 `[-pi,pi)`，它们的普通差值仍可能接近 `2*pi`。

如果未来重新加入 aux theta loss，必须写成：

```python
theta_residual = wrap_radians(pred_theta - gt_theta)
theta_loss = np.mean(theta_residual ** 2)
```

而不是：

```python
theta_loss = np.mean((pred_theta - gt_theta) ** 2)  # 边界处错误
```

PyTorch 可写为：

```python
theta_residual = torch.atan2(
    torch.sin(pred_theta - gt_theta),
    torch.cos(pred_theta - gt_theta),
)
theta_loss = theta_residual.square().mean()
```

或者让 head 预测 `(sin(theta), cos(theta))` 并做单位向量监督。若已有完整旋转矩阵，使用
SO(3) geodesic error 比直接回归 Euler yaw 更稳健。

当前个人工作树中的 aux pose 监督是平面 translation direction，不含 theta；但 diffusion
action 的第三维仍是 path-tangent delta，所以 `_build_actions()` 的 wrap 仍然必要。

### 5.7 这个问题不是 generator 控制器没有 wrap

生成器的运动控制本身已有正确的最短角度处理，例如：

```python
alpha = (yaw_facing(to) - psi + np.pi) % (2 * np.pi) - np.pi
```

以及 terminal alignment 路径中的：

```python
dy = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
```

所以本次 `-179/+179` bug 的核心位置是 **loader 构造 action delta 的普通减法**，不是
Habitat 控制器生成轨迹时不会选择最短转向。

---

## 6. 同型 angle-wrap 风险的仓库范围

当前个人工作树已修复 MemNav 自己的 `_build_actions()`，因为 MemNav 训练调用该 override。
但下面的通用/其他数据路径仍能看到相同的 raw subtraction 形式：

```text
InternNav/internnav/dataset/navdp_dataset_lerobot.py
InternNav/internnav/dataset/navdp_dataset.py
InternNav/internnav/dataset/logoplanner_dataset_lerobot.py
```

典型代码：

```python
pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
```

这不仅是理论上的同型风险。使用本机原始 InternData-N1 做只读穷举检查（109 个 parquet，
对每个可用 start 和 `pred_digit=1..7` 构造 24-step 可达 action index）得到：

```text
包含至少一个 abs(raw delta theta) > pi 的 episode：3 / 109
能形成跨边界序列的 start frame：23
最大未 wrap delta：6.271681 rad，约 359.34 deg
```

这个检查扫描的是“loader 可以构造出的窗口”，不是官方训练 sampler 的真实出现频率，
因此不能直接说官方训练有多少百分比 batch 被污染；但它已经证明 raw subtraction 在官方
数据上存在可达反例，不只是我们的合成 U-turn 数据才会触发。MemNav 明确制造 revisit
和大回转，使该问题更频繁、更容易影响训练。

如果后续要训练通用 NavDP/LogoPlanner，应单独：

1. 确认第三维确实是周期 heading，而不是非周期标量；
2. 将 theta delta 的 wrap 放在 x4 scaling 之前；
3. 为每条数据路径补 `+179/-179` 单元测试；
4. 不在未确认 label semantics 时做全仓库机械替换。

本文没有擅自修改这些非 MemNav 路径。

---

## 7. 自动化验证

测试文件：
[`InternNav/tests/unit_test/test_memnav_pose_conventions.py`](InternNav/tests/unit_test/test_memnav_pose_conventions.py)

### 7.1 axis 测试

`test_generated_forward_motion_stays_in_navdp_xy_plane` 覆盖：

```text
yaw = 0, +pi/2, -pi/2, pi
```

每个朝向都断言：

```text
fixed local  = [1,0,0]
legacy xy    = [0,0]
abs(legacy z)= 1
```

这证明问题不是某一个 yaw 的偶然符号错误，而是 identity mount 在所有机器人朝向下都会
把自车 forward 留在第三个空间坐标。

### 7.2 两个水平自由度测试

`test_corrected_coordinates_recover_both_horizontal_axes` 验证一般位移：

```text
legacy local = [0.2, -1.0, -2.0]
fixed local  = [2.0, -1.0, 0.2]
```

并断言 fixed planar norm 与 Habitat 的真实地面 `x-z` norm 相同。

### 7.3 angle-wrap 测试

`test_angle_wrap_removes_atan2_branch_cut_jump` 验证：

```python
angles = np.array([
    (-np.pi + 0.01) - (np.pi - 0.01),
    (np.pi - 0.02) - (-np.pi + 0.02),
    0.25,
])

expected = [0.02, -0.04, 0.25]
```

即跨边界差值会恢复成小角度，普通非边界角度保持不变。

### 7.4 建议执行命令

从个人工作树根目录运行：

```bash
cd /home/asus/Research/Nav-axis-fix
PYTHONPATH=InternNav python -m unittest \
  InternNav/tests/unit_test/test_memnav_pose_conventions.py
```

本修复没有引入新的第三方依赖；该测试只依赖项目已有的 Python 标准库、NumPy 和
InternNav 包路径。

---

## 8. 如何解释训练结果

### 8.1 为什么错误标签下 action loss 仍然能下降

diffusion loss 比较的是网络输出和 loader 生成的训练 target。target 即使物理含义错误，
网络仍然可以拟合它。因此：

```text
低 training action loss
!= 正确的机器人 forward/lateral
!= 正确闭环导航
```

修复前后的 action loss 也不能直接横向比较，因为标签空间已经改变。

### 8.2 为什么 direction error 会出现约 90/180 度

axis bug 会交换/丢弃水平轴；angle bug 会在 `+pi/-pi` 附近制造整圈 outlier。二者叠加
时，position magnitude、bearing 和 theta 都可能异常。特别短的位移向量上 bearing 本身
也病态，因此应同时报告最小距离阈值后的 direction error，而不能只看一个角度值。

### 8.3 与 keyframe/长程 drift 的关系

axis/angle 与 LingBot keyframe 是不同层的问题：

```text
keyframe/dense-KV：影响 LingBot 长程 pose measurement 本身
axis/extrinsic：   影响 pose/GT 被解释成哪个 NavDP 平面
angle wrap：       影响平面 heading delta 跨边界时的数值表示
```

因此：

- keyframe 修复不能自动修正错误 action label；
- axis 修复不能消除真实长程 VO drift；
- 两者都正确后，仍需保留 angle wrap。

---

## 9. 最终数据流

### 9.1 原始错误路径

```text
Habitat camera-to-world
    -> generator: action_R = M_W @ R_hab
    -> generator: camera_extrinsic_R = I                 [错误]
    -> loader: R_base = action_R @ inverse(I)
    -> optical forward 留在 local z
    -> xyz_to_xyt 只读 local x/y
    -> forward 丢失
    -> theta_i 由错误平面 atan2 得到
    -> theta_(i+1) - theta_i 不做 wrap                  [第二个错误]
    -> diffusion action / goal_rel_pose 被污染
```

### 9.2 修复后的路径

```text
Habitat camera-to-world
    -> generator: action_R = M_W @ R_hab
    -> generator: camera_extrinsic_R = M_W
       或 loader 对有明确 marker 的 legacy identity 做 runtime upgrade
    -> loader: R_base = action_R @ inverse(M_W)
    -> NavDP local [forward, left] 正确落在前两维
    -> xyz_to_xyt 形成连续路径 heading
    -> delta_theta = wrap(theta_(i+1) - theta_i)
    -> wrap 后再做 x4 action scaling
    -> diffusion GT 与 LingBot `[z,-x]` conditioning 使用同一 NavDP 平面
```

## 10. 一句话结论

axis bug 不是“LingBot 没看见向前运动”，也不是官方 NavDP loader 的公式错误，而是我们
的 MemNav generator 漏写 camera mount，导致既有 loader 把相机光学 z-forward 留在第三
个空间坐标后丢弃；`-179/+179` bug 则是上游 NavDP 和我们早期 MemNav 都沿用的 action
label 普通角度减法。前者通过正确 `camera_extrinsic_R=M_W` 和 legacy runtime upgrade
修复，后者在 MemNav 路径中通过“先求角度差、wrap 到 `[-pi,pi)`、再 x4 scaling”修复。
