# LAFAN1 NPZ 数据说明与审计

本次审计检查了 `data/npz/lafan1/` 下全部 40 个 NPZ，使用
`np.load(path, allow_pickle=False)` 逐个读取，没有重新下载或转换数据。
说明依据实际文件，以及 `motion_convert.py`、`motion_library.py`、
`commands.py`、`tracking_env_cfg.py` 和 `recorder.py` 中的数据处理逻辑。

## 数据规模与完整性

| 项目 | 实际结果 |
|---|---|
| NPZ 文件数 | 40 |
| 总帧数 | 441,140 |
| 输出帧率 `fps` | 所有文件均为 50 Hz |
| 输入帧率 `input_fps` | 所有文件均为 30 Hz |
| 总动作时长 | 8,822.00 秒，即 2.4505556 小时 |
| 单个动作帧数范围 | 5,109–13,656 |
| 单个动作时长范围 | 102.16–273.10 秒 |
| NPZ 文件总大小 | 790,881,528 字节，约 754.24 MiB |
| 数值有限性 | 全部数值为有限值，无 NaN/Inf |
| 四元数最大单位范数误差 | `max(abs(norm(body_quat_w) - 1)) = 1.1920928955078125e-7` |
| 字段结构 | 40 个文件均含下表 11 个字段，数值字段 dtype/尾部维度一致 |
| 名称及顺序 | 所有文件的 `joint_names`、`body_names` 完全一致 |
| Manifest 核对 | `manifest.csv` 的 40 条记录与实际文件的帧数、帧率和时长全部一致 |

每个 clip 的时长按首末采样点时间差 `(T - 1) / fps` 计算，以上总时长为
各 clip 时长之和。`MotionLibrary.summary()` 使用 `sum(T) / 50` 估算时长，
因此会比上述精确值多 `40 / 50 = 0.8` 秒；这是计算口径差异。

最短文件为 `fallAndGetUp3_subject1.npz`（5,109 帧，102.16 秒）；最长文件为
`sprint1_subject2.npz` 和 `sprint1_subject4.npz`（各 13,656 帧，273.10 秒）。

## 每个 NPZ 的字段

`T` 表示该动作的帧数。数值轨迹均为 `float32`；名称、来源路径为 NumPy
Unicode 字符串，无需开启 pickle。

| 字段 | 形状 | dtype | 含义和单位 |
|---|---|---|---|
| `fps` | `(1,)` | `float32` | 输出采样率，50 Hz |
| `joint_pos` | `(T, 29)` | `float32` | 29 个关节角，rad |
| `joint_vel` | `(T, 29)` | `float32` | 关节角速度，rad/s |
| `body_pos_w` | `(T, 30, 3)` | `float32` | 30 个身体 link 原点的世界坐标，m |
| `body_quat_w` | `(T, 30, 4)` | `float32` | link 在世界坐标系中的姿态，四元数顺序 **wxyz** |
| `body_lin_vel_w` | `(T, 30, 3)` | `float32` | link 原点的世界线速度，m/s |
| `body_ang_vel_w` | `(T, 30, 3)` | `float32` | link 的世界角速度，rad/s |
| `joint_names` | `(29,)` | `<U26` | `joint_pos`、`joint_vel` 第二维对应的关节名称 |
| `body_names` | `(30,)` | `<U25` | 四个 `body_*` 字段第二维对应的身体名称 |
| `source` | `()` | Unicode 字符串 | 转换时原 CSV 路径；仅作为来源元数据 |
| `input_fps` | `(1,)` | `float32` | 原 CSV 帧率，30 Hz |

例如 `dance1_subject1.npz` 的 `T = 6574`，因此 `joint_pos.shape` 为
`(6574, 29)`，`body_pos_w.shape` 为 `(6574, 30, 3)`，
`body_quat_w.shape` 为 `(6574, 30, 4)`。该文件 `source` 的值是
`data/raw/lafan1/g1/dance1_subject1.csv`；`source` 的 dtype 长度随文件名变化，
不影响数据结构或加载。

NPZ 没有独立的 `root_pos`、`root_quat` 字段。根节点是 `pelvis`，对应
`body_names[0]`，其参考位置、姿态、线速度和角速度分别取自四个
`body_*[:, 0]` 数组。

## 坐标系、姿态和速度

- `_w` 表示世界坐标系，沿用 MuJoCo 的 Z 向上坐标约定；位置、速度包含 XYZ 三个分量。
- `body_pos_w` 是身体 link 坐标原点的位置，不是质心位置；`body_lin_vel_w`
  同样对应 link 原点，而不是质心速度。
- `body_quat_w` 采用 **wxyz**，表示 link 在世界系中的朝向。原 LAFAN1 CSV
  的根四元数为 **xyzw**，转换程序将其重排为 wxyz。
- `body_ang_vel_w` 包括根节点在内，全部以世界坐标系表达。
- 转换时位置、关节角通过线性插值从 30 Hz 重采样到 50 Hz，根姿态通过
  最短路径 SLERP 插值。
- 关节角速度和根线速度使用重采样后轨迹的 `np.gradient` 计算；根角速度使用
  SO(3) 中央差分，边界重复最近的差分结果。
- 全部 link 姿态和速度通过 CPU MuJoCo 正运动学得到。转换程序从 `cvel`
  修正参考点得到 link 原点线速度，具体实现见
  `src/supermimic/data/motion_convert.py::forward_kinematics`。

需要区分角速度的两种表达：NPZ 和 MotionLibrary 参考状态中的根角速度为世界系；
MuJoCo `qvel` 与模型使用的仿真状态中的根角速度为 body 系。
`MotionLibrary.ref_qpos_qvel()` 会通过根四元数的逆旋转完成转换。

## 29 个关节的完整顺序

以下是零起始索引，对应 `joint_pos[:, i]` 和 `joint_vel[:, i]`，与
`src/supermimic/data/motion_convert.py::G1_JOINT_NAMES` 一致。

| 索引 | 名称 |
|---|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |
| 12 | `waist_yaw_joint` |
| 13 | `waist_roll_joint` |
| 14 | `waist_pitch_joint` |
| 15 | `left_shoulder_pitch_joint` |
| 16 | `left_shoulder_roll_joint` |
| 17 | `left_shoulder_yaw_joint` |
| 18 | `left_elbow_joint` |
| 19 | `left_wrist_roll_joint` |
| 20 | `left_wrist_pitch_joint` |
| 21 | `left_wrist_yaw_joint` |
| 22 | `right_shoulder_pitch_joint` |
| 23 | `right_shoulder_roll_joint` |
| 24 | `right_shoulder_yaw_joint` |
| 25 | `right_elbow_joint` |
| 26 | `right_wrist_roll_joint` |
| 27 | `right_wrist_pitch_joint` |
| 28 | `right_wrist_yaw_joint` |

## 30 个身体的完整顺序

以下是零起始索引，对应 `body_*[:, i, :]`。这些是 MuJoCo 模型中除 world
以外的所有 body；不含单独的手指自由度。

| 索引 | 名称 |
|---|---|
| 0 | `pelvis` |
| 1 | `left_hip_pitch_link` |
| 2 | `left_hip_roll_link` |
| 3 | `left_hip_yaw_link` |
| 4 | `left_knee_link` |
| 5 | `left_ankle_pitch_link` |
| 6 | `left_ankle_roll_link` |
| 7 | `right_hip_pitch_link` |
| 8 | `right_hip_roll_link` |
| 9 | `right_hip_yaw_link` |
| 10 | `right_knee_link` |
| 11 | `right_ankle_pitch_link` |
| 12 | `right_ankle_roll_link` |
| 13 | `waist_yaw_link` |
| 14 | `waist_roll_link` |
| 15 | `torso_link` |
| 16 | `left_shoulder_pitch_link` |
| 17 | `left_shoulder_roll_link` |
| 18 | `left_shoulder_yaw_link` |
| 19 | `left_elbow_link` |
| 20 | `left_wrist_roll_link` |
| 21 | `left_wrist_pitch_link` |
| 22 | `left_wrist_yaw_link` |
| 23 | `right_shoulder_pitch_link` |
| 24 | `right_shoulder_roll_link` |
| 25 | `right_shoulder_yaw_link` |
| 26 | `right_elbow_link` |
| 27 | `right_wrist_roll_link` |
| 28 | `right_wrist_pitch_link` |
| 29 | `right_wrist_yaw_link` |

## 从参考动作到训练数据

### MotionLibrary 与参考跟踪

`training/loop.py::Trainer` 通过 `discover_npz()` 发现动作文件，
`MultiClipMotionCommand` 创建 `MotionLibrary`。每帧抽取 pelvis 状态与关节状态，
拼接为以下 71 维参考向量：

```text
root_pos_w(3), root_quat_w(4), joint_pos(29),
root_lin_vel_w(3), root_ang_vel_w(3), joint_vel(29)
```

默认 full 模式还缓存下列 14 个跟踪 body 的位置、姿态和线/角速度：

```text
pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link,
torso_link, left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link,
right_shoulder_roll_link, right_elbow_link, right_wrist_yaw_link
```

锚点为 `torso_link`，根节点为 `pelvis`。这些参考量用于机器人参考状态重置、
跟踪奖励、根部高度/倾斜终止判断等。动作按 50 帧划为约 1 秒的 bin，
本数据集共 **8,842 个 bin**，用于基于回报的重置采样。

### 策略目标特征

`motion_library.py::goal_features` 为每帧构造 68 维目标特征：

```text
joint_pos - default_joint_pos (29)
joint_vel                    (29)
projected_gravity             (3)
root_ang_vel_body             (3)
root_lin_vel_heading          (3)
root_height                   (1)
```

其中根角速度转换为根 body 系，根线速度转换为根 heading（偏航）系。
`goal_window()` 默认拼接当前参考帧及后续 9 帧，得到 **680 维 goal**；
若越过 clip 末尾则重复末帧。其余策略本体观测来自当前仿真状态及历史，
不是直接从 NPZ 读取。

full library 主要 float32 张量的理论大小为：

```text
441140 × 4 bytes × (71 + 14 × 13 + 68) = 566,423,760 bytes
```

即约 0.566 GB，另有索引和分桶等张量开销。compact 模式只保留参考状态，
跟踪 body 和目标特征按需计算。

### 在线仿真转移与世界模型

NPZ 保存的是参考动作，**没有 action、reward、done、contact、观测历史、
域随机化参数或状态转移样本**。训练中执行策略动作后，
`data/recorder.py::ReplayRecorder` 向 GPU replay buffer 记录实际仿真转移，
包括状态 `s`、干净动作 `a`、带噪执行动作 `a_exec`、仿真上下文 `xi`、
奖励 `r`、终止标签 `done`、参考帧/clip 索引和 episode 标记。
世界模型从这些在线仿真转移学习动力学；NPZ 提供跟踪参考、奖励目标和想象目标。

## 加载器当前的约束

这些是当前代码的实际行为，审计未修改加载器：

1. `MotionLibrary` 读取 `joint_pos`、`joint_vel`、四个 `body_*` 数值字段，
   并使用 `body_names` 将被跟踪身体映射到列索引。
2. 当前加载逻辑不读取或校验 `fps`、`input_fps`、`joint_names`、`source`。
   仿真控制频率固定为 `1 / (0.005 × 4) = 50 Hz`，每步推进一个参考帧，
   因而需要输入为 50 Hz。`joint_names` 仅是元数据，实际数组列必须严格遵守
   上述 29 关节顺序。
3. `body_names` 到列索引的映射只从首个文件建立，后续文件沿用，因此全部文件
   的 body 顺序也必须一致。本次全量检查已确认这些条件成立。
4. `source` 是转换时的来源记录，不要求该原 CSV 路径在当前机器仍然存在。
   训练使用现有 NPZ 即可。

## 最小读取示例

在仓库根目录执行：

```bash
env -u PYTHONPATH uv run python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("data/npz/lafan1/dance1_subject1.npz")
with np.load(path, allow_pickle=False) as motion:
    for key in motion.files:
        value = motion[key]
        print(key, value.shape, value.dtype)
    root_index = motion["body_names"].tolist().index("pelvis")
    print("fps:", motion["fps"][0])
    print("frames:", motion["joint_pos"].shape[0])
    print("first root position (m):", motion["body_pos_w"][0, root_index])
    print("first root quaternion (wxyz):", motion["body_quat_w"][0, root_index])
PY
```
