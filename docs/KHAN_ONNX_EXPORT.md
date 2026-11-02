# Khan 策略 ONNX 导出

`scripts/export_onnx.py` 将当前 Khan checkpoint 的 motion encoder 和 actor decoder 一起导出为单文件 FP32 ONNX，并在 ONNX metadata 中保存部署需要的关节顺序、PD 参数、默认姿态、动作缩放和观测契约。

导出时沿用训练 checkpoint 中的网络参数与固定 scale，不引入运行均值/方差归一化。世界模型、critic、优化器、replay、MuJoCo 模型和参考动作库均不进入推理图。动作可以在部署端按需加载，不必为每条动作重新导出网络。

## 导出命令

在本仓库根目录、具备 ONNX 导出及 ONNX Runtime 依赖的训练环境中运行：

```bash
uv sync --extra export

env -u PYTHONPATH .venv/bin/python scripts/export_onnx.py \
  --ckpt logs/Khan_lafan1_clean_goal/ckpt_0100000.pt \
  --out exports/khan_mini/ckpt_0100000.onnx
```

默认采用与已有 `latest_m62k_pico_base.onnx` 相同的单观测输入、单动作输出形式：

| 张量 | batch=1 形状 | 类型 | 含义 |
|---|---|---|---|
| `obs` | `(1, 620)` | float32 | `concat([goal155, policy_obs465], -1)` |
| `actions` | `(1, 27)` | float32 | 当前训练关节顺序下的原始归一化动作 |

图使用 IR 8、opset 17、动态 batch 维，部署通常用 batch=1。权重直接保存在该 ONNX 文件中，没有 external data 文件。旁边的同名 `.json` 是 metadata 和数值验证报告，运行 ONNX 时不依赖这个报告。默认拒绝覆盖已有模型或报告，确认重新导出时添加 `--overwrite`。

可添加一条本地 NPZ，额外验证真实参考状态构造的输入：

```bash
env -u PYTHONPATH .venv/bin/python scripts/export_onnx.py \
  --ckpt logs/Khan_lafan1_clean_goal/ckpt_0100000.pt \
  --out exports/khan_mini/ckpt_0100000.onnx \
  --motion data/npz/khan_mini/lafan1/dance2_subject1.npz \
  --validation-samples 64 --overwrite
```

`--motion` 只参与数值检查，不会把动作数据写入 ONNX。这份输入是参考状态及可重复的上一动作历史，不是策略闭环仿真；覆盖动作首帧回填和末帧窗口截断。

也可显式导出两个输入，便于使用已经分开构造观测的部署端：

```bash
env -u PYTHONPATH .venv/bin/python scripts/export_onnx.py \
  --ckpt logs/Khan_lafan1_clean_goal/ckpt_0100000.pt \
  --out exports/khan_mini/ckpt_0100000_split.onnx \
  --input-layout split
```

此时输入名为 `policy_obs`、`goal`，batch=1 形状分别为 `(1,465)`、`(1,155)`；输出仍为 `actions (1,27)`。两个版本执行同一份 encoder 和 decoder，区别仅为输入打包方式。具体维度、batch 约定及输入名称可通过 ONNX Runtime 的 `session.get_inputs()` 和 metadata 查询。

## 单输入 620 维的精确顺序

所有切片均为 Python 的左闭右开区间。长度为5的窗口都包含当前帧；参考窗口为 `t, t+1, ..., t+4`，控制周期为0.02秒。传感器历史为 `t-4, ..., t`，每项内部从旧到新排列。

| packed `obs` 切片 | 长度 | 数据 |
|---|---:|---|
| `[0:15]` | 15 | 5帧参考根位移，逐帧 XYZ |
| `[15:20]` | 5 | 5帧参考世界高度 Z |
| `[20:155]` | 135 | 5帧参考绝对关节角，每帧27维 |
| `[155:185]` | 30 | 当前机器人 base 下的5帧未来参考姿态，每帧6维 |
| `[185:200]` | 15 | 5帧历史 projected gravity，每帧3维 |
| `[200:215]` | 15 | 5帧历史 base angular velocity，每帧3维，已乘0.5 |
| `[215:350]` | 135 | 5帧历史相对关节角 `q-q0`，每帧27维 |
| `[350:485]` | 135 | 5帧历史关节速度，每帧27维，已乘0.2 |
| `[485:620]` | 135 | 5帧历史上一动作，每帧27维 |

这里采用 **term-major** 打包。先放完整5帧重力，再放完整5帧角速度，依次类推；不能先把每个时刻的重力/角速度/关节等拼成一个大帧，再直接将5帧展平。

`policy_obs` 双输入版本就是上述 `[155:620]`，`goal` 就是 `[0:155]`。训练网络所需的速度 scale 已包含在输入定义中，部署端只乘一次；导出的网络不会再替部署端乘0.5或0.2。

### 参考 goal

第 `k` 帧位移是：

```text
R_ref(t)^T * (p_ref(t+k) - p_ref(t))
```

使用当前参考 base 的完整姿态，不是仅 yaw 的 heading，也不是参考位置减实际机器人位置。第一帧位移恒为零。高度是参考的绝对世界 Z，关节角是参考绝对角度，不减默认姿态。

参考到达动作尾部后，未来索引分别截断到同一动作的最后一帧，不能取到下一条动作。ONNX 不内嵌参考表，也不推进 motion phase；调用侧负责选动作、构造参考窗口和同步推进参考帧。

### 未来相对姿态

每次推理用当前实际机器人 base 姿态重新计算5帧参考相对姿态：

```text
R_relative(k) = R_robot(t)^T * R_ref(t+k)
ori6(k) = [R00, R01, R10, R11, R20, R21]
```

`ori6` 是矩阵前两列再按行展平，不能改成 `[R00,R10,R20,R01,R11,R21]`。这30维每次重新计算，不再累积5帧历史。根和 anchor 都是 `base_link`，不要使用 `waist_yaw_Link` 的姿态代替。

若使用四元数构造这些量，本仓库和动作文件约定为 `wxyz`。重力观测为 `R_robot^T * (0,0,-1)`，角速度为当前 base 坐标系角速度；设备 IMU 若位于其他 link 或轴向不同，需要先完成坐标变换。

### 历史和输出反馈

首次启动时，用首次传感器观测填满各自5帧历史，上一动作初始化为零；之后每个50Hz策略周期每项只推进一帧。切换动作或重新启动时需明确重置历史与 phase，避免混用上一动作的参考窗口。

`last_action` 使用上一轮网络输出的归一化 `actions`，不是 PD 目标关节角、实际关节角或电机力矩。部署推理不注入训练观测白噪声、动作执行高斯噪声或域随机化 encoder bias。

## 动作含义和机器人参数

ONNX 输出没有额外动作裁剪或输出 tanh；encoder 的 latent tanh 是训练网络内部的一部分。正常无噪声控制接口为：

```text
q_target[j] = default_joint_pos[j] + action_scale[j] * actions[j]
```

当前 Khan 的27个 `action_scale` 都为0.5。输出是归一化关节位置偏移，不是力矩；PD 控制仍由部署控制器以 metadata 中的 stiffness/damping 参数执行。若部署端另有 target/torque clipping 或滤波，那是额外控制行为，不由本导出器加入。

关节限位、额定速度和 effort 等机器人参数用于部署端核对控制设置；ONNX 本身不计算 PD 力矩或执行硬件限位。当前 MuJoCo 位置伺服使用 effort 限幅，但 profile 中的速度上限并不是 MuJoCo 中额外启用的速度约束。

保留与旧 ONNX 兼容的 metadata 键：

| 键 | 内容 |
|---|---|
| `joint_names` | 27个策略关节名，逗号分隔，决定全部关节量和动作的顺序 |
| `joint_stiffness` / `joint_damping` | 按上述顺序排列的 Kp / Kd |
| `default_joint_pos` / `action_scale` | 按上述顺序排列的默认角度和动作缩放 |
| `effort_limit` / `velocity_limit` / `armature` | 按策略顺序排列的机器人参数 |
| `joint_pos_lower` / `joint_pos_upper` | URDF硬关节位置范围 |
| `soft_joint_pos_limit_factor` | 训练使用的软关节位置限位比例 |
| `control_dt` / `observation_history` / `goal_frames` | 控制周期0.02秒、历史5帧、参考5帧 |
| `command_names` | `motion` |
| `observation_names` | `command,motion_anchor_ori_b,projected_gravity,base_ang_vel,joint_pos,joint_vel,actions` |
| `anchor_body_name` | `base_link` |
| `body_names` | 当前15个跟踪 body 名称 |
| `onnx_deployment_schema` | JSON形式的详细输入、坐标系、scale、历史和动作契约 |
| `robot_contract` | checkpoint 的机器人/观测契约，含资产 hash 和训练特征定义 |

导出模型中的 metadata 为唯一随模型携带的部署说明。动作文件仍需使用与该契约匹配的27关节顺序、50Hz采样及 wxyz 四元数，部署端也需要按关节名称映射到实际电机接口。

## 与以前部署 ONNX 的区别

已有 `latest_m62k_pico_base.onnx` 也是 `obs (1,620) → actions (1,27)`，但接口形状相同不代表可直接复用旧观测构造：

- 旧模型使用左右交错的关节顺序；当前策略使用左腿、右腿、腰、左臂、右臂、颈部的训练顺序。必须按新 metadata 重排参考关节角、历史关节角/速度/上一动作和输出，不能只重排输出。
- 旧图包含学习到的 `normalizer._mean` 和除法统计；当前 SuperMimic actor 没有该 RMS。不要复制旧模型的均值、方差或外部归一化。
- 当前角速度/关节速度系数是0.5/0.2，不能继续使用旧 Isaac 观测的0.25/0.05。
- 当前网络为 `goal155 → encoder → latent32`，再与 `policy_obs465` 拼接后输入 decoder。不是旧 ONNX 的单一620维 PPO actor。
- 按关节名重排后，旧 ONNX 的 Kp/Kd/q0 与当前 profile 相符；仍应读取新模型 metadata，不依赖旧数组位置。

仅提供元数据不会自动完成真实设备的电机索引和 IMU 坐标映射；本次导出也不修改外部部署仓库。

## 最小 ONNX Runtime 调用

`goal` 和 `policy_obs` 必须按上述定义先构造好：

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "exports/khan_mini/ckpt_0100000.onnx",
    providers=["CPUExecutionProvider"],
)
obs = np.concatenate([goal, policy_obs], axis=-1).astype(np.float32)
assert obs.shape == (1, 620) and np.isfinite(obs).all()
actions = session.run(["actions"], {"obs": obs})[0]

meta = session.get_modelmeta().custom_metadata_map
joint_names = meta["joint_names"].split(",")
q0 = np.fromstring(meta["default_joint_pos"], sep=",", dtype=np.float32)
scale = np.fromstring(meta["action_scale"], sep=",", dtype=np.float32)
q_target = q0 + scale * actions[0]
```

双输入版本的调用改为：

```python
actions = session.run(
    ["actions"],
    {"policy_obs": policy_obs.astype(np.float32), "goal": goal.astype(np.float32)},
)[0]
```

## 验证范围

导出器检查 checkpoint 机器人/观测契约，保留训练网络固定 scale，并使用 ONNX checker 与 ONNX Runtime 比较 PyTorch/ONNX 输出。数值一致性覆盖随机观测及参考动作构造的输入；这验证的是图导出与输入接口，不替代实际部署控制器的闭环 sim2sim 或真机验证。

本次已对 `logs/Khan_lafan1_clean_goal/ckpt_0100000.pt` 导出并验证两个接口：

| 文件 | 图大小 | 接口 |
|---|---:|---|
| `exports/khan_mini/ckpt_0100000.onnx` | 27,473,231 bytes，约26.2 MiB | packed |
| `exports/khan_mini/ckpt_0100000_split.onnx` | 27,472,465 bytes，约26.2 MiB | split |

每个接口均通过零输入、64个随机 batch=1 输入、动态 batch=3，以及来自 `dance2_subject1.npz` 的64个参考状态输入检查。参考帧覆盖动作开头、中间和结尾；每个后端收到完全相同的输入。ONNX Runtime CPU 与 PyTorch CPU FP32 的最大误差如下：

| 检查 | 归一化动作最大绝对误差 | 换算PD目标最大绝对误差 |
|---|---:|---:|
| 全部离线组，packed / split相同 | `6.676e-6` | `3.338e-6` rad |
| NPZ参考状态组，packed / split相同 | `2.146e-6` | `1.073e-6` rad |

检查阈值为 `atol=1e-4, rtol=1e-4`。导出报告位于 `exports/khan_mini/ckpt_0100000.json` 和 `ckpt_0100000_split.json`，包含 checkpoint/动作/ONNX hash、输入schema、metadata、软件版本和每组误差。参考状态组是离线数值检查，没有执行物理仿真。

另外完成了一次实际 mjlab 观测接口 smoke：3个环境分别加载舞蹈、跑步、跳跃，关闭噪声和域随机化，连续64个策略步，由 ONNX Runtime CPU 的动作驱动仿真。每步将环境实际 `policy`/`goal` 同时送入 PyTorch 与 ONNX 比较，归一化动作最大误差 `2.861e-6`，环境实际PD目标与 `q0 + scale*actions` 的最大误差为0，无终止或超时。该测试每环境仅1.28秒，验证实际观测和PD映射接口；不代表长动作benchmark、CPU MuJoCo sim2sim或实机闭环验证。

此 smoke 的结果记录在 `logs/khan_onnx_export/live_observation_smoke.json` 和 `.log`。它是本次手工执行的短测记录，导出命令本身不会自动创建仿真或重跑该 smoke。

导出专用测试为 **11 passed**，覆盖两个接口/动态batch、metadata与维度、未裁剪动作语义、旧契约拒绝、非有限权重拒绝、重复scale拒绝、输出覆盖保护和验证失败时保留原文件：

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_onnx_export.py
```

## 参考

- [RoboJuDo-Plus SuperMimic 部署说明](https://github.com/chifongip/RoboJuDo-Plus/blob/supermimic/docs/supermimic.md)：完整 actor 或 encoder latent 缓存加 decoder 两种推理路径、保持训练观测与动作语义。
- [SuperMimic-Repro-Sim2Real 的 SIM2SIM 文档，固定提交 f8f60d7](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/blob/f8f60d7bbdba68468c050ef100af02d752ca238e/docs/SIM2SIM.md)：网络导出和参考动作表分离，导出后进行数值一致性检查。
- `ref_thirdparty/supermimic_ref/instantTrack/tools/export_onnx.py`：参考了机器人 metadata 的命名；该脚本选择内嵌单条动作并使用 `obs_history/time_step`，本次按用户需求保留通用外部观测接口，不内嵌动作。
- `latest_m62k_pico_base.onnx`：参考其 `obs/actions` 外部输入输出和逗号分隔 metadata 结构，不复制其中的权重、normalizer 或旧关节顺序。
