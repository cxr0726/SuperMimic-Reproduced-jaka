# Khan LAFAN1 PKL → SuperMimic NPZ

数据格式适配阶段只进行了 CPU 数值验证，没有运行训练或修改机器人参数。
此后按用户要求将机器人非足底摩擦对齐 G1 的 1.0；该接触参数变化不影响 FK，
已有 NPZ 无需重新转换。转换报告保留当时模型的哈希，后续重播/审计记录当前模型哈希。

## 输入与输出

输入为用户已重定向的
`ref_thirdparty/jaka_code/dataset_pkl_jaka/lafan_pkl_data_new_wrist_merged_motion_new2_jaka_lafan_new_wrist.pkl`。
它实际采用 joblib 序列化，因此新增了 `joblib>=1.4` 读取依赖，当前锁定为 1.6.0。

| 项目 | 结果 |
|---|---|
| 动作数 | 36，全部转换，无筛除 |
| 帧数 | 396,990，全部输入均为 50 Hz，输出帧数保持一致 |
| 总时长 | 7,939.08 秒，即 2.2053 小时，按各段 `(T−1)/fps` 求和 |
| 单段帧数 | 5,109–13,065 |
| 输出目录 | `data/npz/khan_mini/lafan1/` |
| NPZ 文件总大小 | 1,015,256,120 bytes，约 968.22 MiB |
| 原文件 SHA-256 | `b6a760dd66871ad6430cd9fed70ad9b2c5276f6634ac5540bad24163bbc881d9` |

输出目录还有 `manifest.csv` 和 `conversion_report.json`。报告记录原始包、机器人 XML、
profile 以及每个输出 NPZ 的哈希，用于核对转换时使用的数据与模型。

36 段是源包本身的数量；相比原来的 G1 40 段，源包不含
`fallAndGetUp1_subject1`、`fallAndGetUp1_subject5`、`sprint1_subject2`、`sprint1_subject4`。

## 核实后的输入约定

每段包含 `fps=50`、`root_pos (T,3)`、`root_rot (T,4)`、`dof_pos (T,25)`，
三个数值轨迹都是 float64；`local_body_pos` 和 `link_body_list` 均为 None。
原始所有数值有限，四元数最大单位范数误差约 1.22e-15。

这份实际包的根四元数为 **wxyz**。旧 Jaka 脚本中的 XYZW 注释不适用于本文件：
按 wxyz 解释 walk/dance 的 400 个抽样姿态，根部向上方向 Z 分量中位数为 0.989，
最低脚 link 高度中位数约 0.0164 m；错误地按 xyzw 解释则分别为 −0.313 和 0.707 m。
转换器默认 wxyz，也提供显式 `--quat-order xyzw` 处理其他来源。

25 列不是缺失颈部，而是缺左右 wrist yaw：

```text
源 0:13   = 双腿 12 + waist yaw
源 13:18  = 左臂 shoulder pitch/roll/yaw、elbow、wrist roll
源 18:23  = 右臂同上
源 23:25  = Neck yaw、Neck pitch
```

映射到当前 27 关节模型时，在输出索引 18、24 插入左右 wrist yaw，均取模型 home 的
**0.3 rad**。原有 25 列按名称身份保持，包括源颈部两列（本包中全为 0）。
这个顺序不是 PKL 自带的关节元数据，而是结合旧 Jaka 补腕代码、列数及左右臂数值核实得到；
输出的 `source_joint_names` 明确记录了该解释。

如果未来输入为完整 27 列，则按模型原顺序使用；若提供 `joint_names`，支持按名称重排。
不明确的维度或缺失其他关节会报错，不会静默补值。

## 输出格式

沿用 SuperMimic / mjlab 运动 NPZ 的 11 个基础字段，`T` 为帧数：

| 字段 | 形状 | 说明 |
|---|---|---|
| `fps` / `input_fps` | `(1,)` | 本数据集均为 50 Hz |
| `joint_pos` | `(T,27)` | 按 Khan profile/MJCF 顺序，rad |
| `joint_vel` | `(T,27)` | rad/s |
| `body_pos_w` | `(T,45,3)` | 世界系 link 原点位置，m |
| `body_quat_w` | `(T,45,4)` | 世界系姿态，wxyz |
| `body_lin_vel_w` | `(T,45,3)` | 世界系 link 原点线速度，m/s |
| `body_ang_vel_w` | `(T,45,3)` | 世界系角速度，rad/s |
| `joint_names` / `body_names` | `(27,)` / `(45,)` | 数值列对应的实际名称，不含 world body |
| `source` | `()` | 来源包路径 |

数值字段为 float32，名称及来源字段为 Unicode，不需要 pickle。
另外记录 `robot`、`source_clip`、`source_quat_order`、`source_joint_names`、
`filled_joint_names`、`filled_joint_values` 六个元数据字段。

根为 `base_link`，根状态取自对应 `body_*` 列。所有 body 轨迹都由当前 Khan MJCF
在 CPU 上重新计算 FK。速度采用输出 dt 的梯度和最短路径 SO(3) 差分；写入 MuJoCo
freejoint qvel 前，根角速度转换到 body 系，输出仍为世界系。

没有采用旧脚本的丢前两帧、加高 0.025 m、按最低脚再抬升到 0.035 m 的处理，
也没有采用参考仓库的腿部 IK/穿地修复或关节裁剪。用户已有根轨迹和动作保持不变，
仅做格式、关节映射、数值精度及必要的采样/速度处理。

## 运行与独立验证

在仓库根目录执行：

```bash
# 默认转换指定的 Khan PKL；输出目录已存在时拒绝覆盖
env -u PYTHONPATH uv run python scripts/data/convert_khan_pkl.py

# 验证已经生成的数据，不运行仿真步或训练
env -u PYTHONPATH uv run python scripts/data/validate_khan_dataset.py

# 仅运行数据相关 CPU 测试，避免包含训练测试的完整 tests 目录
env -u PYTHONPATH uv run pytest \
  tests/test_khan_motion_convert.py tests/test_motion_convert.py -q
```

转换支持 `--input`、`--output-dir`、`--asset-dir`、`--limit`、`--output-fps` 和
`--quat-order`；确需重建现有数据时显式使用 `--force`。标准训练数据应保持输出 50 Hz。
整个目录在 staging 中成功生成后才发布，避免失败时留下不完整数据集。

独立验证器不调用转换器。实测结果：

- 全部 36 段、396,990 帧的字段、名称、形状、float32、有限性及四元数范数通过。
- 全帧核对 25→27 映射、原 root 位置/姿态及 NumPy/SciPy 独立速度计算通过。
- 288 抽样帧 × 45 body，使用 `mj_forward` 与 `mj_objectVelocity` 独立对拍通过。
  最大 body 位置误差约 2.384e-7 m、线速度误差约 2.301e-7 m/s、角速度误差约 9.376e-7 rad/s。
- 原始位置超过硬限位 1e-6 rad 的样本数为 **0**。
- **2,162 / 10,718,730** 个关节采样超过 profile 的速度元数据上限，约 0.0202%，
  峰值关节速度约 28 rad/s；仅记录，没有平滑、夹限或删除数据。
- 数据适配与原 G1 转换回归共 **18 passed, 1 warning in 3.49s**。
- mjlab 原生 `MotionLoader` 在 CPU 成功加载 `walk1_subject1.npz`：关节数组
  `(13065,27)`，选中的跟踪 body 数组 `(13065,14,3)`。

详细记录位于 `logs/khan_lafan_data/validation.json`、`tests.log`、`mjlab_loader.json`。

本次完成的是 **Khan 数据格式适配**。SuperMimic 当前的 MotionLibrary/训练器仍有
G1 的 pelvis 名称、29 关节、71 维状态和 G1 FK 假设，不能直接用当前 sm-train 启动 Khan。
本轮按要求没有修改这些训练端逻辑。

## 本轮关于摩擦的核对

非足底 0.6 是上轮转换器对全部 collision geom 统一赋值时引入的，并非 Khan_mini_CFG
或 URDF 中定义。原 Isaac 地面静/动摩擦为 1.0、combine=multiply；机器人所有 body 在
startup 随机化 static `[0.3,1.6]`、dynamic `[0.3,1.2]`，不能解释为固定 0.6。

在目前 MuJoCo 配置中，非足—非足接触 condim=1，没有切向摩擦约束；非足—默认地面
接触由地面的 μ=1.0 和同优先级组合得到有效 μ=1.0；足底优先级更高，足—非足采用
足底 μ=0.6。CPU 接触检查确认，单独把非足表值改为 1.0 不改变这些已检查的有效接触。
该核对发生在数据适配阶段，当时没有改动摩擦。后续已按用户要求将非足底改为 1.0，
作为与 G1 对齐的固定基线；这本身不意味着训练性能改善。
