# Khan_mini 模型转换与 mjlab 预览

本阶段完成机器人资产转换、执行器配置和 mjlab 显示验证。来源为
`ref_thirdparty/jaka_code/Textop/TextOpTracker/source/textop_tracker/textop_tracker/robots/K1L.py`
中的 **Khan_mini_CFG**，以及它指向的 `assets/Khan_mini_simplified/khan.urdf`。
转换器直接解析该配置的 AST 字面量，不导入 Isaac Lab，不使用同文件里的 K1L 配置。

## 已生成的资产

```text
assets/robots/khan_mini/
  robot.xml       # 可独立编译的 MJCF 几何/惯性/关节模型
  profile.json    # 关节顺序、home、执行器、速度元数据、跟踪身体和维度
  source.urdf     # 保留视觉/固定 link、使用本地相对网格路径的 URDF
  source.json     # 原 URDF、K1L.py 和全部网格的 SHA-256，以及输出校验信息
  meshes/         # 44 份独立 STL 副本，不依赖 ref_thirdparty 路径
```

整个目录可复制到其他位置；加载配置时传入 `asset_dir` 即可。原始 URDF、配置和网格未修改。

| 项目 | 结果 |
|---|---|
| 可动关节 | 27：双腿各 6、双臂各 6、腰 1、颈 2 |
| 固定关节 / 机器人 link | 17 / 45 |
| 根 / 上身锚点 | `base_link` / `waist_yaw_Link` |
| 浮动根关节 | `floating_base` |
| `nq / nv / njnt` | 34 / 33 / 28（包含根的自由关节） |
| `nbody / ngeom / nmesh` | 46 / 71 / 44，nbody 包含 world |
| 执行器 | 原始 MJCF 的 nu=0；mjlab EntityCfg 创建 27 个 position servo |
| 总质量 | 40.8325349886391 kg |
| 初始根位置 | `(0, 0, 0.65)` m |
| 初始关节角 | 按 Khan_mini_CFG：双肩 roll=-1.57、双肘=1.57、双腕 yaw=0.3，其余 0 |
| 软关节限位 | 硬限位区间的 90% |
| 碰撞 | 45 个视觉 geom 与 26 个碰撞 geom 分开；保留自碰撞 |

带地面的 mjlab scene 多出一个 terrain body 和一个地面 geom，因此预览汇总显示
`nbody=47, ngeom=72`，机器人本身的拓扑保持上述数值。

## 转换处理

- 显式关闭 URDF 导入时的 `discardvisual` 和 `fusestatic`，保留全部网格及固定 link。
- 为根添加 freejoint，保留原始大小写、关节轴、限位和父子关系，包括
  `Right_wrist_yaw__Link` 的双下划线。
- 视觉放在 group 2，碰撞放在 group 3；视觉不参与接触，碰撞 contype/conaffinity 均为 1。
- 按源配置将 18 个 cylinder 碰撞形状转成 capsule，保留半径和轴向半长，端部由半球补齐。
  双脚共 14 个 capsule，接触 condim=3；其他接触 condim=1。
- 接触摩擦按当前 G1 对齐：足底 `(0.6, 0.005, 0.0001)`，非足底
  `(1.0, 0.005, 0.0001)`。该用户指定的配置在重建时保持，并记录在 profile 的
  `contact_friction` 和 `parameter_overrides` 中，不等同于 Isaac Lab 的材质参数。
- 对 MJCF 导出重新写入 17 位有效数字的质量、惯量和坐标变换，避免默认六位输出精度造成舍入。
- MJCF 不固定仿真器选项；预览环境显式使用 0.005 s 物理步长、4 倍 decimation，控制频率 50 Hz。

参考了 `ref_thirdparty/supermimic-retargeted/src/supermimic/robots/build.py` 和
`assets.py` 的 MjSpec→MJCF→profile→EntityCfg 结构。没有采用其中丢弃视觉网格、
自动修改惯量、猜测执行器增益或禁用自碰撞的配置。

## 执行器参数

下表均来自 Khan_mini_CFG 的有效值。PD、转子惯量和力矩限制由
`BuiltinPositionActuatorCfg` 应用，初态由 `EntityCfg.InitialStateCfg` 应用。

| 关节组 | 数量 | kp | kd | armature | 力矩限额 Nm | 速度元数据 rad/s |
|---|---:|---:|---:|---:|---:|---:|
| 髋、膝 | 8 | 187 | 18.7 | 0.03 | 120 | 11.52 |
| 踝 pitch | 2 | 100 | 2 | 0.016 | 96 | 14.6 |
| 踝 roll | 2 | 50 | 0.5 | 0.016 | 96 | 14.6 |
| 肩 pitch/roll | 4 | 102 | 10.2 | 0.016 | 96 | 11.5 |
| 肩 yaw、肘 | 4 | 40.8 | 4 | 0.01 | 36 | 15.7 |
| 腕 | 4 | 6.7 | 0.67 | 0.01 | 8 | 12.5 |
| 腰 yaw | 1 | 187 | 18.7 | 0.03 | 120 | 11.519 |
| 颈 | 2 | 6.7 | 0.67 | 0.01 | 8 | 8 |

mjlab 1.6 的内建 position servo 不支持 Isaac Lab `velocity_limit_sim` 的硬速度约束；
这些值保存在 profile 的 `velocity_limit` 中，`velocity_limit_enforced=false`。
源配置腰关节速度误写为 120 rad/s，现按用户确认修正为 URDF 的 **11.519 rad/s**。
转换器会在每次重建时应用该修正，`parameter_overrides` 记录原值及来源；
`ref_thirdparty` 内的原始输入保持不变。
PhysX 的速度上限、去穿透速度及求解器设置也不能直接等同 MuJoCo。

原先 profile 的 `action_scale` 沿用 G1 的 `0.25 * effort / kp`：其含义是在零速度时，
单位动作引入的 PD 比例力矩约为额定力矩的 25%。例如 Khan 髋关节为
`0.25*120/187 ≈ 0.16043 rad`，踝 pitch 为 0.24，踝 roll 为 0.48，腕为约 0.29851。

现按用户要求，**全部 27 个关节统一为 0.5**，不再使用该推导公式。
`get_khan_mini_action_cfg()` 返回与现有 G1 相同的 `NoisyJointPositionActionCfg` 接口，
实际读取 profile 的缩放；预览也使用这个工厂并关闭执行噪声：

```text
无执行噪声：q_target = q_home + 0.5 * action
有执行噪声：q_target = q_home + 0.5 * (action + epsilon)
```

action=1 对应相对 home 增加 0.5 rad（约 28.65°）。动作本身默认没有强制限制在 [-1,1]，
因此 ±0.5 rad 不是硬位置范围；URDF 关节限位和执行器力矩限制仍由模型处理。
K1L.py 末尾的 `K1_ACTION_SCALE` 属于另一台 K1L，不用于 Khan_mini。

## 使用方法

资产已经生成。需要重建时在仓库根目录执行：

```bash
env -u PYTHONPATH uv run python scripts/setup/build_khan_mini.py --force
```

默认拒绝覆盖已有目录；`--force` 会重新生成该机器人资产。可通过 `--urdf`、`--config`、
`--output-dir` 指定其他来源或输出目录。

有限步 GPU 检查并保存初始姿态图：

```bash
env -u PYTHONPATH uv run python scripts/preview_khan_mini.py \
  --headless --num-envs 4 --steps 200
```

输出为 `logs/khan_mini/preview.png` 和 `preview.json`。截图来自真实 mjlab 环境的
`env.render()`，只展示第一个环境；相机跟踪机器人根，适用于多个环境原点。

打开原生 mjlab 窗口：

```bash
env -u PYTHONPATH uv run python scripts/preview_khan_mini.py --viewer native --steps 0
```

窗口默认暂停，方便检查初始模型；使用空格切换播放。也可用 `--viewer viser` 启动
本机浏览器查看器（`127.0.0.1:8080`）。关闭窗口或 Ctrl+C 结束。

在其他 mjlab 场景中导入：

```python
from supermimic.robots.khan_mini import get_khan_mini_action_cfg, get_khan_mini_cfg

robot_cfg = get_khan_mini_cfg()  # 或传入完整 asset_dir
# 交给 SceneCfg(entities={"robot": robot_cfg}, ...) 使用
model = robot_cfg.build().spec.compile()  # 也可仅在 CPU 验证导入
assert (model.nq, model.nv, model.nu) == (34, 33, 27)
action_cfg = get_khan_mini_action_cfg()  # 全关节 0.5，默认执行噪声 std=0.1
```

## 验证记录

- 4 个 CUDA 环境完成 200 个控制步（每环境 4 秒），qpos/qvel/qacc、观测及奖励均有限。
- 960×720 的完整初态图已渲染并检查；原生 mjlab 窗口实际打开并完成 100 次暂停刷新后关闭。
- Viser 查看器也已完成初始化、10 次暂停刷新及关闭检查，测试后没有常驻窗口或服务。
- 7 项 CPU 资产测试全部通过：拓扑及质量、4 个姿态×45 个 link 的原 URDF FK 对拍、
  整目录搬迁加载、全部执行器和 home、碰撞与足底分组、拒绝覆盖本地编辑、重建后的速度修正与统一动作缩放。
- 2 个 CUDA 环境的实际非零动作检查通过：27 关节输入从 -1 到 1 的动作，
  应用后的关节目标严格等于 `q_home+0.5*action`，最大误差为 0；记录为
  `logs/khan_mini/action_scale_check.json`。
- 完整测试集 **31 passed, 1 warning in 27.09s**，无失败或跳过，包含原有 G1 训练回归。
  唯一提示为第三方 `torch.jit.script` 弃用提示；日志为 `logs/khan_mini/pytest.log`。

零动作仅保持关节目标，不提供浮动基座平衡控制。实测 4 秒 PD 检查中根高度从 0.65 m
降到约 0.099 m；这是没有平衡策略的动态预览，不是站立控制验收。

## 与当前仓库 G1 的对齐情况

这里的 G1 指当前 `tracking_env_cfg.py` 实际导入的 mjlab G1，而非 Isaac Lab 配置。
**模型资产与执行器/动作接口已对齐，机器人参数不相同，完整训练路径尚未切换到 Khan。**

| 项目 | Khan_mini | 当前 G1 |
|---|---|---|
| 资产 / 关节执行器 | MJCF + EntityCfg + BuiltinPositionActuatorCfg | 相同 |
| 原始 MJCF 的执行器 | nu=0，由 EntityCfg 创建 | 相同 |
| 软限位系数 | 0.9 | 0.9 |
| 动作处理 | NoisyJointPositionActionCfg，全部 scale=0.5 | 相同类型，scale 按 0.25×effort/kp 分组 |
| 视觉 / 碰撞组 | 2 / 3 | 2 / 3 |
| 自碰撞 mask | 1 / 1，开启 | 1 / 1，开启 |
| 足底碰撞 | 14 个，condim=3、priority=1 | 相同 |
| 非足底接触 | condim=1、priority=0 | 相同 |
| 碰撞几何数 / 覆盖 body | 26 / 14 | 33 / 18 |
| 足底摩擦 | (0.6, 0.005, 0.0001) | 相同 |
| 非足底摩擦 | (1.0, 0.005, 0.0001) | 相同 |
| 显式碰撞排除对 | 0 | 4 |
| 动作 / 状态 / goal / 历史观测维度 | 27 / 67 / 640 / 870 | 29 / 71 / 680 / 930 |
| sm-train 实际选用 | 尚未接入 | 当前固定为 G1 |

G1 显式排除 pelvis 与左右 hip_roll，以及左右 elbow 与 wrist_pitch 共四对碰撞。
两种模型都保留 MuJoCo 默认的同 body、焊接体和直接父子过滤，因此开启自碰撞不表示
任意两个 geom 都会检测。

还在无地面的纯机器人模型上，使用限位内确定性姿态探测到腰与左腕的 3 个自接触，
确认不只是 mask 数值开启，实际碰撞检测也能生效。实编译对比及探测姿态记录在
`logs/khan_mini_collision/g1_comparison.json`。

Khan 的 26 个碰撞 geom 只覆盖 14 个 body，其余 31 个 link 没有独立碰撞体，
包括头颈和部分髋/肩部。这是源 URDF 的简化结果；碰撞体截图显示其真实覆盖范围。
可用下列命令重新渲染多视角：

```bash
env -u PYTHONPATH uv run python scripts/render_khan_collisions.py
```

输出在 `logs/khan_mini_collision/`。碰撞渲染使用独立 CPU 模型，只改变显示颜色与透明度，
不改变 MJCF 或实际接触配置；淡色外观轮廓仅用于定位，不表示新增碰撞体。

后续训练仍需适配动作重定向、状态与观测维度、FK 和训练配置。Khan_mini 的维度是
state=67、每帧 goal=64、10 帧 goal=640、10 帧本体观测=870、action=27；当前 G1
训练路径的 71/680/930/29 维及 G1 LAFAN1 NPZ 不可直接用于 Khan_mini。
