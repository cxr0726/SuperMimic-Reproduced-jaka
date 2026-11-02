# Khan_mini 模型转换与 mjlab 预览

本文记录机器人资产转换、执行器配置、当前训练绑定和 mjlab 显示。来源为
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
| root / anchor / COM body | `base_link` / `base_link` / `waist_yaw_Link` |
| 跟踪 body | 原14个部位末尾追加 `Neck_pitch_Link`，共15个 |
| 浮动根关节 | `floating_base` |
| `nq / nv / njnt` | 34 / 33 / 28（包含根的自由关节） |
| `nbody / ngeom / nmesh` | 46 / 59 / 44，nbody 包含 world |
| 执行器 | 原始 MJCF 的 nu=0；mjlab EntityCfg 创建 27 个 position servo |
| 总质量 | 40.8325349886391 kg |
| 资产home根位置 | `(0, 0, 0.65)` m；运动跟踪reset另在参考root上加0.05 m |
| 初始关节角 | 按 Khan_mini_CFG：双肩 roll=-1.57、双肘=1.57、双腕 yaw=0.3，其余 0 |
| 软关节限位 | 硬限位区间的 90% |
| 碰撞 | 45 个视觉 geom 与 14 个碰撞 geom 分开；含双脚各1个完整mesh凸包，保留自碰撞 |

带地面的 mjlab scene 多出一个 terrain body 和一个地面 geom，因此预览汇总显示
`nbody=47, ngeom=60`，机器人本身的拓扑保持上述数值。

## 转换处理

- 显式关闭 URDF 导入时的 `discardvisual` 和 `fusestatic`，保留全部网格及固定 link。
- 为根添加 freejoint，保留原始大小写、关节轴、限位和父子关系，包括
  `Right_wrist_yaw__Link` 的双下划线。
- 视觉放在 group 2，碰撞放在 group 3；视觉不参与接触，碰撞 contype/conaffinity 均为 1。
- 非脚部的4个cylinder碰撞按源配置转成capsule，保留半径和轴向半长，端部由半球补齐。
- 双脚原有每脚7个sole capsule替换为每脚1个完整CAD mesh凸包，包含脚背与踝连接结构。
  复用随资产分发的 `Left_ankle_roll_Link.STL` / `Right_ankle_roll_Link.STL`，与SDK脚碰撞源mesh一致；
  使用visual原始局部姿态，由MuJoCo生成碰撞凸包，不引入SDK的其他接触掩码或关节动力学设置。
  脚接触仍为condim=3、priority=1、μ=0.6；非脚接触仍为condim=1。碰撞geom密度为0，显式URDF惯量不变。
  新名称为 `Left_ankle_roll_Link_convex_collision` / `Right_ankle_roll_Link_convex_collision`，profile和训练足部正则同步。
  部署只需已生成资产，不依赖外部SDK或ref目录；已有NPZ无需重新转换。
  左脚凸包在ankle_roll局部坐标中的包围范围约为X[-0.07496,0.12497]、Y±0.0475、Z[-0.048,0.014] m；
  最低点仍约为−0.048 m，替换凸包不会自动修复原始参考的接地偏差。
- 接触摩擦按当前 G1 对齐：足底 `(0.6, 0.005, 0.0001)`，非足底
  `(1.0, 0.005, 0.0001)`。该用户指定的配置在重建时保持，并记录在 profile 的
  `contact_friction` 和 `parameter_overrides` 中，不等同于 Isaac Lab 的材质参数。
- 对 MJCF 导出重新写入 17 位有效数字的质量、惯量和坐标变换，避免默认六位输出精度造成舍入。
- MJCF 不固定仿真器选项；预览环境显式使用 0.005 s 物理步长、4 倍 decimation，控制频率 50 Hz。

### 腰部碰撞盒

当前腰部box半尺寸为 **`(0.075, 0.07, 0.15) m`**，全尺寸 **`0.15×0.14×0.30 m`**。
原URDF的半尺寸为 `(0.075, 0.10, 0.15)`；本地仅缩窄Y，不改变X/Z、局部位置/姿态、质量或显式惯量。
有效尺寸由 `src/supermimic/robots/build_khan_mini.py` 持久化，重建后不会恢复成原宽度；
profile的 `parameter_overrides.waist_collision_box` 记录源尺寸、有效尺寸及原因。

140 mm宽度用于贴合约126–140 mm的中部视觉壳体，属于**碰撞核心近似**：上部较宽凸缘没有完全包住，
不能描述成整件视觉mesh的保守包络，也没有据此重新估算质量或惯量。
源文件及 `ref_thirdparty` 不变，覆盖仅应用到本地生成的MJCF。

旧40k策略的鲁棒评估冻结的是18 cm宽腰盒、14-body/waist-anchor配置，不能与当前14 cm宽模型混用；
对应资产SHA、录像及结果见 [KHAN_40K_ROBUSTNESS.md](KHAN_40K_ROBUSTNESS.md)。

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

## 历史资产验证记录

以下31项测试及初态预览来自较早的资产版本；当前腰宽、15-body/base-anchor及颈部奖励的
验证由本轮对应配置、FK、奖励和contract测试覆盖，历史数字不代表新语义的完整测试总数。
上一轮155/465观测、Isaac原scale、287维critic通过165项全量测试及8环境50步正式网络训练smoke，详见
[KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。
上一版SuperMimic scale、467维critic、goal仍加噪的版本通过167项全量测试及8环境50步训练smoke；
这些历史观测改动当时没有重建或修改机器人资产；当前v5另有本文记录的脚部凸包替换。
历史version-4干净goal版本通过168项全量测试及8环境50步正式网络/batch训练smoke，
15个跟踪body与模型接口确认；验证记录同见上述观测文档，不构成长片收敛或接地修复的证明。

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
**模型资产与执行器/动作接口已对齐；训练可通过 `--robot khan_mini` 选择 Khan，默认仍为 G1。**

| 项目 | Khan_mini | 当前 G1 |
|---|---|---|
| 资产 / 关节执行器 | MJCF + EntityCfg + BuiltinPositionActuatorCfg | 相同 |
| 原始 MJCF 的执行器 | nu=0，由 EntityCfg 创建 | 相同 |
| 软限位系数 | 0.9 | 0.9 |
| 动作处理 | NoisyJointPositionActionCfg，全部 scale=0.5 | 相同类型，scale 按 0.25×effort/kp 分组 |
| 视觉 / 碰撞组 | 2 / 3 | 2 / 3 |
| 自碰撞 mask | 1 / 1，开启 | 1 / 1，开启 |
| 足底碰撞 | 2个完整CAD mesh凸包，condim=3、priority=1 | 14个足底几何，condim=3、priority=1 |
| 非足底接触 | condim=1、priority=0 | 相同 |
| 碰撞几何数 / 覆盖 body | 14 / 14 | 33 / 18 |
| 足底摩擦 | (0.6, 0.005, 0.0001) | 相同 |
| 非足底摩擦 | (1.0, 0.005, 0.0001) | 相同 |
| 显式碰撞排除对 | 0 | 4 |
| 动作 / 状态 / goal / policy prop维度 | 27 / 67 / 155 / 465 | 29 / 71 / 680 / 930 |
| tracked bodies / anchor / COM body | 15 / base / waist | 14 / torso / torso |
| 奖励 / 最大单步奖励 | 原8项+颈部位置/速度两项 / 0.112 | 原8项 / 0.1 |
| sm-train 机器人选择 | `--robot khan_mini` | 默认 `--robot g1` |

G1 显式排除 pelvis 与左右 hip_roll，以及左右 elbow 与 wrist_pitch 共四对碰撞。
两种模型都保留 MuJoCo 默认的同 body、焊接体和直接父子过滤，因此开启自碰撞不表示
任意两个 geom 都会检测。

早期宽腰盒版本还在无地面的纯机器人模型上，用限位内姿态探测到腰与左腕的3个自接触，
确认自碰撞检测实际生效；这不是当前14 cm宽腰盒的接触数量结论。历史实编译对比及姿态记录在
`logs/khan_mini_collision/g1_comparison.json`。

Khan 的14个碰撞geom只覆盖14个body，其余31个link没有独立碰撞体，包括头颈和部分髋/肩部。
这与训练中跟踪15个body是两件事：增加 `Neck_pitch_Link` 跟踪不等于新增头部碰撞体。
这是源URDF的简化结果；碰撞体截图显示真实覆盖范围。
可用下列命令重新渲染多视角：

```bash
env -u PYTHONPATH uv run python scripts/render_khan_collisions.py
```

输出在 `logs/khan_mini_collision/`。碰撞渲染使用独立 CPU 模型，只改变显示颜色与透明度，
不改变 MJCF 或实际接触配置；淡色外观轮廓仅用于定位，不表示新增碰撞体。

训练已按机器人解析状态、观测、策略、世界模型、FK与replay尺寸。Khan当前默认 `isaac_khan`：
state=67、5帧参考goal=155、policy prop=465（未来姿态30＋5帧传感器历史435）、action=27、context=31。
actor本体角速度/关节速度有效scale为0.5/0.2；critic为545维，其中365维是干净参考特征，另增78维当前跟踪误差。
155维actor参考goal也在真实环境和想象训练中保持干净；prop本体/未来朝向噪声仍保留。
anchor使用base frame；COM随机化仍显式绑定waist。当前增加0–20 ms执行器延迟随机化及实际机器人reset的5 cm高度余量；
这些不改变关节PD/惯量、参考轨迹或actor输入维度。nominal eval关闭延迟，但保留同样的reset高度余量。
Khan version-5 `robot_contract` 保存上述仿真语义、模型/profile SHA、actor尺度及critic扩展schema；
旧Khan contract包括version-4 clean-goal/100k均不能直接用于当前训练、eval或export，
应使用新run或匹配的历史源码与资产。
G1的NPZ与检查点不能混用于Khan。已有45-body NPZ无需因本次观测、脚部碰撞或reset调整而重新转换。
正式参数见 [KHAN_TRAINING.md](KHAN_TRAINING.md)，Isaac观测的公式和差异见
[KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。
