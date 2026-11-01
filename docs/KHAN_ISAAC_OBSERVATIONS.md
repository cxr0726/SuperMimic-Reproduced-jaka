# Khan 观测：Isaac 布局、SuperMimic 尺度与 critic 参考

当前Khan默认选择 `isaac_khan`，对齐用户提供的 `commands_multi.py.command` 和
`tracking_env_cfg_jaka.py.ProjGravObservationsCfg.PolicyCfg`。实际Isaac训练命令见
`ref_thirdparty/jaka_code/Textop/USAGE.md`：任务为 `Tracking-Flat-jakamini-ProjGravObs-MNMLP-v0`，
显式覆盖anchor为 `base_link`、future_steps为5。

当前仍采用SuperMimic世界模型与想象训练。actor观测的内容、顺序、坐标系沿用Isaac布局，
本体有效scale按要求改回SuperMimic；155维参考goal在真实环境和想象rollout中均不加噪声，
与原G1默认参考窗口一致。prop本体和未来朝向保留此前噪声，下面逐项比较差异。
没有把训练算法改成Isaac端的PPO，也没有迁移PPO的running观测归一化。
15个跟踪body、base anchor、腰部COM随机化、颈部奖励和MJCF保持已有设置；本次范围仅为观测与对应网络输入适配。

## 先区分奖励归一化与观测归一化

Khan正奖励权重之和为5.6，dt=0.02，所以理想单步上界为 `r_max=0.112`。
若一个回合实际执行T步，其日志回报是：

```text
episode_norm_return = sum(r_t) / (0.112 * T)
train_ret = mean(episode_norm_return of completed episodes in the logging interval)
```

新增颈部奖励前分母使用 `0.1*T`，现在分母同步增加，因此 **train_ret理论上仍最大为1.0，
不是1.12**。没有clamp到[0,1]；动作变化/限位处罚可以使它为负。浮点计算可能带来极小的边界误差。
benchmark及录制的回报也使用正奖励总上界乘实际步数。早失败尚未执行的尾段不进入分母，
评估时仍须同时看摔倒数和完整步数覆盖率。

困难动作采样使用折扣return-to-go，尺度为 `Gmax=r_max/(1-gamma)`。
当前gamma=0.99，Khan为11.2，G1为10；Khan的bin return初值也改成11.2：

```text
hard_distribution = softmax(-25 * bin_return / 11.2)
sampling_distribution = 0.5 * uniform + 0.4 * hard + 0.1 * memory
```

这调整了采样器中回报的相对尺度，温度和混合比例没有改变。Gmax是尺度而非强制截断上限，
尤其critic bootstrap的估计不被clamp到Gmax。训练中的真实奖励、想象奖励及critic目标仍使用原始
dt缩放奖励，没有为日志归一化而统一除以0.112。

上述奖励归一化与下面的观测单位缩放、Isaac PPO的running mean/std是三件不同的事。

## goal command：155维

5个参考帧是 **t、t+1、t+2、t+3、t+4**，包含当前参考帧；在50 Hz下覆盖0至80 ms。
设参考base位置/旋转为 `p_ref(k), R_ref(k)`，则：

```text
root_pos_diff[k] = transpose(R_ref(t)) * (p_ref(t+k) - p_ref(t))
root_z[k]        = p_ref(t+k).z
joint_pos[k]    = q_ref(t+k)
goal            = [flatten(root_pos_diff), flatten(root_z), flatten(joint_pos)]
```

位移使用**当前参考base的完整旋转**，不是机器人当前旋转，也不是仅yaw。
第0帧位移恒为0；该项不会直接提供机器人相对参考的全局位置误差。
root_z保留世界高度；当前平地的环境origin z为0。参考关节角是绝对角，不减default pose。

| 顺序 | 分量 | 维度 | 加性均匀噪声半宽 | scale |
|---|---|---:|---:|---:|
| 1 | 5帧root_pos_diff | 15 | 0 | 1 |
| 2 | 5帧root_z | 5 | 0 | 1 |
| 3 | 5帧joint_pos | 135 | 0 | 1 |
| 合计 | | **155** | | |

拼接按分量分组，每组内部按未来时间排列，不是每帧31维交错。
与上一版相比，goal不再包含参考关节速度、参考重力、参考角速度、参考线速度或末尾9维anchor扩展。
未来越过当前clip末帧时重复末帧，不跨到下一段。
这155维在训练中也始终保持干净：真实环境的goal group关闭corruption且term没有noise，
想象rollout直接使用参考command，不再采样goal噪声。`obs_noise=True`仍控制prop的本体和未来朝向，
不重新开启goal噪声。Isaac原配置中的高度±0.01、参考q±0.05已按本次要求移除。

## policy prop：465维

未来anchor姿态使用当前**机器人base**为坐标系：

```text
R_relative[k] = transpose(R_robot(t)) * R_ref(t+k)
ori6[k]      = [R00, R01, R10, R11, R20, R21]
```

这是旋转矩阵前两列按行展开，5帧共30维。
该项每次按当前机器人姿态重算，不加历史缓冲，也不包含anchor相对位置。

| 顺序 | 分量 | 时间布局 | 维度 | 噪声半宽（缩放前） | scale |
|---|---|---|---:|---:|---:|
| 1 | motion_anchor_ori_b | 参考t…t+4，相对当前机器人 | 30 | 0.05 | 1 |
| 2 | projected gravity | 历史t−4…t | 15 | 0.07 | 1 |
| 3 | base angular velocity | 历史t−4…t | 15 | 0.2 rad/s | **0.5** |
| 4 | joint position relative to default | 历史t−4…t | 135 | 0.01 rad | 1 |
| 5 | joint velocity relative to default | 历史t−4…t | 135 | 0.5 rad/s | **0.2** |
| 6 | last action | 历史t−4…t | 135 | 0 | 1 |
| 合计 | | | **465** | | |

历史按项分组、每项最旧到最新。首次reset后，第一条已经加噪和缩放的观测填满该项5个历史槽；
后续逐步移入新帧。噪声在scale前加入，所以最终角速度与关节速度噪声半宽均为0.1。
goal和prop的静态网络scale均为1，避免环境内乘0.5/0.2后又重复缩放。
想象训练沿用本库从replay状态/动作重建历史的方法，并重新采样观测噪声；没有保存、重放真实环境
当时的噪声取值。reset回填帧复用同一条首次噪声观测，保持Isaac的历史相关性；后续有效帧各自采样。
IsaacLab v2.1.0的[ObservationManager](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/managers/observation_manager.py)
和[CircularBuffer](https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py)
确认了上述顺序与reset填充规则。

## 原SuperMimic本体scale与当前scale

原SuperMimic的单帧本体顺序为 `[joint_pos, joint_vel, base_ang_vel, gravity, last_action]`。
`PolicyCfg.obs_term_scale=(1, 5, 2, 1, 1)` 表示**除数**，不是直接乘数。
原策略在网络输入侧用其倒数乘观测；原goal同样使用分量除数 `(1, 5, 1, 2, 1, 1)`，
对应参考q、qd、重力、角速度、线速度和高度，可选anchor扩展除数为1。

| 本体分量 | 原SuperMimic除数 | 原有效乘数 | Isaac原配置乘数 | 当前Khan乘数 |
|---|---:|---:|---:|---:|
| joint position | 1 | 1 | 1 | 1 |
| joint velocity | 5 | 0.2 | 0.05 | **0.2** |
| base angular velocity | 2 | 0.5 | 0.25 | **0.5** |
| projected gravity | 1 | 1 | 1 | 1 |
| last action | 1 | 1 | 1 | 1 |

当前Khan将本体顺序改成上表之前说明的gravity、角速度、关节位置、关节速度、动作，
并在环境/解析观测路径内先加噪、再乘固定scale；网络侧的goal/prop scale均为1。
当前本体有效scale与原SuperMimic一致，`isaac_khan` 名称只标识特征布局和5/5窗口，
不再表示全部尺度与Isaac相同。这是固定单位缩放，不是运行均值/方差归一化；没有引入新的统计量或PPO训练组件。
G1仍沿用原网络侧的分量除数。世界模型与critic的canonical state缩放独立于actor，保持原值。

## 原仓库与当前噪声的比较

这里核对的是本项目上游复现仓库 `CuboiLeo/SuperMimic-Reproduced`，本地 `upstream/main`
锁定为 `ba6105349069bf040f1fdad60873d13138d505a2`，不是将其称为论文作者官方训练实现。
该提交 `envs/tracking_env_cfg.py:97–126`、`imagination/rollout.py:18–69`、
`policy/nets.py:24–48` 给出下表；`ref_thirdparty/supermimic-retargeted` 的对应路径使用相同机制。

| 分量 | 上游训练噪声半宽，scale前 | 当前Khan，scale前 | 当前网络入口半宽 |
|---|---:|---:|---:|
| 实际joint position | 0.01 rad | 0.01 rad | 0.01 |
| 实际joint velocity | 0.5 rad/s | 0.5 rad/s | 0.1 |
| 实际base angular velocity | 0.2 rad/s | 0.2 rad/s | 0.1 |
| 实际projected gravity | **0.05** | **0.07** | 0.07 |
| last action | 0 | 0 | 0 |
| 默认goal中的参考特征 | **不加噪** | **155维全部不加噪** | 0 |
| 可选anchor扩展 | 单帧pos 0.25 m、ori 0.05 | 无该9维扩展；5帧future orientation为0.05 | 0.05 |

原仓库默认 `anchor_obs=False`，因此默认goal整个参考窗口都是干净的；只有显式开启的
额外anchor项加噪。当前goal已按原G1方式保持干净；gravity的0.07及prop未来朝向0.05仍保留，
本次没有关闭整个actor的观测噪声，也没有把gravity范围改成原版0.05。
critic的全部参考输入始终保持干净，包含本次增加的180维。

噪声均为逐分量均匀加性噪声，不是Gaussian：

```text
epsilon = Uniform(-half_width, +half_width)
原真实环境：raw_obs + epsilon → history → policy固定scale
当前真实环境：(raw_obs + epsilon) * scale → history → policy(scale=1)
当前想象路径：analytic_obs_scaled + epsilon_scaled → history → policy(scale=1)
epsilon_scaled = Uniform(-half_width*scale, +half_width*scale)
```

以上位置不同但分布等价。mjlab管理器实际顺序为compute→noise→clip→scale→history；
当前观测没有配置额外clip。噪声在每个新观测、每个环境、每个分量上重新采样。
历史缓存保留各帧当时加入的噪声；想象训练按原SuperMimic方法从replay状态/动作重建历史并重新采样，
不保存真实环境白噪声的具体取值。当前reset回填共享首次噪声，原版imagine对回填的各槽独立采样。

训练有两个独立开关：`env.obs_noise` 控制真实环境，`policy_trainer.obs_noise` 控制想象训练，
默认均为True；只关闭前者不会自动关闭后者。benchmark、play和record显式关闭观测噪声。
表中的白噪声与每回合encoder bias±0.01是不同机制；last_action观测无噪声，执行动作仍有
独立Gaussian std=0.1，这属于动作执行噪声。

## 当前接口与代码修改

| 接口 | 上一版Khan | 当前Khan | G1原默认 |
|---|---:|---:|---:|
| 参考 / 历史帧数 | 10 / 10 | 5 / 5 | 10 / 10 |
| goal | 649 | 155 | 680 |
| policy prop | 870 | 465 | 930 |
| decoder输入（含latent32） | 902 | 497 | 962 |
| critic输入 | 751 | **467** | 788 |
| state / action / context | 67 / 27 / 31 | 67 / 27 / 31 | 71 / 29 / 33 |

- `observation_layout.py` 统一定义布局、尺寸、干净参考command、未来anchor姿态及critic扩展。
- `config.resolve_training_observations` 在建网络前解析机器人默认值：Khan为5/5，G1保持10/10；
  明确传入不兼容的Isaac帧数或旧anchor扩展会报错。
- 环境观测与MotionLibrary提供对应窗口；解析观测和想象rollout使用同样的项顺序、坐标公式、
  噪声与scale，并分别处理未来姿态和本体历史。
- 策略、critic及评估入口按布局确定输入尺寸；Khan观测不再套原SuperMimic分量除数。
- critic使用独立的365维干净参考输入：command155＋未来anchor姿态30＋5帧速度/重力特征180。
  `ImagRollout.critic_goals` 保留独立参考接口，其前155维与actor的干净goal一致，
  参考特征按原SuperMimic单位尺度进入critic。
- Khan checkpoint contract升级为version 4，记录干净actor goal、layout、帧偏移、排序和尺度，防止旧观测策略误用。
- 服务器入口显式选择 `isaac_khan`，默认run名为 `khan_lafan1_clean_goal`；smoke也解析同一布局。

encoder/decoder隐藏层、latent、WM网络、critic隐藏层、优化器、batch、H、UTD及学习率计划沿用本库正式值。
输入层宽度随观测变化。G1继续使用原SuperMimic观测及可选9维anchor路径。

## 原critic与当前critic的输入

本次保留SuperMimic的critic结构：当前canonical state、参考信息、物理上下文xi及root误差。
参考输入补齐原motion goal中的速度与重力信息，同时保留当前command和未来朝向：

```text
原G1默认，无anchor扩展：71 + 680     + 33 + 4 = 788
上一版Khan，含9维anchor：67 + 649     + 31 + 4 = 751
上一轮5帧布局Khan：     67 + 155 + 30       + 31 + 4 = 287
当前Khan：              67 + 155 + 30 + 180 + 31 + 4 = 467
```

原G1参考680维为10帧×68维；上一版Khan参考649维为10帧×64维＋9维anchor。
当前参考365维由干净command155、干净未来anchor朝向30和新增180维组成，actor的噪声不进入该参考输入。
未来朝向依赖当前机器人姿态，因此在想象rollout每一步重新计算。

新增180维按帧排列：`[frame_t的36维, frame_t+1的36维, …, frame_t+4的36维]`，每帧依次为：

| 分量 | 每帧维度 | 定义 | critic网络入口乘数 |
|---|---:|---|---:|
| 参考joint velocity | 27 | NPZ的参考关节速度 | 0.2 |
| 参考projected gravity | 3 | `R_ref(k)^T * (0,0,-1)` | 1 |
| 参考root angular velocity | 3 | `R_ref(k)^T * omega_ref_world(k)` | 0.5 |
| 参考root linear velocity | 3 | `R_yaw_ref(k)^T * v_ref_world(k)` | 1 |

这里使用每个未来参考帧自身的完整root旋转或heading，不使用实际机器人的坐标系。
MotionLibrary提供干净原始值，critic的scale buffer只对新增180维应用上述乘数，前185维仍乘1。
连同已有q、root_z，当前critic覆盖原SuperMimic每帧参考特征的全部类型；区别仍有5帧窗口、
参考位移与未来相对朝向等附加信息。参考重力与已有未来朝向存在信息冗余，但显式提供可以减少网络
自行恢复该量的负担；关节/根速度则直接给出目标运动速度，无需从短位置窗口近似差分。
这使value估计的信息更完整，是否提升稳定性和最终回报仍需训练对照，不能由smoke test推断。

Khan的canonical state共67维，使用 `StateSpec.canonical()`，没有历史堆叠：

| 顺序 | 分量 | 维度 | 除数（保持原SuperMimic值） |
|---|---|---:|---:|
| 1 | 当前root世界高度 | 1 | 1 |
| 2 | 去除当前yaw后的root姿态rot6 | 6 | 1 |
| 3 | 当前heading坐标系中的root线速度 | 3 | 1 |
| 4 | 当前body坐标系中的root角速度 | 3 | 2 |
| 5 | 实际关节绝对位置 | 27 | 1 |
| 6 | 实际关节速度 | 27 | 5 |
| 合计 | | **67** | |

这里的q不减default pose，也不叠加actor的encoder bias；速度仍按0.5/0.2缩放，
本次actor恢复原有效scale后也与它一致。世界模型使用同一canonical表示及原有误差、输出尺度，均未修改。

物理上下文xi为31维：足底摩擦1维、encoder bias 27维、腰部 `waist_yaw_Link` 的COM偏移3维。
按 `StateSpec.normalize_xi()` 做固定变换 `(xi-offset)/scale`：

| 分量 | offset | scale |
|---|---:|---:|
| 足底摩擦 | 0.75 | 0.45 |
| encoder bias | 0 | 0.01 |
| 腰部COM偏移 | 0 | 0.05 |

root误差最后4维为 `[位置误差3, yaw误差1]`：位置取实际root减参考root，再旋转至
**参考yaw坐标系**；yaw误差取实际减参考并wrap到[-pi, pi]。这4维没有额外单位缩放。
其位置坐标系与goal位移所用的“当前参考完整旋转坐标系”不同，保持原critic定义。

critic隐藏层仍为256/128，优化参数不变。没有迁移Isaac的完整body_pos/body_ori特权观测，
也没有为actor、critic或世界模型增加running mean/std。

## 世界模型输入与固定缩放

Khan的动力学 `DynamicsMLP` 每次输入 **125维**，不是actor的620维信息，也不直接读取goal或历史：

```text
DynamicsMLP input = canonical(s_t)[67] + action[27] + normalize_xi(xi)[31]
state target      = 下一步物理状态[qpos34 | qvel33]
```

Replay直接记录仿真器的qpos/qvel，未叠加actor的观测白噪声。canonical特征及输入尺度为：

| 状态分量 | 维度 | 输入除数 | 实际乘数 |
|---|---:|---:|---:|
| root高度 | 1 | 1 | 1 |
| 去yaw姿态rot6 | 6 | 1 | 1 |
| heading坐标系线速度 | 3 | 1 | 1 |
| root/body坐标系角速度 | 3 | 2 | 0.5 |
| 实际关节绝对位置 | 27 | 1 | 1 |
| 实际关节速度 | 27 | 5 | 0.2 |
| 合计 | **67** | | |

动作27维直接拼接归一化动作值，没有在MLP入口再乘机器人动作scale=0.5。
WM拟合使用实际执行的 `a_exec=a+Gaussian(std=0.1)`；控制端再做
`q_target=q_default+0.5*a_exec`。固定执行器/PD映射属于要学习的物理动力学关系，
不能将0.5既在模型入口乘一次、又按原训练动作含义解释一次。

xi沿用上述31维固定标准化：足摩擦 `(mu-0.75)/0.45`、encoder bias `/0.01`、
腰部COM偏移 `/0.05`。encoder bias作为已知上下文保留；它不会被写进物理qpos，
只有解析传感器观测使用 `q+bias-q0`。想象过程对actor观测加白噪声，但对预测物理state不加。

想象时 `wm.step(s_hat,a,xi)` 使用策略输出的clean action，不额外重采执行Gaussian；
这是原G1采用的名义动作rollout近似。观测噪声和动作执行噪声是独立机制，不能据此声称真实采样
与想象包含完全相同的随机扰动。

学习式termination head另外接收4维：`[height_error/0.25, tilt_error/0.8]` 及二者绝对值，
不是将125维动力学输入直接送给终止MLP。原 `StateSpec` 输出增量scale、损失误差scale、
显式积分和学习率均保留；这些固定尺度与PPO的running normalization无关。

## 与Isaac训练仍不同之处

| 项目 | 用户的Isaac训练 | 当前SuperMimic Khan |
|---|---|---|
| actor观测信息 | 按上述顺序合为620维 | 同样155+465的信息，分别输入goal encoder和decoder |
| 算法 / actor结构 | PPO-MNMLP；USAGE隐藏层2048/1024/512 | 可微世界模型＋想象优化；encoder/decoder各2048/1024/512/256，latent32 |
| actor本体有效scale | 角速度0.25、关节速度0.05 | 按SuperMimic恢复为0.5、0.2；其余1 |
| goal参考噪声 | 高度±0.01、参考q±0.05 | 155维全部不加噪，与G1默认参考窗口一致 |
| critic输入 | 无噪声特权观测：command、未来anchor位置/姿态、body位姿、根速度、q/qd/action等 | canonical state67＋干净参考365＋context31＋root error4，共467；隐藏层256/128 |
| running观测归一化 | `empirical_normalization=True`，运行均值/方差，eps=0.01 | 无running mean/std；仅上述单位scale |
| 关节零点随机化 | startup改default q，且同时改action offset | 每回合encoder bias使观测为q+bias−nominal q0；不改控制offset |
| last action | action manager收到的归一化动作，已含PPO采样探索 | 执行额外Gaussian噪声之前的clean动作；执行std仍为0.1 |
| 域随机化及奖励 | Isaac配置含质量/材质等随机化和其他奖励 | 保留本库已有DR及10项奖励；本次只调整观测相关路径 |
| 动作末尾 | 缓冲越界夹末帧；命令到达末尾可重采样 | 未来窗口夹末帧；按本库clip-end终止与reset |
| 仿真 | Isaac Sim/PhysX | mjlab/MuJoCo Warp |

critic保留SuperMimic的canonical state、context和root error结构，参考部分增加5帧速度/重力并保留
干净command和未来anchor朝向；未迁移Isaac完整的body_pos/body_ori等特权观测。
actor的prop仍按上表加入观测噪声，goal保持干净。

Isaac的running归一化由本地 `deps/rsl_rl-modular-normed/rsl_rl/runners/on_policy_runner.py`
及 `modules/normalizer.py` 实现，作用于已拼接、已缩放的观测，统计更新到1e8样本。
它与当前采用的0.5/0.2固定scale不同，本次没有额外迁移该PPO组件。

Isaac和当前环境都使用0.005 s物理步长、4倍decimation、50 Hz控制，并按27个关节构造参考、
传感器观测和动作。**不能据此断言两边运行时关节列序逐字节相同**：Isaac转换脚本先按名字映射到
PhysX articulation，再按 `robot.data.joint_pos` 顺序导出；本地没有其Khan运行时names/ONNX metadata。
当前使用已验证的MJCF/NPZ关节名称顺序，所有输入与动作保持一致，但不支持直接加载Isaac网络权重。

Isaac `last_action` 同样位于动作scale之前；标准RSL-RL wrapper可以额外clip输入动作，
本地参考文件没有提供实际train.py的wrapper参数，因此未将是否启用该clip作为已核实的对齐项。

## 启动与验证边界

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 4096 --run-name khan_lafan1_clean_goal
```

需要新run，不能续训此前14-body、649/870观测、上轮287维critic/Isaac scale，或version-3带goal噪声的checkpoint。旧 `--anchor-obs` /
`--no-anchor-obs` 服务器参数会被拒绝；未来anchor姿态已固定包含在prop内。
原有36段NPZ可继续使用，本次没有修改接地、root高度或机器人资产。

此前151项测试和 `khan_v2_base_head_neck_smoke` 的8环境50步训练属于上一版649/870观测，
不能作为本轮155/465布局的验证结果。历史旧策略全库benchmark和抗推结果也只对应各自冻结版本。

上一轮验证（Isaac原scale、287维critic）的命令与结果：

```bash
env -u PYTHONPATH uv run pytest tests -q
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py \
  --run-name khan_isaac_obs_smoke_final --num-envs 8 --steps 50 --activation-checkpoint
```

该历史版本全量测试 **165 passed**，日志为 `logs/khan_isaac_observations/pytest.log`，覆盖CPU的坐标系、
未来窗口、分量顺序、缩放及噪声检查，以及GPU的实际环境观测、reset历史、想象rollout和可微编译路径。
全量运行出现一次Warp host浮点转换overflow警告；相关两个GPU观测用例以 `-W error` 单独复验均通过，
未复现该警告，也没有通过屏蔽警告或改变噪声设置使测试通过。

该历史版本训练smoke **passed**，使用全部36段动作、8个环境、50步，实际确认goal155、prop465、decoder497、
critic287和15个跟踪body。WM batch2048、policy batch4096、H32、UTD0.4、正式网络/学习率计划均保留；
本机12 GB显存显式开启可选激活重计算，正式服务器入口仍默认关闭。
完成4次外层更新，即40次WM、4次actor、80次critic优化；模型参数确实更新且有限，
实际TensorBoard标量有限，模型、优化器及采样器状态恢复检查通过。
结果为 `logs/khan_isaac_obs_smoke_final/summary.json`，控制台日志为
`logs/khan_isaac_observations/training_smoke_final.log`。

这些历史结果不能替代当前467维critic和SuperMimic scale的验证，也不能据此判断长片动作的稳定性、收敛效果，
或默认关闭激活重计算时4096环境的显存需求。

## 历史 SuperMimic scale / critic467（goal仍加噪）的验证

```bash
env -u PYTHONPATH uv run python -m pytest tests -q
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py \
  --run-name khan_sm_scale_critic467_smoke --num-envs 8 --steps 50 --activation-checkpoint
```

该历史版本完整测试 **167 passed、8 warnings**，日志为
`logs/khan_supermimic_scale_critic/pytest.log`。新增检查包括actor实际/解析观测缩放一致、
噪声先后缩放等价、critic180维的独立坐标变换与拼接、365维参考scale、
干净critic输入、actor/critic梯度、编译、重计算等价及旧schema拒绝。
警告包括已知TorchScript弃用、capsule/mesh的MULTICCD提示，以及一次Warp host浮点转换overflow；
测试没有屏蔽这条警告，实际观测、奖励与训练参数的有限性检查均通过。

额外将该overflow警告提升为异常的全量诊断得到166 passed/1 failed，栈停在
`Simulation._finish_init -> mjwarp.put_data -> _create_array` 的初始化拷贝，早于观测管理器创建；
输入为MuJoCo工作区的float64数据转float32，而非本次Torch观测加噪路径。
加入仅用于记录字段的诊断包装后，同一严格运行又得到167 passed/7 warnings，未复现该转换警告；
初始化未使用工作区与分配状态的关系仍需上游定位，未修改依赖或屏蔽警告。
两份诊断日志为 `warp_warning_diagnostic.log` 和 `warp_field_diagnostic.log`，均在上述验证目录内。

该历史版本训练smoke **passed**，使用全部36段动作、8环境、50控制步，实际维度为goal155、prop465、
encoder155、decoder497、critic467，15个跟踪body。保持正式网络隐藏层、WM batch2048、
policy batch4096、H32、UTD0.4、优化器和学习率计划，完成4次外层更新，即40次WM、
4次actor、80次critic优化。模型参数实际更新且有限，实际TB标量有限；模型、优化器及采样器
状态清空后恢复检查通过。

summary为 `logs/khan_sm_scale_critic467_smoke/summary.json`，控制台日志为
`logs/khan_supermimic_scale_critic/training_smoke.log`。本机12 GB GPU测试显式启用激活重计算，
正式服务器默认仍关闭；该次Torch峰值allocated约2.15 GiB、reserved约2.92 GiB，含冷编译耗时约197.95秒。
这些数值不代表4096环境或默认关闭重计算时的显存需求；短smoke也不能证明新增critic参考改善了最终收敛。

当前version-4的155维goal已去除噪声，维度与网络隐藏层不变；上述version-3结果仅作为历史记录，
不能替代本轮干净goal实现的验证。

## 当前干净goal / version-4的验证

```bash
env -u PYTHONPATH uv run python -m pytest tests -q
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py \
  --run-name khan_clean_goal_smoke --num-envs 8 --steps 50 --activation-checkpoint
```

本轮完整回归为 **168 passed、7 warnings**，日志为 `logs/khan_clean_goal/pytest.log`。
真实环境与想象rollout在obs噪声开启/关闭时都验证155维goal等于干净参考；prop本体与未来朝向
仍按配置加噪，critic参考仍干净。警告仅有TorchScript弃用和capsule/mesh的MULTICCD提示，
本轮未观察到上一版的Warp host cast警告；没有为此屏蔽警告或修改依赖。

最终训练smoke **passed**，使用全部36段动作、8环境、50控制步，实测动力学输入125、goal155、
prop465、decoder497、critic467及15个跟踪body。保留正式网络、WM batch2048、policy batch4096、
H32、UTD0.4、compile和学习率计划；完成4次外层更新，即40次WM、4次actor、80次critic优化。
模型参数确实更新且有限，实际TensorBoard标量有限，模型、优化器和采样器状态清空后恢复通过。

summary为 `logs/khan_clean_goal_smoke/summary.json`，控制台日志为
`logs/khan_clean_goal/training_smoke.log`。本机12 GB测试显式启用激活重计算，正式服务器默认仍关闭。
该次17.82秒包含缓存编译，不能作为正式训练吞吐量或4096环境显存的依据；短smoke也不证明收敛。
服务器默认新run名 `khan_lafan1_clean_goal` 的4096环境dry-run已通过。
