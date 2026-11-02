# Khan_mini 训练适配

## 当前训练入口

在项目根目录执行，默认读取已有全部36条 Khan LAFAN1 NPZ：

```bash
env -u PYTHONPATH uv run sm-train --robot khan_mini \
  --run-name khan_lafan1_delay_convex
```

当前配置为 **15个跟踪body、base anchor、独立腰部COM随机化、直接颈部位置/速度奖励**，
并默认使用 `isaac_khan` 观测：5帧参考、5帧本体历史。当前为Khan contract v5，新增执行器延迟、完整脚凸包、reset高度余量及critic误差特征。
包括version-4 clean-goal/100k在内的旧Khan检查点均不能直接续训、评估或导出，须使用新run。
服务器命令、50步验证及同语义恢复见 [KHAN_SERVER_TRAINING.md](KHAN_SERVER_TRAINING.md)。

`--robot` 默认仍为 `g1`。G1默认数据目录是 `data/npz/lafan1`，Khan是
`data/npz/khan_mini/lafan1`；可通过 `--data.npz-dir` 指定其他目录。加载器验证机器人、
关节名称与顺序、50 Hz、四元数及数组布局。现有NPZ包含45个body，新增头部跟踪不需要重新转换数据。

## 机器人绑定与观测

`robots/registry.py` 分别记录root、anchor、COM body和tracked bodies：

- root与anchor均为 `base_link`。第三方Khan类默认anchor原为腰部，但
  `ref_thirdparty/jaka_code/Textop/USAGE.md` 的实际mini训练命令显式覆盖成 `base_link`；当前按该用法对齐。
- COM随机化及replay context仍绑定 `waist_yaw_Link`，由独立 `RobotSpec.com_body` 指定，不随anchor改变。
- 原14个G1对应部位末尾追加 `Neck_pitch_Link`，共15个；该body的位姿和线/角速度进入原body跟踪奖励。
- 27个关节均保留执行器，动作缩放均为0.5；PD、转子惯量及力矩限制沿用Khan配置。

| 张量或网络接口 | G1，29关节，原默认 | 当前Khan，27关节 |
|---|---:|---:|
| qpos / qvel | 36 / 35 | 34 / 33 |
| state / canonical feature | 71 | 67 |
| dynamics delta | 70 | 66 |
| action | 29 | 27 |
| 每帧本体观测 / 历史帧数 | 93 / 10 | 87 / 5 |
| 参考窗口 | 未来10帧 | 当前参考帧至t+4，共5帧 |
| goal | 680；可选anchor后689 | 155 |
| policy prop输入 | 930 | 465＝未来姿态30＋历史435 |
| context ξ | 33 | 31 |
| tracked bodies | 14 | 15 |
| decoder输入（含latent32） | 962 | 497 |
| critic输入 | 788；可选anchor后797 | **545＝原467＋当前跟踪误差78** |

Khan的goal按项拼接 `[5×root_pos_diff, 5×root_z, 5×joint_pos]`，维度为15+5+135。
位移在**当前参考base完整姿态**下表达，首帧位移为0；root_z是世界高度，关节角是参考绝对角。

policy prop顺序为 `[5帧未来anchor姿态, 5帧历史重力, 5帧历史角速度, 5帧历史q,
5帧历史qd, 5帧历史action]`，维度为30+15+15+135+135+135。未来姿态是参考base相对当前机器人base的
旋转矩阵前两列，按行展平；只对传感器项累积历史，历史按最旧到最新排列。
角速度乘0.5、关节速度乘0.2，与原SuperMimic的有效scale一致；噪声在缩放前加入，网络不重复缩放。
`isaac_khan` 标识特征布局；155维参考goal已按G1默认方式改为始终不加噪声，
不受真实/想象的 `obs_noise` 开关影响。prop仍有本体噪声及未来朝向±0.05，
其中gravity±0.07仍与上游±0.05不同，详见下面链接中的源码核查。
critic输入为canonical state67＋干净参考command155＋干净未来anchor朝向30＋5帧参考速度/重力180＋context31＋root error4，
再在原root error4后拼接当前跟踪误差78，共545维。误差依次为参考减实际关节角27、重锚后body位置误差45
（实际root heading系，15个body）、完整root相对旋转rot6d6；这78维干净且scale为1，旋转完全对齐时为单位旋转6D表示而非零向量。
180维未来参考每帧为qd27、重力3、root系角速度3、heading系线速度3，critic入口分别乘0.2/1/0.5/1。
保留SuperMimic结构，没有迁移Isaac整套特权观测；是否改善收敛需训练对照。
想象历史从replay状态/动作重建后重新采样噪声，reset回填共享首次噪声；不重放真实环境原噪声取值。

`auto` 会为Khan选择 `isaac_khan`，为G1保留 `supermimic`。Isaac布局严格要求future/history=5/5，
原 `--env.anchor-obs` 额外9维不能与其叠加。G1的10帧历史、10帧参考及可选anchor路径保持不变。
公式、噪声表、历史填充和与Isaac训练框架的区别见 [KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。

## 世界模型输入与scale

动力学MLP输入为 **125维＝canonical state67＋动作27＋标准化context31**，不包含actor的goal、
prop、观测历史或额外传感器白噪声。67维canonical state依次为root高度1、去yaw姿态6、heading系线速度3、
root系角速度3、关节绝对位置27、关节速度27；对应输入除数为`1,1,1,2,1,5`，
即角速度实际乘0.5、关节速度乘0.2，其余乘1。

训练动作使用replay保存的 `a_exec=a+Gaussian(std=0.1)`，即执行器延迟队列之前的归一化命令，在模型入口不再乘机器人动作scale=0.5；
控制端的0.5用于PD目标映射 `q_target=q_default+0.5*a_exec`。context沿用固定标准化：
足摩擦`(mu-0.75)/0.45`、27维encoder bias除0.01、3维腰部COM偏移除0.05。
物理状态和下一步状态目标直接来自qpos/qvel，不混入actor观测白噪声。

想象时模型接收clean策略动作，沿用原G1不额外重采执行Gaussian的名义rollout；
预测state经解析观测生成后，prop才单独加入观测噪声，参考goal始终干净。
termination head单独接收高度误差/0.25、倾斜误差/0.8及两者绝对值，共4维；
原状态输出增量scale、损失scale、显式积分与网络隐藏层均未修改。

## 奖励与归一化

G1原8项奖励保持不变。Khan保留它们，并增加两项直接颈部跟踪奖励，共10项：

| 奖励项 | 权重（乘dt之前） | 指数尺度std / 定义 |
|---|---:|---|
| 全局anchor位置 / 朝向 | 0.5 / 0.5 | 0.3 m / 0.4 rad；当前anchor为base |
| 相对body位置 / 朝向 | 1 / 1 | 0.3 m / 0.4 rad；对15个body平均 |
| 世界系body线速度 / 角速度 | 1 / 1 | 1 m/s / 3.14 rad/s；对15个body平均 |
| `motion_neck_pos` | 0.5 | 0.3 rad |
| `motion_neck_vel` | 0.1 | 1 rad/s |
| `action_rate_l2` | −0.1 | clean动作相邻步差值平方和 |
| `joint_limit` | −2 | 超出soft limits的距离和 |

两个颈部项分别对 `Neck_yaw_joint`、`Neck_pitch_joint` 计算：

```text
r_neck_pos = dt * 0.5 * exp(-mean((q_neck - q_ref_neck)^2) / 0.3^2)
r_neck_vel = dt * 0.1 * exp(-mean((qd_neck - qd_ref_neck)^2) / 1.0^2)
```

`mean` 是两个颈关节的均方误差。现有36条数据的颈部q/qd均为0，但实现读取参考值，
并未硬锁关节或移除颈部动作。仅在goal里放入0不等于奖励会监督0，这也是此次显式增加
颈部项及头部body跟踪的原因。仿真与世界模型解析奖励使用同一误差函数和配置；
权重/尺度由 `TrackingRewardCfg.for_joint_names` 选择，不是额外的命令行优化参数。

控制步长为0.02 s，Khan最大单步奖励由0.1变为
`0.02*(5+0.5+0.1)=0.112`；G1仍为0.1。训练回报、benchmark及轨迹录制动态归一化：

```text
normalized_return = sum(rewards) / (max_step_reward * executed_steps)
```

`train_ret` 是上述逐回合归一化回报的均值，理论上仍不超过 **1.0**，不是1.12；
分母随正奖励总上界一起增加。这里没有将结果clamp到[0,1]，处罚足够大时可以为负，
浮点误差也可能导致极小的越界。奖励及actor/critic实际优化仍使用原始dt缩放奖励，没有整体除以0.112。

采样器保存折扣return-to-go，与日志的按步平均归一化是两件事。gamma=0.99时，
`Gmax=0.112/(1-0.99)=11.2`（G1仍为10），困难度分布为
`softmax(-25 * bin_return / 11.2)`；当前起始bin采样器的初始bin return为0，11.2仅作为Gmax尺度。
温度25以及50%均匀、40%困难度、10%记忆混合比例保持原值。
新旧Khan的奖励项和body集合不同，不能直接按回报数值比较策略优劣。

root高度/倾斜终止仍作用于 `base_link`，阈值分别为相对参考高度偏差0.25 m及投影重力z分量差0.8，
后者不是0.8 rad；clip-end和time-out处理不变。改变anchor也会改变全局anchor奖励及body重锚坐标，
并非只替换观测中的一个body名字。

## 域随机化、仿真与网络参数

足底摩擦每回合共享采样 `[0.3,1.2]`，encoder bias每关节 `[-0.01,0.01]`；
startup COM偏移仍加在腰部，x为±0.025 m，y/z为±0.05 m。重置与动作执行噪声保持原配置，
执行噪声std=0.1。非足底摩擦1.0，足底名义摩擦0.6；评估关闭噪声/DR并将足底固定为1.0。

Khan默认启用类似MorphMimic的执行器延迟：每个环境、每个融合执行器组在episode reset时
从0/1/2/3/4个物理步均匀采样，即0/5/10/15/20 ms，一回合保持不变；每5 ms推进目标队列。
所有Khan builtin position执行器使用同一延迟配置并融合共享延迟组，不为每个关节独立采样。
`domain_randomization=False` 的nominal eval关闭延迟。清零行为保留MorphMimic方式：reset时pending target队列为
**物理关节目标0**，没有用参考关节角或q_default预填，因此初始延迟子步可能读取零目标。

ξ保持31维，WM动力学输入保持125维，不加入delay参数或历史队列；想象rollout也没有显式延迟队列。
replay的a_exec记录当前带执行噪声命令，不是每个5 ms子步实际取出的历史目标。
因此这是按用户选择的隐式拟合近似，未显式恢复延迟系统的完整Markov状态；不能认为WM精确模拟了延迟。

Khan运动跟踪环境的所有reset默认将**实际机器人root z加0.05 m**，包含训练、play和nominal benchmark；
G1默认仍为0。此项不受DR/noise开关影响，NPZ、参考reward/goal和后续参考轨迹均不抬升。
这是起始接触余量，真实机器人与参考初始高度因此相差5 cm；只缓解常态穿地，不能修复约25 cm的严重参考帧。
可通过 `--env.reset-root-z-offset` 明确调整，配置写入contract；运动学NPZ replay不施加该offset。

物理步长0.005 s、decimation=4、50 Hz控制、iterations=10、ls_iterations=20均不变。
Khan仍使用保守容量128/512；它们不是测得的最低必要值，也未证明G1的35/250不足。
腰部碰撞盒当前全尺寸0.15×0.14×0.30 m，模型近似及重建记录见 [KHAN_MINI.md](KHAN_MINI.md)。

| 网络或优化项目 | 沿用原仓库的单GPU正式值 |
|---|---|
| WM hidden / termination hidden | 2048×3 / 64×2 |
| 策略encoder / decoder hidden | 各 `(2048,1024,512,256)` |
| latent / critic hidden | 32 / `(256,128)` |
| WM batch / window / 每轮批数 | 2048 / 8 / 10 |
| policy batch / horizon | 4096 / 32 |
| critic | 5 epochs × 4 minibatches |
| UTD / policy:WM | 0.4 / 1:1 |
| warmup / replay capacity | 40控制步 / 每环境1000行 |
| WM、actor、critic LR | 均2e-4 |
| gamma / lambda | 0.99 / 0.95 |
| torch.compile / 激活重计算 | `default` / 默认关闭 |
| 总步数 / LR衰减 | 320000 / 250000步余弦衰减至0.03× |

## 初步训练与显存

```bash
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py
```

该入口使用32环境、全部36段动作、50个控制步，保留正式网络、batch、H和更新比例。
`--schedule.stop-after-iterations 50` 仅限制本次执行，不缩短学习率计划；续训时相对恢复点计数。
训练链路测试检查有限损失/梯度、模型参数变化、实际优化次数、日志及恢复，不能据此判断收敛。

此前14-body配置在本机12 GB RTX 4070 SUPER上，原始执行路径即使降到1个环境仍在首次actor反传OOM；
通过的32环境50步测试显式开启了可选激活重计算。历史记录见 [KHAN_MEMORY.md](KHAN_MEMORY.md)，
不能视为当前155/465观测版本的显存实测。当前正式入口仍默认关闭重计算。
若明确选择该资源优化，smoke加 `--activation-checkpoint`，直接训练加
`--policy-trainer.activation-checkpoint`；它重算完整batch中间激活，没有缩网络、batch或H。

## 检查点与评估

新检查点保存 `robot_contract`：root、anchor、COM body、tracked bodies、关节顺序、动作缩放、
奖励配置，以及Khan `robot.xml` / `profile.json` 的SHA-256。恢复和评估在加载权重前验证这些语义。
Khan contract格式version=5，记录干净goal、observation layout、未来偏移、拼接顺序、坐标系、prop噪声尺度、critic参考与78维误差扩展，以及延迟/reset/脚凸包语义；
G1 contract仍为version=1。profile的 `training_semantics_version=2` 是另一项资产语义标记。

**旧Khan检查点缺少contract或采用旧观测contract都会被拒绝。**
此前649/870输入与当前155/465输入不同；上轮同为155/465但采用Isaac scale、287维critic，
或version-3带goal噪声、version-4 clean-goal/467维critic的检查点也不兼容。
不要通过补字段、关闭anchor或跳过校验强行加载。
当前配置必须新run；以后仅能恢复相同contract的检查点。历史G1无contract检查点仍按原规则兼容。

```bash
env -u PYTHONPATH uv run python scripts/play.py \
  --ckpt logs/khan_lafan1_delay_convex/ckpt_latest.pt --clips walk1_subject1

env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/khan_lafan1_delay_convex/ckpt_latest.pt --num-envs 36
```

旧策略的36段全库benchmark及隔离运行命令见
[KHAN_LEGACY_BENCHMARK.md](KHAN_LEGACY_BENCHMARK.md)；40k录像与抗推另使用18 cm腰盒快照，见
[KHAN_40K_ROBUSTNESS.md](KHAN_40K_ROBUSTNESS.md)。其结果不代表本轮新配置已经完成正式训练或获得相同性能。

已有NPZ可继续使用。compact缓存只打包root/q/qd参考和布局元数据，tracked bodies在运行时按当前配置做FK；
已有缓存若robot、关节顺序、root及fps元数据正确，且动作数据未变，无需仅因新增头部跟踪而重新打包。
下面是需要新建或独立核对缓存时的命令：

```bash
env -u PYTHONPATH uv run python scripts/data/build_library.py \
  --robot khan_mini --npz-dir data/npz/khan_mini/lafan1 \
  --out data/library/khan_isaac_obs_compact.pt

env -u PYTHONPATH uv run sm-train --robot khan_mini \
  --run-name khan_isaac_obs_compact --data.library data/library/khan_isaac_obs_compact.pt
```

## 接地与验证边界

本轮没有修正NPZ的root高度、腿部姿态或接地；仅增加实际机器人reset的5 cm余量。
历史7-capsule足部审计中至少一足碰撞底低于地面1 mm的帧占97.0279%，最深约92.90 mm；
它不是新凸包的重测统计。参考高度仍保留，风险与重播见 [KHAN_REPLAY.md](KHAN_REPLAY.md)。
新增头部跟踪、颈奖励及anchor修改不等于完成了接地修复。

**历史验证（上一版649维goal/870维本体输入）：** `pytest tests -q` 为151 passed，包含15-body FK、10项仿真/解析奖励、
base-anchor观测、waist COM及contract拒绝路径。结果保存于
`logs/khan_waist_collision_adjustment/training_v2_tests.log`。

该历史版本另完成正式网络与优化参数的50步训练验证；以下是当时命令记录，不是当前默认观测的测试结果：

```bash
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py \
  --run-name khan_v2_base_head_neck_smoke --num-envs 8 --steps 50 --activation-checkpoint
```

记录为 `logs/khan_v2_base_head_neck_smoke/summary.json`：4次外层更新、40次WM更新、
4次actor更新、80次critic更新，模型/梯度/日志有限，模型、优化器与采样器状态清空后恢复校验通过。
WM batch2048、policy batch4096、H32及所有正式网络和学习率计划均保留；
为本机12 GB显存显式开启激活重计算，默认训练仍关闭。该次Torch峰值allocated约2.77 GiB、
reserved约3.41 GiB，不能据此推断默认关闭重计算或4096环境时的显存。
上一轮155/465观测、Isaac原scale的版本通过165项全量测试，以及critic287维的8环境50步正式网络/batch训练smoke。
同样完成40次WM、4次actor、80次critic优化，模型参数与实际TB标量有限，恢复检查通过；
本机12 GB测试显式开启激活重计算，正式默认仍关闭。命令、日志和一次Warp警告的复验说明见
[KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。这些历史结果不替代当时467维critic/SuperMimic scale验证，也不代表当前v5验证；
数值一致性和短训练均不构成长片动作稳定性的证明。

上一版SuperMimic scale、467维critic、goal仍加噪的版本通过 **167项全量测试**，并完成8环境50控制步训练smoke：
goal155、prop465、decoder497、critic467实测匹配，40次WM、4次actor、80次critic更新与
模型/优化器/采样器恢复通过；实际参数和TB标量均有限。保留正式隐藏层、batch、H和UTD；
本机12 GB测试显式开启激活重计算，服务器默认关闭。历史命令、结果及8条警告的说明见
[KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)中的对应历史验证段；它不替代当前干净goal版本的验证。

历史version-4干净goal版本通过 **168项全量测试**，并完成8环境、50控制步训练smoke。
动力学输入125、goal155、prop465、decoder497、critic467和15-body均实测匹配；
正式网络、batch2048/4096、H32、UTD0.4及学习率计划保持，完成40次WM、4次actor、80次critic优化。
参数与实际TB标量有限，模型/优化器/采样器清空后恢复通过。本机12 GB测试显式开启激活重计算，
服务器默认关闭。结果为 `logs/khan_clean_goal_smoke/summary.json`，完整命令和边界见上述观测文档的当前验证段。

## 当前v5验证记录

2026-11-02 当前v5通过 **269项测试、1项跳过**，包括真实CUDA fused actuator延迟、episode固定lag与部分reset，
两个reset入口实际root抬升且reference不变、脚凸包/FK一致性、critic误差及末状态梯度、契约拒绝和ONNX数值一致性。
全量命令为 `.venv/bin/python -m pytest -q`，本机结果保存在 `logs/khan_v5_pytest_20261102.log`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python -u scripts/smoke_train_khan.py \
  --run-name khan_delay_convex_smoke_20261102 --num-envs 8 --steps 50 --activation-checkpoint
```

全部36段LAFAN、396990帧载入成功；实测state67/action27/ξ31/WM输入125、goal155/prop465/decoder497/critic545、15跟踪body。
保留正式隐藏层、WM batch2048、policy batch4096、H32、UTD0.4及320k学习率计划，完成40次WM、4次actor、80次critic优化。
模型参数、实际TB标量有限，模型/优化器/采样器状态清空再恢复校验通过。结果为
`logs/khan_delay_convex_smoke_20261102/summary.json`，控制台日志为同级 `khan_delay_convex_smoke_20261102_console.log`。
控制台nse8为nan表示此时buffer未达原在线评估2048行门槛，该指标未计算；实际写入的TB指标均有限。
本机12 GB测试显式开启激活重计算，耗时19.60秒（缓存编译），Torch峰值allocated约2.29 GiB、reserved约3.28 GiB；
服务器默认仍关闭。该结果不代表正式4096环境显存、收敛、抗推或实机稳定性。

新脚凸包对36段LAFAN全部396990帧的CPU几何审计：原参考最低−94.84 mm，中位−39.43 mm，
至少一脚低于−1 mm占97.2130%；假设各帧作为reset姿态且实际root刚性抬高50 mm，
最低变为−44.84 mm、中位+10.57 mm、低于−1 mm占6.0944%。这支持保留5 cm初始余量，
但仍有穿地帧；它不是动态轨迹结果，也不是训练实际采样比例。原NPZ和随后的参考目标未加高。
报告为 `logs/khan_foot_convex_review/lafan_whole_foot_clearance_reset_lift.json`。

正式训练须使用新run `khan_lafan1_delay_convex`；上述历史测试/旧smoke属于各自冻结配置。
