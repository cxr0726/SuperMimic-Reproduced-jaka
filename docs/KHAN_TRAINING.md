# Khan_mini 训练适配

## 当前训练入口

在项目根目录执行，默认读取已有全部36条 Khan LAFAN1 NPZ：

```bash
env -u PYTHONPATH uv run sm-train --robot khan_mini --env.anchor-obs \
  --run-name khan_lafan1_base_neck
```

当前配置为 **15个跟踪body、base anchor、独立腰部COM随机化、直接颈部位置/速度奖励**。
它改变了此前14-body/waist-anchor训练的语义，必须使用新run，不能直接续训旧Khan检查点。
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

| 张量或网络接口 | G1，29关节 | 当前Khan，27关节 |
|---|---:|---:|
| qpos / qvel | 36 / 35 | 34 / 33 |
| state / canonical feature | 71 | 67 |
| dynamics delta | 70 | 66 |
| action | 29 | 27 |
| 单帧本体观测 / 10帧历史 | 93 / 930 | 87 / 870 |
| 每帧参考goal / 未来10帧 | 68 / 680 | 64 / 640 |
| 开启anchor后的完整goal | 689 | 649 |
| context ξ | 33 | 31 |
| tracked bodies | 14 | 15 |
| 开启anchor后的decoder / critic输入 | 962 / 797 | 902 / 751 |

Khan本体历史顺序与原实现相同，均为10帧term-major布局：关节位置270维、关节速度270维、
基座角速度30维、投影重力30维、上次clean action270维，合计870。

每个未来参考帧为关节位置27、关节速度27、重力3、角速度3、heading线速度3、高度1，共64维。
额外anchor仅在整个640维窗口末尾追加一次：当前参考base在机器人base坐标系下的位置3维与姿态6维。
**不是未来10帧各追加9维**。anchor位置噪声仍为±0.25 m、姿态6D分量噪声±0.05。
从旧waist anchor改为base时，原640维参考内容与所有接口宽度不变，但末尾9维坐标语义改变。

服务器及smoke入口默认开启anchor；通用 `TrainConfig` 和G1默认仍关闭。
直接调用 `sm-train` 需明确传 `--env.anchor-obs`；关闭时goal为640维。

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

回报采样器 `Gmax` 同比由10变为11.2，保持相对回报尺度；softmax温度25以及50%均匀、
40%困难度、10%记忆混合比例不变。新旧Khan的奖励项和body集合不同，不能直接按回报数值比较策略优劣。

root高度/倾斜终止仍作用于 `base_link`，阈值分别为相对参考高度偏差0.25 m及投影重力z分量差0.8，
后者不是0.8 rad；clip-end和time-out处理不变。改变anchor也会改变全局anchor奖励及body重锚坐标，
并非只替换观测中的一个body名字。

## 域随机化、仿真与网络参数

足底摩擦每回合共享采样 `[0.3,1.2]`，encoder bias每关节 `[-0.01,0.01]`；
startup COM偏移仍加在腰部，x为±0.025 m，y/z为±0.05 m。重置与动作执行噪声保持原配置，
执行噪声std=0.1。非足底摩擦1.0，足底名义摩擦0.6；评估关闭噪声/DR并将足底固定为1.0。

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
不能视为本轮15-body配置的显存实测。当前正式入口仍默认关闭重计算。
若明确选择该资源优化，smoke加 `--activation-checkpoint`，直接训练加
`--policy-trainer.activation-checkpoint`；它重算完整batch中间激活，没有缩网络、batch或H。

## 检查点与评估

新检查点保存 `robot_contract`：root、anchor、COM body、tracked bodies、关节顺序、动作缩放、
奖励配置，以及Khan `robot.xml` / `profile.json` 的SHA-256。恢复和评估在加载权重前验证这些语义。
contract格式version=1；profile的 `training_semantics_version=2` 是另一项资产语义标记。

**旧Khan检查点缺少contract会被拒绝，即使goal同样是649维也不能直接加载。**
不要通过补字段、关闭anchor或跳过校验把旧14-body/waist策略当成新配置策略。
当前配置必须新run；以后仅能恢复相同contract的检查点。历史G1无contract检查点仍按原规则兼容。

```bash
env -u PYTHONPATH uv run python scripts/play.py \
  --ckpt logs/khan_lafan1_base_neck/ckpt_latest.pt --clips walk1_subject1

env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/khan_lafan1_base_neck/ckpt_latest.pt --num-envs 36
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
  --out data/library/khan_base_neck_compact.pt

env -u PYTHONPATH uv run sm-train --robot khan_mini --env.anchor-obs \
  --run-name khan_base_neck_compact --data.library data/library/khan_base_neck_compact.pt
```

## 接地与验证边界

本轮没有修正NPZ的root高度、腿部姿态或接地。已有审计中至少一足碰撞底低于地面1 mm的帧占97.0279%，
最深约92.90 mm；参考高度仍保留，相关风险与重播见 [KHAN_REPLAY.md](KHAN_REPLAY.md)。
新增头部跟踪、颈奖励及anchor修改不等于完成了接地修复。

本轮完整回归 `pytest tests -q` 为 **151 passed**，包含15-body FK、10项仿真/解析奖励、
base-anchor观测、waist COM及contract拒绝路径。结果保存于
`logs/khan_waist_collision_adjustment/training_v2_tests.log`。

新配置另完成正式网络与优化参数的50步训练验证：

```bash
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py \
  --run-name khan_v2_base_head_neck_smoke --num-envs 8 --steps 50 --activation-checkpoint
```

记录为 `logs/khan_v2_base_head_neck_smoke/summary.json`：4次外层更新、40次WM更新、
4次actor更新、80次critic更新，模型/梯度/日志有限，模型、优化器与采样器状态清空后恢复校验通过。
WM batch2048、policy batch4096、H32及所有正式网络和学习率计划均保留；
为本机12 GB显存显式开启激活重计算，默认训练仍关闭。该次Torch峰值allocated约2.77 GiB、
reserved约3.41 GiB，不能据此推断默认关闭重计算或4096环境时的显存。
数值一致性测试与短训练不构成长片动作稳定性的证明。
