# Khan_mini 训练适配

## 训练入口

在项目根目录执行，默认读取已转换的全部 36 个 Khan LAFAN1 NPZ：

```bash
env -u PYTHONPATH uv run sm-train --robot khan_mini --env.anchor-obs --run-name khan_lafan1_anchor
```

服务器的固定原参数启动脚本、部署包、dry-run与50步验证/恢复命令见
[KHAN_SERVER_TRAINING.md](KHAN_SERVER_TRAINING.md)。

`--robot` 默认 `g1`，保持原有训练入口。`--data.npz-dir` 不指定时，G1 使用
`data/npz/lafan1`，Khan 使用 `data/npz/khan_mini/lafan1`。也可显式传入目录；
加载器会校验机器人、全部关节名称和顺序、50 Hz、四元数和数组布局，防止串用数据。

## 对齐范围

`robots/registry.py` 统一选择模型、EntityCfg、动作缩放、14 个跟踪 body、根节点、锚点、
足底碰撞匹配和默认数据目录。Khan 根节点为 `base_link`，锚点和 COM 随机化 body
为 `waist_yaw_Link`；跟踪 body 是 G1 14 个部位对应的 Khan 部位，不额外加入头颈奖励。

| 张量 | G1，29 关节 | Khan，27 关节 |
|---|---:|---:|
| qpos / qvel | 36 / 35 | 34 / 33 |
| 状态 / canonical feature | 71 | 67 |
| dynamics delta | 70 | 66 |
| action | 29 | 27 |
| 单帧本体观测 | 93 | 87 |
| 10 帧本体观测 | 930 | 870 |
| 单帧参考 goal / 10 帧参考 goal | 68 / 680 | 64 / 640 |
| 当前Khan启动配置的完整goal | 原G1默认680；开启anchor为689 | 649（640+9） |
| 域随机化 context | 33 | 31 |
| 跟踪 body | 14 | 14 |

本体观测项、顺序、10 帧 term-major 历史和噪声保持原样：关节位置、关节速度、基座角速度、
投影重力、上次 clean action。goal 保持未来 10 帧的关节位置/速度、重力、角速度、
heading 线速度和高度。当前Khan服务器脚本与smoke默认开启anchor observation，
在整个640维参考窗口末尾追加当前位置的腰部相对目标位置3维和相对目标姿态6维，得到649维。
不是每个未来参考帧都加9维。本体观测仍为870维，decoder输入仍为902维；critic输入从742变成751维。
这些相对位姿用机器人 `waist_yaw_Link` 坐标系表达；传感器噪声沿用位置±0.25、姿态6D分量±0.05。
通用 `TrainConfig` 的G1默认仍为 `anchor_obs=False`；直接调用 `sm-train` 时需显式传 `--env.anchor-obs`。
这项观测增强参考迁移仓库的实际训练命令，不属于原始纯本体观测口径；效果仍需服务器训练验证。
当前版本全套测试132项通过，包含真实CUDA anchor观测/噪声和正式hidden网络反传；
后文110项及50步训练结果为开启anchor之前的历史记录。

8 项奖励、权重、尺度、动作变化和软关节限位惩罚保持原值；root height/tilt、clip-end
与 time-out 终止方式保持原值。根位置/姿态终止作用于 Khan 根，anchor 奖励作用于 Khan 腰部，
与 G1 中 root/anchor 的职责一致。

域随机化保持 G1：足底滑动摩擦每回合共享随机采样 `[0.3,1.2]`，encoder bias
每关节 `[-0.01,0.01]`；腰部 COM 在 startup 按 x `±0.025 m`、y/z `±0.05 m` 随机化。
重置位置、姿态、速度和关节噪声，以及动作执行噪声 `std=0.1` 均未改变。
非足底名义摩擦为 1.0；全部动作 scale=0.5，PD/惯量/限位等继续使用既有 Khan 模型配置。

仿真仍为 timestep=0.005 s、decimation=4、控制 50 Hz、iterations=10、ls_iterations=20。
Khan 的 nconmax=128、njmax=512 是迁移时保守预留的接触/约束容量，不是测试得到的最低必要值，
尚未证明 G1 的35/250对Khan不足。nconmax按环境数乘出一个共享接触池，njmax是每环境的约束行
上限；二者不是关节数、求解器迭代或奖励参数。容量足够后，继续增大不会提高碰撞精度。

状态 I/O、terminal replay 行、MotionLibrary full/compact 模式、解析 FK/奖励/观测头、
世界模型、策略与 critic 均按机器人维度构造。固定焊接 link 仍参与 FK，全部 45 个 Khan body
与 MuJoCo 运动学一致。检查点保存机器人选择，恢复时拒绝机器人不一致的检查点。

## 正式参数与初步训练

```bash
env -u PYTHONPATH uv run python scripts/smoke_train_khan.py
```

该脚本将并行环境从 4096 降到 32，并将本次执行限制为 50 个控制步。
使用全部 36 段、396,990 帧数据，不更改原始根高度。优化参数自动对照 `TrainConfig()`；
如果隐藏层、batch、更新等被缩小或关闭，脚本会报错；anchor是用户要求的明确观测差异。

按后续“只改变并行环境数”的要求，脚本现默认关闭激活重计算，保持原始训练执行路径。
本机12 GB RTX 4070 SUPER实测1024和1个环境都在第42步第一次actor更新时OOM；
此前32环境的原始路径也失败。固定batch=4096、H=32的想象反传图是主要显存瓶颈，
仅降低采集环境数未能解决。完整结果见 [KHAN_MEMORY.md](KHAN_MEMORY.md)。

先前通过的50步测试显式开启了本次新增的可选激活重计算。使用 `--activation-checkpoint`
可重现该资源优化：保持同一个完整 batch、H=32 和相同目标，在反向时重算
policy/ImagStep 中间激活，以计算量换显存。使用 `use_reentrant=False` 并保存 RNG 状态，
随机噪声仍在重计算块外生成；没有微批或梯度累积，也未缩小网络和 batch。
该资源开关不属于优化超参数；生产入口与smoke现在均默认关闭，显式选择该优化时，
生产入口加 `--policy-trainer.activation-checkpoint`。本轮环境数测试没有开启它。

| 项目 | 原仓库正式默认值，测试同值 |
|---|---|
| 世界模型 hidden / termination hidden | 2048×3 / 64×2 |
| 策略 encoder / decoder hidden | 各 `(2048,1024,512,256)` |
| latent / critic hidden | 32 / `(256,128)` |
| 世界模型训练 | batch=2048，window=8，每轮 10 batches |
| 策略想象 | batch=4096，H=32 |
| critic | 5 epochs × 4 minibatches |
| UTD / policy:WM | 0.4 / 1:1 |
| warmup | 40 控制步 |
| 学习率 | WM、actor、critic 均 2e-4 |
| gamma / lambda | 0.99 / 0.95 |
| torch.compile | 两个训练分支均为 `default` |
| replay capacity | 每环境 1000 行 |
| 学习率时间基准 | 总 320,000 步，250,000 步余弦衰减至 0.03× |

`--schedule.stop-after-iterations 50` 是本次执行预算，不修改 `total_iterations` 和学习率
衰减时间基准；续训时从恢复点再执行指定数量。默认 None 表示按正式训练长度运行。
等价的基础入口为：

```bash
env -u PYTHONPATH uv run sm-train --robot khan_mini --env.anchor-obs --run-name khan_anchor_short \
  --env.num-envs 32 --schedule.stop-after-iterations 50
```

初步测试验证三个模型参数变化、有限损失和梯度、实际优化器更新次数、TensorBoard、
检查点权重/优化器/采样器恢复。该测试证明完整训练链路可执行，不用于判断收敛或动作质量。

## 历史本地实测记录（anchor关闭）

2026-10-31，RTX 4070 SUPER 12 GB，全部 36 段 / 396,990 帧数据，32 个环境，50 个控制步，
开启激活重计算，正式 batch、H、compile 与更新设置均未缩小：

以下记录来自本次开启anchor之前的640维goal配置，不能视为649维配置的正式收敛结果。

| 检查 | 结果 |
|---|---|
| 外层训练更新 | 4 次 |
| 世界模型 Adam 更新 | 每参数 40 次 |
| actor Adam 更新 | 每参数 4 次 |
| critic Adam 更新 | 每参数 80 次 |
| 模型变化 | WM 14 / actor 20 / critic 6 个参数张量改变，全部有限 |
| TensorBoard | 所有已记录标量有限 |
| 最后记录的 WM loss / critic loss | 43.27605 / 0.01008439 |
| actor grad norm | 4.23928e-6，非零 |
| 学习率 | 0.0001999999804，仍按正式 250k 周期衰减 |
| Torch CUDA 峰值 allocated / reserved | 2.39 / 3.07 GiB（不含 Warp/驱动内存） |
| 总耗时 | 243.85 s，包含首次策略计算图编译 |

记录在 `logs/khan_formal_smoke_checkpoint/summary.json`，检查点为同目录 `ckpt_latest.pt`。
初次未开启重计算的尝试在策略图保存激活时 OOM，记录保留于
`logs/khan_formal_smoke/summary.json`，没有通过缩 batch 隐藏该限制。

独立 CPU 恢复验证直接调用真实 `Trainer.load`，先清空三个优化器全部 state 并将学习率改为
123，再扰动三个模型和 `bin_return/bin_memory`。恢复后逐 tensor 和参数组严格一致，
Adam 步数仍为 40/4/80，报告为 `logs/khan_training/checkpoint_restore.json`。
smoke 脚本也已内置该严格验证，更新次数使用预热/UTD 独立计算预期值。

测试还包括实际 Khan CUDA 环境的观测历史、8 项解析奖励与 mjlab 奖励逐项对齐、
DR、重置噪声、终止、状态 I/O 和 replay 采样；激活重计算有包含真实 Khan FK 的
CPU 对照测试，覆盖 anchor 开关、噪声、目标、梯度、Adam 更新和 RNG 一致性。

完整测试集（含原 G1 训练/恢复回归）实跑 **110 passed，3 warnings，207.89 s**，
没有失败或跳过。日志和 JUnit 分别为 `logs/khan_training/all_tests.log` 和 `all_tests.xml`。
第三方提示为 TorchScript 弃用及 capsule/mesh 配对只支持单接触点。

另用生成的真实检查点走 `scripts/eval_benchmark.py` 的自动 Khan 选择和数据路径：
`walk1_subject1` 能正常加载/执行，61 步触发训练终止，归一化回报 0.49387。
这只有 4 次 actor 更新，尚未学会完成动作；报告为 `logs/khan_training/checkpoint_eval.json`。

控制台 `nse8 nan` 是小缓冲区不足 2048 行时跳过 online-error 诊断的占位显示，
没有作为非有限 TensorBoard 标量写入；实际 WM 训练损失、梯度和参数均有限。

## 训练后使用

```bash
env -u PYTHONPATH uv run python scripts/play.py \
  --ckpt logs/khan_lafan1_anchor/ckpt_latest.pt --clips walk1_subject1

env -u PYTHONPATH uv run python scripts/eval_benchmark.py \
  --ckpt logs/khan_lafan1_anchor/ckpt_latest.pt --num-envs 32
```

上述入口从检查点读取机器人选择、history 和 future_frames。旧 G1 检查点缺少机器人字段时
仍按 G1 解释。旧640维Khan检查点不能直接恢复到649维训练；需要新run，
或明确关闭anchor以复现旧配置。加载前会检查anchor、future_frames、history和goal布局。
可选紧凑缓存：

```bash
env -u PYTHONPATH uv run python scripts/data/build_library.py \
  --robot khan_mini --npz-dir data/npz/khan_mini/lafan1 \
  --out data/library/khan_lafan1_compact.pt

env -u PYTHONPATH uv run sm-train --robot khan_mini --env.anchor-obs --run-name khan_compact_anchor \
  --data.library data/library/khan_lafan1_compact.pt
```

## 接地偏差

现有数据的穿地审计结果没有被训练适配掩盖或修正：至少一脚碰撞体低于地面 1 mm 的帧
占 97.0279%，最深约 92.90 mm。仿真会据当前模型处理初始接触，参考 FK 仍保留数据高度；
可能影响后续收敛和动作质量。相关证据与运动学重播命令见 [KHAN_REPLAY.md](KHAN_REPLAY.md)。
