# Khan 服务器训练：Isaac观测布局、SuperMimic尺度与原优化参数

本文件对应 `scripts/training/train_khan_server.sh`。脚本采用原仓库的单GPU LAFAN
网络隐藏层与优化配置，选择Khan模型/动作；默认4096并行环境，采用 `isaac_khan` 观测。
不启用激活重计算、AMP、微批或梯度累积，不缩小训练batch和想象H。

脚本复用 `TrainConfig` 的优化默认值，显式传入机器人、路径、环境数及Isaac观测布局和5/5帧数。
网络隐藏层、学习率、batch、H及UTD不变；当前Khan另有明确的机器人适配：

- 15个tracked bodies，末尾新增 `Neck_pitch_Link`；root/anchor均为 `base_link`。
- COM随机化及replay ξ仍绑定 `waist_yaw_Link`，与anchor解耦。
- 新增颈部位置/速度跟踪，权重0.5/0.1、尺度0.3 rad/1 rad/s；仿真与WM解析奖励同源。
- 最大单步奖励0.112，训练/eval/record使用动态归一化，采样 `Gmax=11.2`；G1仍为0.1/10。
- 腰部box全尺寸0.15×0.14×0.30 m；质量、惯量及执行器参数不变。
- goal为5帧参考位移/高度/关节角，共155维，真实环境与想象训练均不加噪声；
  prop为未来姿态30＋5帧传感器历史435，共465维，保留本体和未来朝向噪声。

decoder输入为497、critic输入为467；state67、action27、context31保持不变。
actor本体角速度/关节速度有效scale为0.5/0.2，恢复原SuperMimic值；prop噪声保留此前范围，参考goal保持干净。
critic除原canonical state/context/root error外，接收干净command155、未来朝向30和5帧参考速度/重力180。
新增180维按帧排列，每帧为qd27、重力3、root系角速度3、heading系线速度3，网络入口乘数为0.2/1/0.5/1。
原末尾9维anchor goal由prop前部5帧未来姿态替代。**此前649/870、287维critic或version-3带goal噪声的检查点均不兼容，必须新run。**
观测公式、噪声及与Isaac PPO的差异见 [KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。
完整奖励公式及对齐范围见 [KHAN_TRAINING.md](KHAN_TRAINING.md)。
世界模型动力学输入为canonical state67＋实际执行动作27＋标准化context31，共125维；
状态输入除数为1/1/1/2/1/5，动作在MLP入口不再乘0.5，详见该训练文档的世界模型输入段。

## 35/250与参考仓库的来源

能核实的历史是：

1. 本仓库首个提交 `afda1f4a7e29f8d47fdb831b90ff408ea98de685` 已直接使用
   `nconmax=35, njmax=250`，该版本 `tracking_env_cfg.py:237`；未见本仓库计算或调优过程。
2. 锁定mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376` 的通用tracking模板也使用
   同样容量和timestep/solver配置，本仓库沿用了该模板。
3. mjlab提交 `1bf8d1d58bac60d96d140862940de2451afbb652`
   （`Upgrade to mujoco_warp fb9bf88.`）将tracking的 `nconmax=150_000` 改成35，
   同一提交明确了nconmax按world预算共享接触池、njmax为每world约束上限的语义。
   `njmax=250` 更早已存在于mjlab最初公开提交 `a61e0ff`。
4. 公开历史没有给出为什么精确选35、250的峰值统计、容量扫描或公式。它们是上游配置预算，
   不能说由G1关节数推导，也不能称为最小安全值。

例如 `35*4096=143360` 与旧150000的规模接近，但没有文字证据证明35就是这样取整出来的。

用户提供的 `ref_thirdparty/supermimic-retargeted` 在
`src/supermimic/envs/tracking_env_cfg.py:252` 统一设置：

| 机器人 | nconmax | njmax |
|---|---:|---:|
| G1 | 35 | 250 |
| A3 | 256 | 1024 |
| X2 Ultra | 256 | 1024 |

这两个非G1模型使用同一组常量，公开代码/提交说明/文档没有容量测量依据。
该仓库完整训练日志未公开，不能断言作者从未测试；也不能把256/1024当成Khan的最低安全值。
参考A3/X2还关闭了自碰撞，而当前Khan开启，因此容量不能仅按型号直接照搬。

当前Khan保持128/512，明确属于本次迁移的保守预留，未证明35/250不足或128/512最小。
这与网络/优化超参数分开记录；没有再次无依据地改成256/1024。

## 需要同步到服务器的文件

部署前同步包含 `isaac_khan` 布局、SuperMimic actor尺度、467维critic及Khan version-4 contract的源码和当前机器人资产。
若使用Git，确认服务器已检出包含这些改动的版本；先前14-body或15-body的649/870观测版本不能代表当前训练配置。
也可同步工作区或重新生成部署包。
必须包含 `src/`、`scripts/`、`assets/robots/khan_mini/`、`pyproject.toml`、`uv.lock`、
`README.md` 和 `.python-version`。

旧 `dist/khan_server_code_20261031.tar.gz` 是历史14-body交付，不能直接当作本轮新配置部署包。
新包应包含生成的Khan资产并记录SHA，不包含数据、`.venv`、Git目录、运行日志或ref。

数据需另外同步整个 `data/npz/khan_mini/lafan1/`（36个NPZ，约969 MiB），保持这个相对路径。
不要把本机 `.venv` 复制过去；服务器重建环境。生成的Khan资产可独立复制，训练无需ref目录、原始PKL或URDF源目录。

## 配置环境并检查命令

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
bash scripts/setup/setup_server_env.sh --project-dir "$PWD"

# 只检查路径、打印完整命令，不启动CUDA/训练，不创建训练日志
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_clean_goal_1024 --dry-run
```

安装细节见 [SERVER_SETUP.md](SERVER_SETUP.md)。脚本从自身路径定位项目，换项目目录无需改源码。
使用项目 `.venv/bin/python`，清除PYTHONPATH/PYTHONHOME及遗留torchrun配置；只启动单GPU。
`CUDA_VISIBLE_DEVICES` 可用于选卡，例如命令前加 `CUDA_VISIBLE_DEVICES=0`。

## 先验证50步，再继续正式训练

旧14-body版本已提供4096环境的服务器训练日志，但这不等于本轮新配置已完成正式训练。
显存需求需在目标设备验证；历史G1约24 GB及旧Khan测试都不是新配置的显存保证。
先执行50步以覆盖首次完整WM、actor反向、critic及Adam状态分配：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_clean_goal_1024 --stop-after-iterations 50
```

成功后日志应显示 `it 50 ... updates 4`，并保存
`logs/khan_lafan1_clean_goal_1024/ckpt_latest.pt`。然后从该检查点继续到正式320000步：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_clean_goal_1024 \
  --resume logs/khan_lafan1_clean_goal_1024/ckpt_latest.pt
```

恢复时重新预热40步填充新进程replay，沿用已有模型/Adam状态/全局迭代和采样器统计。
恢复时使用同一套36段数据。要改成原4096并行数，将上述1024全部改成4096并使用相应run-name。
也可直接从头完整运行：

```bash
bash scripts/training/train_khan_server.sh --num-envs 4096 --run-name khan_lafan1_clean_goal_full
```

这是前台命令，不会替用户自动启动后台任务。控制台可直接查看，TensorBoard位于
`logs/<run-name>/tb`，解析后的完整配置位于 `logs/<run-name>/config.json`。
已有检查点时脚本要求明确 `--resume` 或新run-name，避免静默覆盖。
当前检查点记录 `robot_contract`，加载前检查root/anchor/COM/body、关节、动作缩放、奖励配置、
模型/profile SHA，以及version-4观测schema和goal/history布局。旧无contract及旧观测contract的Khan
检查点会被拒绝，包括此前649维goal的40k/100k/200k模型；当前配置必须新run。
服务器脚本拒绝旧 `--anchor-obs`/`--no-anchor-obs` 开关；新布局固定包含未来anchor姿态，不使用旧9维扩展。
旧策略通过独立导出的旧源码/资产运行器评估，见
[KHAN_LEGACY_BENCHMARK.md](KHAN_LEGACY_BENCHMARK.md)。该入口不切换当前训练配置。
40k抗推录像所用的另一份18 cm宽腰盒快照见
[KHAN_40K_ROBUSTNESS.md](KHAN_40K_ROBUSTNESS.md)。

## 从原配置复用的主要参数

| 项目 | 值 |
|---|---|
| 世界模型 / termination hidden | 2048×3 / 64×2 |
| 策略encoder、decoder hidden | 各2048/1024/512/256 |
| latent / critic hidden | 32 / 256、128 |
| WM batch / window / 每轮批数 | 2048 / 8 / 10 |
| policy batch / H | 4096 / 32 |
| critic | 5 epochs × 4 minibatches |
| UTD / actor:WM外轮 | 0.4 / 1:1 |
| warmup / replay | 40步 / 每环境1000行 |
| WM、actor、critic LR | 单GPU均2e-4 |
| gamma / lambda | 0.99 / 0.95 |
| 训练长度 / LR衰减 | 320000步 / 250000步余弦衰减到0.03× |
| obs/reset/域随机化/执行噪声 | prop观测噪声开启、参考goal干净；其余沿用本库 |
| 观测 / 激活重计算 | isaac_khan，未来/历史5/5 / 关闭 |
| torch.compile | default |

`--stop-after-iterations` 只限制当前进程执行步数，不缩短学习率衰减时间基准。
脚本复用原配置，不提供batch/H等参数覆写入口。完整观察、奖励和DR列表见
[KHAN_TRAINING.md](KHAN_TRAINING.md)。12 GB失败记录见 [KHAN_MEMORY.md](KHAN_MEMORY.md)。

## 本地验证边界

启动脚本的CPU测试会解析生成命令并与真实 `TrainConfig` 对照，确认语法和全部正式参数。
还测试了路径含空格、dry-run不执行解释器、恢复与短测、缺失环境/资产/数据、检查点覆盖保护。
历史15-body实现的检查范围包括FK、10项仿真/解析奖励、649维base-anchor观测、waist COM上下文、
动态回报上界及contract拒绝路径。此前132项测试及12 GB训练记录属于旧配置的历史验证，
不能作为本轮新配置的正式训练或收敛记录。
本轮40k的90条件抗推结果使用冻结旧语义，也不能替代新配置服务器训练。

上一版15-body/base-anchor、649/870观测实现通过151项回归测试，并完成8环境、50控制步训练smoke：
保留正式网络、batch、H及优化参数，执行40次WM、4次actor、80次critic更新，
参数与日志有限，checkpoint和优化器恢复通过。该本机12 GB测试明确开启了可选激活重计算；
正式服务器脚本仍默认关闭，不能用这个smoke推断4096环境的显存需求或训练收敛效果。

上一轮155/465观测、Isaac原scale、287维critic通过165项全量测试，另完成8环境50步训练smoke，
正式网络、batch2048/4096、H32和UTD0.4均保留；40次WM、4次actor、80次critic优化及恢复检查通过。
本机12 GB测试显式开启激活重计算，正式服务器入口仍关闭。这不替代当前467维critic/SuperMimic scale验证。命令和结果见
[KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。

上一版SuperMimic scale、467维critic、goal仍加噪的版本通过 **167项全量测试**，并完成8环境50步正式网络/batch的
训练smoke，包含40次WM、4次actor、80次critic更新，以及模型/优化器/采样器恢复检查；
实际观测、参数和TB标量有限。该本机12 GB测试显式开启激活重计算，服务器脚本默认仍关闭。
结果在 `logs/khan_sm_scale_critic467_smoke/summary.json`，完整说明见上述观测文档的对应历史验证段。
这不替代当前version-4干净goal版本的验证。

当前version-4干净goal版本另通过 **168项全量测试** 和8环境50步正式参数训练smoke：
动力学输入125、goal155、prop465、decoder497、critic467和15个跟踪body确认；
40次WM、4次actor、80次critic更新及模型/优化器/采样器恢复通过，参数与实际TB有限。
本机测试显式开启激活重计算，正式服务器入口仍关闭；结果在
`logs/khan_clean_goal_smoke/summary.json`。默认run名 `khan_lafan1_clean_goal` 的4096环境dry-run也通过。
