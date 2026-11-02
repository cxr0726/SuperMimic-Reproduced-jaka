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

decoder输入为497、critic输入为545；state67、action27、context31保持不变。
critic在原467维后新增78维当前误差：q_ref−q27、重锚body位置误差45（实际root heading系）、完整root相对旋转rot6d6；scale均为1。
Khan新增0–20 ms每回合固定的执行器延迟；nominal eval关闭延迟。ξ31及WM125不变，WM/想象不加显式队列，使用隐式拟合近似。
脚部改成每脚1个完整CAD凸包，质量/惯量不变；运动跟踪所有reset对实际机器人root加0.05 m，包含eval，参考/NPZ/goal不变。
actor本体角速度/关节速度有效scale为0.5/0.2，恢复原SuperMimic值；prop噪声保留此前范围，参考goal保持干净。
critic除原canonical state/context/root error外，接收干净command155、未来朝向30和5帧参考速度/重力180。
新增180维按帧排列，每帧为qd27、重力3、root系角速度3、heading系线速度3，网络入口乘数为0.2/1/0.5/1。
原末尾9维anchor goal由prop前部5帧未来姿态替代。**包括version-4 clean-goal/100k在内的旧Khan检查点均不兼容，当前v5必须新run；旧策略需匹配的历史源码/资产做eval或export。**
观测公式、噪声及与Isaac PPO的差异见 [KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。
完整奖励公式及对齐范围见 [KHAN_TRAINING.md](KHAN_TRAINING.md)。
世界模型动力学输入为canonical state67＋延迟队列前的带执行噪声命令27＋标准化context31，共125维；
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

部署前同步包含 `isaac_khan` 布局、SuperMimic actor尺度、545维critic及Khan version-5 contract的源码和当前机器人资产。
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
  --num-envs 1024 --run-name khan_lafan1_delay_convex_1024 --dry-run
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
  --num-envs 1024 --run-name khan_lafan1_delay_convex_1024 --stop-after-iterations 50
```

成功后日志应显示 `it 50 ... updates 4`，并保存
`logs/khan_lafan1_delay_convex_1024/ckpt_latest.pt`。然后从该检查点继续到正式320000步：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_delay_convex_1024 \
  --resume logs/khan_lafan1_delay_convex_1024/ckpt_latest.pt
```

恢复时重新预热40步填充新进程replay，沿用已有模型/Adam状态/全局迭代和采样器统计。
恢复时使用同一套36段数据。要改成原4096并行数，将上述1024全部改成4096并使用相应run-name。
也可直接从头完整运行：

```bash
bash scripts/training/train_khan_server.sh --num-envs 4096 --run-name khan_lafan1_delay_convex_full
```

这是前台命令，不会替用户自动启动后台任务。控制台可直接查看，TensorBoard位于
`logs/<run-name>/tb`，解析后的完整配置位于 `logs/<run-name>/config.json`。
已有检查点时脚本要求明确 `--resume` 或新run-name，避免静默覆盖。
当前检查点记录 `robot_contract`，加载前检查root/anchor/COM/body、关节、动作缩放、奖励配置、
模型/profile SHA，以及version-5观测/仿真schema和goal/history布局。旧无contract及旧观测contract的Khan
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
| obs/reset/域随机化/执行噪声 | prop噪声开、goal干净；Khan延迟0–20 ms；实际机器人reset z+0.05 m；执行噪声std0.1 |
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
本机12 GB测试显式开启激活重计算，正式服务器入口仍关闭。这不替代当时467维critic/SuperMimic scale验证，也不代表当前v5验证。命令和结果见
[KHAN_ISAAC_OBSERVATIONS.md](KHAN_ISAAC_OBSERVATIONS.md)。

上一版SuperMimic scale、467维critic、goal仍加噪的版本通过 **167项全量测试**，并完成8环境50步正式网络/batch的
训练smoke，包含40次WM、4次actor、80次critic更新，以及模型/优化器/采样器恢复检查；
实际观测、参数和TB标量有限。该本机12 GB测试显式开启激活重计算，服务器脚本默认仍关闭。
结果在 `logs/khan_sm_scale_critic467_smoke/summary.json`，完整说明见上述观测文档的对应历史验证段。
这不替代当时version-4干净goal版本的验证，更不代表当前v5的新物理配置已验证。

历史version-4干净goal版本另通过 **168项全量测试** 和8环境50步正式参数训练smoke：
动力学输入125、goal155、prop465、decoder497、critic467和15个跟踪body确认；
40次WM、4次actor、80次critic更新及模型/优化器/采样器恢复通过，参数与实际TB有限。
本机测试显式开启激活重计算，正式服务器入口仍关闭；结果在
`logs/khan_clean_goal_smoke/summary.json`。默认run名 `khan_lafan1_clean_goal` 的4096环境dry-run也通过。

## 起始 bin 采样器（2026-11-02）

本节记录该次起始bin采样器改动：当时G1和Khan共用新采样实现，且该次未改变网络、观测、奖励、域随机化和正式优化参数。当前Khan v5的延迟、脚凸包、reset余量和critic扩展另见本文开头。
这次修改对齐 `supermimic_vibed`、`instantTrack`、`Uni-SuperMimic` 共有的正常截断平均奖励续值，
并采用 vibed 的重复 bin 聚合和严格想象起点筛选；不复制 instantTrack/Uni 的重复索引覆盖及
`no_sync` 固定候选不足时仍可能放入无效起点的做法。

| 项目 | 原 G1 / Khan 复现实现 | 当前实现 |
|---|---|---|
| 正常片尾 | 不续值 | 用回合平均奖励续值 |
| 最大回合时长截断 | critic 接口未接入，丢弃截断前不足100步的回报样本 | 同样用平均奖励续值 |
| 真实高度/倾斜失败 | 不续值 | 不续值；与 timeout 同时发生也按失败处理 |
| 更新对象 | 各步 RTG 对应的所有经过 bin | 只更新回合首个真实控制步的起始 bin |
| 重复 bin | 按步骤求均值 | 同批已结束回合先按起始 bin 求均值，再更新一次 |
| bin 新目标权重 | 0.05 | 0.2 |
| bin 初值 | G_max | 0 |
| 历史分布 | 每次起帧采样做 EMA，0.01 | 相同调用单位，EMA 0.001 |
| 混合分布 | 均匀/当前困难/历史 = 0.5/0.4/0.1 | 相同 |
| 策略想象 H=32 | 参考到末尾后重复最后一帧继续 rollout | 严格筛 `frame+H<=clip_last`，rollout 再校验 |

同批并行环境可能有多个结束回合来自同一个起始 bin。例如旧 `G=1`，两个回合的目标为3和7，
当前先求均值5，再做一次EMA：`0.8*1+0.2*5=1.8`。instantTrack/Uni直接向重复索引写入，
未执行该均值聚合，不能将其视为相同更新；当前也不逐回合连续执行多次EMA，以免结果依赖回合排列顺序。

“完整参考”要求H个想象转移和末端状态均位于原始参考内，不要求replay已经存储之后H步的真实转移。
例如末帧100、H=32时，起点68合法，69不合法。instantTrack/Uni普通路径也用循环重采保证这一点；
它们的 `no_sync` 路径仅从2B候选排序后取B个，有效候选不足时可能混入无效项。当前直接从全部有效
replay起点的mask采样，无有效起点时明确报错。该保证不意味着比它们的快速路径运行更快。

令 `D=sum(gamma**t * reward[t])`、`T` 为实际控制步数、`r_mean=sum(reward)/T`，
正常片尾及时间截断用于更新 bin 的目标为：

```text
target = D + gamma**T * r_mean / (1-gamma)
```

真实失败的目标只有 `D`。补值仅用于重置采样统计，不改变真实奖励、`train_ret`、世界模型目标、
actor 的 H 步末端 critic 续值或 termination 条件。它是假设截断后保持本回合平均质量的估计，
并不预测实际未来奖励；不同奖励时序下与 mjlab 参考库的有限片段归一化方法不完全等价。

`G_max` 仍然是正奖励上界除以 `(1-gamma)` 的 softmax 归一化常量；改为零的是 `G_b` 初始估计，
不是 `G_max`。所有 bin 初值相等时采样都均匀；部分 bin 更新后，零初值会保留未访问 bin 的困难权重，
而 G_max 初值将未访问区域暂视为容易。历史项平均的是 `softmax(-25 G/G_max)` 概率，而非回报或失败次数。

为保持原有 bin 编号及诊断工具兼容，MotionLibrary 的 bin 表不改变；但仅含最终状态、没有后继状态的
bin 在均匀、困难、历史三个分支中均无概率。bin 内起帧严格小于真实末帧，不把最后 singleton bin
折叠到前一个 bin。K 帧未来 goal 保留现有末帧 padding；真实 H 步想象不会跨片尾。

奖励累计改为每环境的 sum、discounted sum 和步数，不再保存整回合奖励矩阵，也没有旧1024步日志上限。
初始化/手动 reset 的旧奖励不再被计入，terminal reward 每控制步只计一次。
mjlab 显式 reset 的 `dt=0`、自动 reset 的逐环境零 dt 只刷新参考，不再推进帧；非重置环境正常推进。
本次没有改变仿真侧判断 clip end 的控制步时序，也没有更改 reward 所使用的参考帧约定。

新 checkpoint 保存 `sampler_contract`，包含算法、系数、bin 边界约定及有序 clip 名称/长度的哈希。
新版本恢复时严格校验 contract 和统计张量。恢复没有该字段的旧 checkpoint 时，会明确告警并将
旧 bin 回报归零、历史分布重置为有效 bin 上均匀分布；网络、优化器状态和迭代/LR进度继续恢复。
因此恢复旧权重不是从零开始训练，也不会延续旧的片尾采样偏置。为比较采样算法效果，建议新建训练实验：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 4096 --run-name khan_lafan1_sampler_v2
```

本次全量回归为 **234 passed、1 skipped**（一项旧的观测一致性测试在短rollout发生reset时主动跳过）；另完成8环境50步的正式网络和batch训练短测，
40次WM、4次actor、80次critic更新及模型/优化器/新采样统计恢复均通过。
该12GB本地短测显式启用了激活重计算，正式训练入口仍默认关闭；配置与结果保存在
`logs/khan_sampler_v2_smoke/summary.json`。50步时replay不足2048行，尚未触发独立的online NSE8评估，
因此控制台该字段显示缺省 `nan`；实际训练损失、参数和已写出的TensorBoard标量均有限。
短测用于验证实现及更新链路，不证明长期训练回报或benchmark已经改善。

## 当前v5配置与验证记录

当前默认run名为 `khan_lafan1_delay_convex`，必须从头训练。网络隐藏层、latent、batch、H、UTD、优化器和学习率计划继续使用原正式值；
critic仅将输入层从467扩为545。actor155/465、decoder497、ξ31和WM125均不变，G1默认不受本次Khan物理适配影响。
旧100k/168项测试、上面的采样器234项测试、历史benchmark与旧ONNX验证均属于对应历史版本。

delay pending target在reset时清零为物理关节目标0，没有用参考姿态预填；模型未加入延迟队列或delay参数，采用隐式拟合近似。
reset z+0.05 m覆盖训练与nominal eval，参考保持原高；这是常态接触余量，无法修复严重约25 cm穿地。
当前v5实际回归、smoke和恢复结果在完成后单独记录；此处不沿用历史测试总数作为新版本结论。
