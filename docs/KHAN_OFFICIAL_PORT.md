# Khan 官方组件迁移分支

分支：`feat/official-window-value-normalization`。基于官方 SuperMimic
`5efbf114e11f725c31d83e217be987c13c0e4ce2`，在现有 Khan 框架内适配。
`main` 上已有训练方案保持在原提交。请为当前新配置新建训练 run；
历史策略仍按其 checkpoint 保存的动作和网络配置 play/export。

## 默认组件

| 组件 | 本分支默认 |
|---|---|
| reset 采样 | `window_value`，每 20 个物理步更新，memory=256 |
| 更新对象 | 窗口中每个环境、回合、bin 的访问回报先平均，再按新样本数更新 |
| 截断尾值 | 失败=0；动作结束=全 bin 价值均值；timeout/窗口末=来源clip内下一帧bin价值，后继不越clip |
| 分布 | 均匀 0.5＋当前困难 0.4＋累计 focus² 历史 0.1 |
| WM 输入 | 状态、物理 PD 目标分别运行均值/方差归一化；ξ 保持范围归一化 |
| WM 隐层 | 2048/2048/2048，Linear→LayerNorm→GELU |
| 想象终止 | 解析软阈值，softness=0.05；不训练 termination MLP |
| critic | bootstrap、GAE target 仅上界限制，当前Khan为0.122/(1−0.99)=12.2 |
| actor 输入 | obs/goal 运行归一化，输入维度仍为 465＋155=620 |
| actor 动作 | **参考残差动作**，PD target=`q_ref_raw(t+1)+0.5*decoder_output`，beta=1 |
| actor 网络 | encoder1024/512/256，decoder2048/1024/512/256，隐藏层均为LayerNorm/GELU，latent32 |
| critic 网络 | 256/128，隐藏层为LayerNorm/GELU，最终输出仍为普通Linear |
| Actor EMA | 0.995，采集/优化使用 live；eval/export 默认自动选择 EMA |

保留当前 clip 内分 bin、尾 bin 合并、物理及想象 H32 合法性、15 body/颈关节奖励、
base_link anchor、Khan DR、reset 实际 root 抬高 0.05m、5/5 observation
窗口以及原有优化参数和学习率调度。执行器 delay 和参考 FK 编译默认关闭。
WM、Actor、Critic隐藏层均为Linear→LayerNorm→GELU。encoder最终latent仍经tanh，
decoder动作输出和critic价值输出仍为普通Linear，不添加输出归一化或限幅。
WM 仍使用现有 heading 表示（Khan67维状态特征）、66维速度/pose residual 输出、
8次真实transition监督及原误差尺度；没有同时改成官方33维纯速度预测模型或切换Muon。
运行输入归一化取代WM原固定输入除数，输出scale和loss scale保持现有实现。

Khan的Isaac布局新训练默认启用参考残差，`--reference-residual` 可显式指定；
`--no-reference-residual` 恢复默认姿态动作做消融。G1及Khan的旧SuperMimic布局
仍使用原默认姿态动作。原有encoder goal noise保留为z±0.01m、q±0.05rad；
critic/reward和参考直通分支保持干净，可用 `--no-goal-noise` 关闭encoder噪声。

## 参考残差与运行归一化

将clean、原始单位的goal与encoder输入明确分开：

```text
clean goal（关节值为rad） ── q_ref(t+1) ── beta*(q_ref-q0)/alpha ──┐
        └─ encoder goal noise ── running normalization ── encoder  │
sensor observations ── running normalization ── decoder residual ─┴─ actions
```

参考分支在调用encoder之前计算，不读取标准化值，也不读取加噪的encoder goal。
当前Khan的goal窗口从t开始，offset=1选第二个关节块，即clean goal的47:74。
alpha是实际执行器action scale，所有Khan关节为0.5；beta=1。
除以alpha只是将物理角度映射为执行器动作坐标，不是使用运行mean/std标准化参考角度。

```text
actions = decoder_output + beta*(q_ref_raw(t+1)-q0)/alpha
PD target = q0 + alpha*actions
          = (1-beta)*q0 + beta*q_ref_raw(t+1) + alpha*decoder_output
```

运行归一化只作用于网络支路；不会修改clean goal或将其原地标准化。
decoder为零且beta=1时，物理目标应等于原始参考q_ref(t+1)，与normalizer的
mean/std、encoder噪声无关。参考分支保留梯度，不添加新的输出限幅或裁剪。
物理采集、想象、EMA及ONNX都调用同一个Policy.forward，残差只叠加一次。
Replay和last_action继续保存完整default-relative动作，WM在内部还原物理PD目标后归一化。

## 归一化与部署契约

运行归一化采用 `(x−mean)/(std+0.01)`，累计 fresh 物理采集数据的经验统计。
forward 不更新统计；replay、想象 rollout、eval、ONNX 推理均只读。
所有 buffer 原地更新并保存至 checkpoint；多卡先聚合各 rank 的充分统计。

Khan 每次仅用历史窗口最新一帧更新 87 维 sensor 统计，所有历史 lag 共用它。
未来姿态 30 维和 goal155 各有独立统计。部署输入仍按现有 C++ 约定提供
角速度×0.5、关节速度×0.2；网络内部先还原原始单位，再归一化。
ONNX 内嵌全部 actor 统计，C++ 不要再次标准化。EMA 的统计 buffer 精确复制 live，
不参与参数指数平均。

采样器窗口统计、value/count/effort/概率和窗口游标纳入 checkpoint。
进程重启不恢复物理仿真世界；恢复后的首段 fresh 采集不能与旧世界尾段拼成同一回合。
训练入口会检查 checkpoint 的动作、归一化、WM 结构和采样契约。
本次还会检查Actor/Critic的LayerNorm与激活函数，拒绝把旧SiLU权重静默解释成新模型。
历史未记录这些字段的checkpoint仍按SiLU、无隐藏LayerNorm加载用于play/export。
此前本分支180k/200k使用default动作，它们仍按原配置play/export，但不能直接resume
到本次reference-residual动作；训练入口会在加载权重前拒绝动作契约不匹配。
相同配置的reference-residual checkpoint可以继续恢复。

## 下肢 startup armature 随机化（khan_dr_v2）

v2阶段使用 `khan_dr_v2`；当前新训练使用下文的v3，旧v2保持可用。
双腿各6轴（hip pitch/roll/yaw、knee、ankle pitch/roll）共12轴的armature
分别乘 `U(0.8,1.2)`，每个并行世界仅在startup采样一次，所有reset保持不变。
不包含腰部和上肢。名义关节参数保持原值；踝pitch/roll仍为0.02/0.01。
Kp/Kd仍为每次reset独立乘 `U(0.8,1.2)`。

`waist_yaw_Link`局部COM在startup采样：x±0.035m，y/z±0.07m；
此前范围分别为±0.025m、±0.05m，本次扩大40%。腰部附加质量−2..3kg
及其他随机化保持原语义；脚部质量和奖励尚未修改，服务器delay默认关闭。
这不是整机COM直接移动上述距离。

ξ扩展为99维：摩擦1、encoder bias27、COM3、Kp27、Kd27、下肢armature12、
腰部附加质量1、delay1。armature来自实际模型字段读回，按上述双腿顺序排列；
COM的范围归一化除数同步设为0.035/0.07/0.07。WM输入193维，critic613维；
actor仍620维。startup事件通过mjlab标准常量刷新，reset不再刷新armature。

历史检查点缺少env.context_layout时按其WM保存值或旧khan_dr_v1加载；
旧v1仍为91维ξ、四个踝轴reset随机化及旧COM范围，用于play/eval/export。
新v2记录独立的v8机器人契约，不能把v1的WM/critic直接resume到v2；请新建run。

v2验证（2026-11-12）：纯CPU回归745通过；实际CUDA环境验证startup、partial/自动reset、
ξ读回及MuJoCo派生常量，与CPU64常量计算在既有10ppm容差内一致。
本地完整8452 PT、8环境运行50步，执行WM40/actor4/critic80次优化，参数全部有限；
恢复后再运行50步成功。网络、batch及优化参数采用当前训练值；本机显存测试
单独开启activation checkpoint，服务器默认仍关闭。旧official200k EMA与
旧residual160k live检查点均按v1成功加载并前向。上述短测验证链路正确，
不代替新配置的收敛、sim2real与单腿表现评估。

## 双脚附加质量与独立跟踪（khan_dr_v3）

当前新训练、NPZ及PT服务器脚本均选择 `khan_dr_v3`。在v2基础上增加：

- 左右 `ankle_roll_Link` 各自独立在startup采样附加质量 `U(0,0.2) kg`，
  加到原始质量上，均值0.1kg；后续完整、局部和自动reset保持，不累加。
  假设附加点质量在该body原始COM处，保持局部COM和绕COM旋转惯量；
  这是点载荷近似，不是未知真实足部质量分布的惯量校准。标准startup刷新派生常量。
- ξ保留v2前99维，末尾加入 `foot_mass_delta` 两维（左、右，单位kg），
  读回实际body质量减原始质量，范围归一化为 `(delta_mass-0.1)/0.1`。
  总计101维，WM输入195维，critic615维；actor输入仍为620维。
- 新增 `motion_foot_pos` / `motion_foot_ori`，各权重0.25，
  std分别0.1m和0.25rad。两脚各自计算指数奖励后取平均，独立于原15body平均。
  使用已有xy/yaw reference reanchor，保留参考z与roll/pitch；不消除足部高度差。
  物理与想象使用同一奖励核、干净参考，想象复用已有FK，不添加contact/滑移项。

对于位置误差e和姿态测地角误差theta，两项分别为：

```text
r_foot_pos = 0.02*0.25*mean_feet(exp(-||e||² / 0.1²))
r_foot_ori = 0.02*0.25*mean_feet(exp(-theta² / 0.25²))
```

原奖励及惩罚权重保持，原始单步上界0.112→0.122，critic理论上界12.2；
采样器价值尺度及归一化分母同步由实际奖励配置推导。train_ret与benchmark满分
仍为1，但新旧奖励定义不同，不能只按数值横比。若全部奖励和惩罚统一乘
0.1/0.122，理想最优策略与项间权重不变，但梯度/value数值也需要一起重标；
目前没有必要为固定0.1引入这一步。只缩正奖励而保留惩罚则会改变目标权衡。

v3使用v9机器人契约；历史v1/v2仍保持原奖励及质量，正常play/eval/export。
关闭DR的名义评估不添加足部质量。v3与旧上下文维度/奖励不同，需要新建run；
PT和NPZ参考数据都不需要重新转换。

```bash
bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_official_residual_dr_v3_feet_4096 --num-envs 4096
```

v3验证（2026-11-12，足部权重各0.25）：纯CPU756项通过；实际CUDA下两脚质量
独立读回、startup跨reset保持、CPU64常量oracle及16随机状态全奖励物理/想象
逐项一致性通过。完整8452 PT、8环境、50物理步完成WM40/actor4/critic80次优化，
模型/EMA/优化器状态全部有限，checkpoint记录ξ101、单步上界0.122及sampler g_max12.2。
网络和batch采用当前正式值；仅本机显存短测开启activation checkpoint，服务器仍关闭。
旧v1/v2真实checkpoint契约和策略加载仍通过。此处验证实现一致性，效果需重新训练评估。

## 学习侧执行加速（2026-11-13）

当前新训练默认启用 `wm_trainer.cuda_graph`、`policy_trainer.cuda_graph` 和
`policy_trainer.batch_reference_encoder`。这是执行优化，保持网络、batch、H32、
更新比例、奖励、DR和原始参考残差动作不变，也不引入BF16或Muon。

Khan的reference-only goal在H32上合并运行一次encoder；实际相对参考朝向仍逐步
由想象state计算。encoder保留梯度，goal噪声只进入encoder副本，残差直通继续
使用clean raw关节角。G1含state-dependent anchor的goal保留逐步编码。

Replay在学习块前根据完整episode/terminal/环形覆盖/H32条件，在GPU刷新固定地址
的合法权重表。每个新replay版本只在边界检查一次空集；图内multinomial和gather
无Tensor转Python。仍是合法环境×时间位置等权，不采用Party的候选拒绝采样或
不完整窗口cut-mask。同一WM学习块的10个batch复用这张表。

WM图覆盖整轮10个minibatch：每批的采样、8步损失、反向、裁剪、Adam更新，以及
轮末指标汇总都在同一张图中，每次重放准确执行10次Adam更新。
Actor/Critic图覆盖H32、Actor反向/Adam/EMA，以及原5×4次Critic优化。
前三轮调用是计入训练的真实warmup，第四轮记录并重放一次，不多做优化步。
EMA首次复制与计数在设备端处理；cosine LR通过设备标量原地更新，重放可见。
CPU和多卡梯度归约使用明确的普通更新回退，不捕获collectives。

采集/reset保持原实现：`env.step → Recorder → fresh normalization → prepare →
WM update → Actor/Critic update`。startup惯量/质量及per-reset PD、摩擦、bias
不变，不引入Party的masked-reset限制或Torch/Warp捕获补丁。官方代码本身也采用
普通采集、独立捕获learners的边界，因此无需先改整个物理采集/reset。

checkpoint仍保存普通Adam与EMA状态；恢复后转换旧float LR/CPU step张量并重建图。
不保存CUDA Graph对象。同v3模型可继续resume；旧v1/v2到v3的语义限制仍适用。

两种服务器脚本提供对照开关：

```bash
# 默认：batched encoder + full learner update capture
bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_v3_learning_capture_4096 --num-envs 4096

# 普通更新对照，网络和学习预算保持相同
# 在上述命令后追加：--no-update-capture
# 若也要逐步encoder对照，再追加：--no-batch-reference-encoder
```

固定同一真实采集replay的速度对照工具：

```bash
.venv/bin/python scripts/benchmarks/benchmark_learner_updates.py \
  --config logs/YOUR_RUN/config.json \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --num-envs 8 --warmup 4 --samples 5 --output logs/learner_update_benchmark.json
```

三组均保留新合法采样器，分别比较逐步encoder普通更新、批量encoder普通更新、
批量encoder捕获更新；它们不代表与旧b024710所有实现的完整差异。
报告warmup/编译成本和稳态耗时；小网络结果不能外推为正式4096batch的加速倍率。

本机实测（当前未提交工作树，RTX 4070 SUPER，8452紧凑PT，固定同一份真实采集
replay）：正式网络、WM batch=2048/window=8/每轮10次、Actor batch=4096/H32、
Critic每轮5×4次更新均保持；物理采集只用8个环境准备replay。三组都启用已有的
activation checkpoint以适应本机12GB显存，服务器脚本默认仍关闭。每组先运行4个
真实学习块，再测5个块，以下为中位数，不含物理采集和首次编译/捕获：

| 执行模式 | WM每轮 | Actor/Critic每轮 | 合计 |
|---|---:|---:|---:|
| 逐步encoder，普通更新 | 391.75 ms | 809.10 ms | 1200.29 ms |
| 批量encoder，普通更新 | 389.96 ms | 811.31 ms | 1202.16 ms |
| 批量encoder，完整更新捕获 | 390.29 ms | 743.31 ms | 1128.78 ms |

各列中位数独立计算，故不一定相加恰等于合计。正式尺寸下合计耗时约下降6.0%，
批量encoder本身未显示提速；WM未测出明确收益，主要收益在Actor/Critic。较小网络与
batch=64的同协议对照为396.81→365.65→63.25 ms，说明launch/调度占比影响很大。
这些结果不能直接推导4090、4096物理环境、关闭activation checkpoint时的it/s。
批量encoder需要保存H×batch的激活，图还会保留私有内存池，不能假设一定省显存。
原始结果为`logs/benchmark_learner_updates_formal_full_20261113.json`和
`logs/benchmark_learner_updates_full_20261113.json`，包含逐次计时、首次编译/捕获
成本、工作树源码及参数指纹和优化步数；三组使用相同初始模型、归一化统计与replay。

验证：CPU非slow回归781项通过；GPU验证合法采样、完整WM/Actor/Critic更新的参数、
梯度和指标一致性，以及变化的LR、图内新噪声、EMA计数与checkpoint恢复。GPU批量
矩阵乘法/Adam执行路径允许FP32舍入误差，不声称逐位相同。完整真实mjlab短训练
正式尺寸运行60步，准确执行WM80/Actor8/Critic160次；小尺寸运行80步后恢复并再
运行50步，累计WM200/Actor20/Critic400次，EMA20，两图均重建并成功重放，参数和
优化器状态均有限。短测验证实现与执行预算，不代替长期收敛或实机评估。
正式尺寸及恢复后的检查结果分别保存在
`logs/smoke_learning_capture_formal_full_20261113/verification.json`和
`logs/smoke_learning_capture_full_resume_20261113/verification.json`。

物理图的取舍依据：官方`runners/supermimic_runner.py::_step_and_store`仍普通调用
`env.step`并写replay，在`_flush_normalization`后独立运行captured learners。
Party的`train.py::_collect_core`则连同采集/reset一起捕获，但其
`env.py::reset_masked`明确限制无stateful sensing、push、per-episode mjlab
randomization和actuator delay；还补丁TorchArray写入、禁用嵌套Warp step图并对齐
Torch/Warp stream。当前Khan每次reset的PD/摩擦/bias、Recorder的terminal额外行和
动态env IDs需另行适配，不能直接套用这部分，也不是本次学习图捕获的前置条件。

## 自举后继边界

采样器每20次物理step更新一次；单个环境reset不重启统计窗口。普通窗口末使用
`V[bin(f+1)]`，其中f为保存的步前参考帧。框架额外将后继限制为
`min(f+1, clip_last_of(f))`，所以只会读取当前来源clip的价值；不能改用自动reset
之后的command.frame。明确clip-end仍用全库V均值，有效failure仍用0，timeout继续
自举；timeout覆盖同时发生的failure这一既有规则保留。

此修复只改变采样器读取的后继索引，不改变物理终止、参考推进、尾帧重复步、
reset时机或奖励。新sampler contract记录`bootstrap_next_frame=clip_clamped`，
与旧的跨clip统计版本不兼容，需要新建训练run。

例如同一clip内连续执行参考帧30～49，共20次transition，且没有发生结束或失败：

```text
G49 = r49 + 0.99 * V[bin(50)]
G48 = r48 + 0.99 * G49
...
G30 = r30 + 0.99*r31 + ... + 0.99^19*r49 + 0.99^20*V[bin(50)]
```

这里V是本轮统计更新前的bin价值表，不是神经网络critic，也不是整条clip的平均价值。
20步是跨环境同步的统计窗口，不是每次reset后的固定回合长度；窗口内若发生结束，
会按该边界的规则切断递推，下一回合的奖励不会倒灌到上一回合。

## 服务器训练

先同步并切换到本分支，再在服务器项目目录运行。以下命令默认 eager 参考 FK：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka

bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_official_residual_norm_4096 \
  --num-envs 4096
```

开启参考 FK 编译时增加 `--reference-fk-compile`，并使用不同 run-name。
编译对整体吞吐的收益需要在服务器 GPU 上测量；它不改变采样或归一化语义。
仅检查路径和命令可增加 `--dry-run`，它不会加载 PT 张量或启动 CUDA。

NPZ 混合库沿用原入口：

```bash
bash scripts/training/train_khan_server.sh \
  --data-dir data/npz/khan_mini/mixed \
  --run-name khan_mixed_official_residual_norm_4096 \
  --num-envs 4096
```

两个 wrapper 都保留环境变量清理、路径/资产检查、已有 checkpoint 防覆盖和
`--resume`、`--stop-after-iterations` 功能。只允许从相同本分支配置恢复；
`--stop-after-iterations` 不缩短正式 320k 学习率调度。

## 观察指标

- `sampler/window_updates`、`sampler/visited_bin_fraction`、`sampler/max_probability`、entropy。
- `normalization/wm_count` 及保存的 actor normalizer count，确认只有 fresh 采集增加计数。
- critic 原始 bootstrap/value/target 的超界比例、最大值、P99，以及 `critic_value_ceiling`。
- `train/episode_norm_return`、WM 多步误差、完整动作 benchmark 和随机 reset 失败率。

这些组件来自官方实现，但不等同于官方整套训练配置，也不保证未经正式训练就提高
Khan 的 benchmark 或实机稳定性。仍需用新 run 与旧策略做受控对比。

## 迁移验证（2026-11-11）

- CPU 回归：672 passed、34 skipped（CUDA 条件测试）。包含官方采样器 golden
  数值、reset 边界、计数加权、恢复、双进程统计、梯度和 ONNX 对照。
- RTX 4070 SUPER 上完整加载 8452 条紧凑 PT，并以 8 环境运行 50 步；另对
  36 条 LAFAN NPZ 跑同样测试。两组都实际执行 40 次 WM、4 次 actor、80 次
  critic optimizer step，保存并恢复模型、优化器、EMA 和采样统计，全部通过。
- 学习 batch 仍为 WM2048、actor4096、H32，网络和学习率调度保持正式配置。
  本地测试启用可选激活重计算以适应 12GB 显存；服务器 wrapper 默认关闭。
- 每组结束时 WM state/action、actor sensor/goal/orientation 的计数都是
  8×50=400；想象和 replay 没有增加计数。采样器完成 2 个20步窗口，保存的
  待完成窗口为10步；真实重启恢复时会丢弃这个旧轨迹窗口。
- 实际带运行统计的 EMA 模型导出 ONNX 后，最大 action 误差为3.88e-8，
  PD target误差为1.20e-7 rad。该文件是 smoke 权重，不是可部署的已训练策略。
- 历史160k residual checkpoint仍能按旧配置加载，运行归一化保持关闭。

本地结果位于 `logs/smoke_official_window_norm_20261111/`（PT，delay开启）和
`logs/smoke_official_window_norm_npz_20261111/`（NPZ，delay关闭）。首次编译耗时
较长，不应将该50步测试的平均速度当作正式训练吞吐。以上验证不包含收敛性或实机测试。

## 后继修复与 Actor/Critic 隐层验证（2026-11-11）

- 完整 CPU 回归：710 passed、34 skipped。新增末帧、timeout、失败、partial reset、
  H20正常窗口、合并尾bin、恢复契约及隐藏层结构/梯度/导出的检查。
- 普通窗口30～49的自举结果修前修后完全相同；来源clip为0～99时，80～99窗口
  最后一步的构造数值由错误的8.02修正为4.06，没有改变物理帧和结束标记轨迹。
- 真实mjlab单环境在walk clip末帧13064执行一次物理step，没有提前结束或reset；
  sampler值为3.9062686，与同clip自举一致，未读取下一clip对应的7.8662686。
- 完整8452条PT库、8环境、50步GPU训练通过；正式网络宽度、学习batch及调度不变，
  实际执行40次WM、4次actor、80次critic optimizer step。模型参数均更新且有限，
  模型、优化器、EMA及sampler恢复检查通过。本地启用激活重计算，delay关闭。
- 新LayerNorm/GELU EMA actor实际导出ONNX，对照零输入、随机输入、动态batch和
  NPZ参考状态，最大action误差2.42e-8，PD target误差1.19e-7 rad。
- 旧运行归一化SiLU smoke checkpoint和旧160k residual checkpoint均按原结构加载，
  有限值推理通过；没有把历史缺失字段解释为新LayerNorm/GELU结构。

本次结果位于 `logs/smoke_clip_successor_ln_gelu_20261111/`。导出文件仅用于验证，
属于50步smoke权重。以上结果证明链路与边界检查通过，不代表正式训练已收敛或实机性能提升。

## 恢复参考残差验证（2026-11-12）

- 原可选残差实现已使用clean raw goal；本次进一步在encoder调用前构造独立的
  reference action，并将Khan Isaac新训练及两个server wrapper的默认模式改为残差。
- 全量已有CPU回归724 passed、34 skipped；另有9项归一化残差专项，检查非零且
  不同维度的mean/std、非均匀物理action scale、encoder噪声隔离、raw弧度梯度、
  统计只读、编译梯度、EMA恢复及packed/split ONNX动态batch一致性。
- 完整8452条PT库、8环境、50步GPU训练通过，保持正式网络宽度、WM batch2048、
  policy batch4096、H32和学习率调度。实际完成40次WM、4次actor、80次critic优化，
  参数更新且全有限，模型/optimizer/EMA/sampler扰动后恢复通过。
- 本地测试启用激活重计算以适应12GB显存，delay关闭；server默认激活重计算关闭。
  采集统计计数为400。8环境50步的Replay不足常规online-error诊断的2048行门槛，
  终端nse8的nan表示该诊断缺测；训练loss、模型与记录的TB标量均有限。
- 正式宽度的归一化残差EMA actor导出后，最大action和PD target误差均约2.38e-7。
  旧200k归一化default策略、旧180k固定scale残差策略及新smoke策略均按保存配置加载。

本次训练/导出验证产物位于 `logs/smoke_official_residual_norm_20261112/`；
smoke仅用于训练链路验证，不是新模式正式训练结果或部署策略。
