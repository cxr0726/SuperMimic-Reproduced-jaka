# Khan 官方组件迁移分支

当前消融分支：`feat/khan-dr-v3-only`，由 `feat/official-window-value-normalization`
的足部奖励版本分出。官方组件基于 SuperMimic
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
| critic | bootstrap、GAE target 仅上界限制，当前Khan为0.112/(1−0.99)=11.2 |
| actor 输入 | obs/goal 运行归一化，输入维度仍为 465＋155=620 |
| actor 动作 | **默认姿态动作**，PD target=`q_default+0.5*actor_output`；参考残差保留为可选项 |
| actor 网络 | encoder1024/512/256，decoder2048/1024/512/256，隐藏层均为LayerNorm/GELU，latent32 |
| critic 网络 | 256/128，隐藏层为LayerNorm/GELU，最终输出仍为普通Linear |
| Actor EMA | 0.995，采集/优化使用 live；eval/export 默认自动选择 EMA |
| 额外足部跟踪 | 默认关闭；保留原15 body及颈关节奖励，不改变足部质量随机化 |
| DR | 保留 `khan_dr_v3`，ξ101维、WM输入195维、critic615维 |

保留当前 clip 内分 bin、尾 bin 合并、物理及想象 H32 合法性、15 body/颈关节奖励、
base_link anchor、Khan DR、reset 实际 root 抬高 0.05m、5/5 observation
窗口以及原有优化参数和学习率调度。执行器 delay 和参考 FK 编译默认关闭。
WM、Actor、Critic隐藏层均为Linear→LayerNorm→GELU。encoder最终latent仍经tanh，
decoder动作输出和critic价值输出仍为普通Linear，不添加输出归一化或限幅。
WM 仍使用现有 heading 表示（Khan67维状态特征）、66维速度/pose residual 输出、
8次真实transition监督及原误差尺度；没有同时改成官方33维纯速度预测模型或切换Muon。
运行输入归一化取代WM原固定输入除数，输出scale和loss scale保持现有实现。

本分支用新run隔离上一轮的参考残差和额外足部奖励：默认关闭这两项，保留
扩展后的DR v3、官方采样器/归一化/LayerNorm/GELU、网络、优化器及学习率调度。
`--reference-residual` 和 `--foot-tracking` 可分别恢复残差和足部奖励；
`--no-reference-residual` 和 `--no-foot-tracking` 可显式指定当前默认值。
这两个开关独立，不会关闭双脚质量随机化，也不会缩减ξ或critic输入。
原有encoder goal noise仍为z±0.01m、q±0.05rad，与上一轮相同；critic/reward
和可选参考直通分支保持干净，可用 `--no-goal-noise` 关闭encoder噪声。

关闭额外足部奖励后，原始单步奖励上界恢复为0.112，critic理论上界、采样器
价值尺度及回报归一化分母同步由实际奖励配置得到11.2/0.112；`train_ret`和
benchmark的满分仍为1。这没有改变原15 body和颈部奖励的相对权重。当前默认
与上一轮足部权重各0.25的训练目标不同，须新建run，不能直接比较两种定义的
原始`train_ret`或直接续训；效果比较须使用一致奖励定义和条件。

## 可选参考残差与运行归一化

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
历史default和reference-residual策略仍按各自保存的配置play/export。恢复训练
必须同时匹配动作语义、奖励定义、上下文和完整训练契约，不能因网络维度相同
而静默切换动作或奖励。此前DR v3残差＋足部奖励run的恢复，必须显式提供
`--reference-residual --foot-tracking`，并使用该checkpoint对应的数据和其他配置；
本分支默认的default动作＋无额外足部奖励实验应新建run。

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

## 双脚附加质量与可选独立跟踪（khan_dr_v3）

当前新训练、NPZ及PT服务器脚本均选择 `khan_dr_v3`。在v2基础上增加：

- 左右 `ankle_roll_Link` 各自独立在startup采样附加质量 `U(0,0.2) kg`，
  加到原始质量上，均值0.1kg；后续完整、局部和自动reset保持，不累加。
  假设附加点质量在该body原始COM处，保持局部COM和绕COM旋转惯量；
  这是点载荷近似，不是未知真实足部质量分布的惯量校准。标准startup刷新派生常量。
- ξ保留v2前99维，末尾加入 `foot_mass_delta` 两维（左、右，单位kg），
  读回实际body质量减原始质量，范围归一化为 `(delta_mass-0.1)/0.1`。
  总计101维，WM输入195维，critic615维；actor输入仍为620维。
- 可选 `motion_foot_pos` / `motion_foot_ori`，启用时各权重0.25，
  std分别0.1m和0.25rad。两脚各自计算指数奖励后取平均，独立于原15body平均。
  使用已有xy/yaw reference reanchor，保留参考z与roll/pitch；不消除足部高度差。
  物理与想象使用同一奖励核、干净参考，想象复用已有FK，不添加contact/滑移项。
  当前DR-v3-only分支默认关闭两项；`--foot-tracking`启用。DR上下文版本
  只决定动力学随机化和ξ，不再隐式决定是否加入这两项奖励。

对于位置误差e和姿态测地角误差theta，两项分别为：

```text
r_foot_pos = 0.02*0.25*mean_feet(exp(-||e||² / 0.1²))
r_foot_ori = 0.02*0.25*mean_feet(exp(-theta² / 0.25²))
```

启用额外足部奖励时，原奖励及惩罚权重保持，原始单步上界0.112→0.122，critic理论上界12.2；
采样器价值尺度及归一化分母同步由实际奖励配置推导。train_ret与benchmark满分
仍为1，但新旧奖励定义不同，不能只按数值横比。若全部奖励和惩罚统一乘
0.1/0.122，理想最优策略与项间权重不变，但梯度/value数值也需要一起重标；
目前没有必要为固定0.1引入这一步。只缩正奖励而保留惩罚则会改变目标权衡。

历史v3足部奖励run使用v9机器人契约；当前新run显式记录足部奖励开关。
历史v1/v2/v3仍按原奖励及质量配置正常play/eval/export。关闭DR的名义评估
不添加足部质量。改变上下文维度、动作或奖励定义需要新建run；PT和NPZ参考
数据都不需要重新转换。

以下是恢复上一轮残差＋足部奖励配置的显式选择示例，不是当前默认消融：

```bash
bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_official_residual_dr_v3_feet_4096 --num-envs 4096 \
  --reference-residual --foot-tracking
```

v3验证（2026-11-12，足部权重各0.25）：纯CPU756项通过；实际CUDA下两脚质量
独立读回、startup跨reset保持、CPU64常量oracle及16随机状态全奖励物理/想象
逐项一致性通过。完整8452 PT、8环境、50物理步完成WM40/actor4/critic80次优化，
模型/EMA/优化器状态全部有限，checkpoint记录ξ101、单步上界0.122及sampler g_max12.2。
网络和batch采用当前正式值；仅本机显存短测开启activation checkpoint，服务器仍关闭。
旧v1/v2真实checkpoint契约和策略加载仍通过。此处验证实现一致性，效果需重新训练评估。

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

先同步并切换到本分支，再在服务器项目目录运行。切换前应保存服务器自己的
代码改动；以下命令不覆盖本地修改。默认eager参考FK、delay关闭、default动作、
额外足部奖励关闭，保留DR v3和encoder goal噪声：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
git fetch origin
git switch feat/khan-dr-v3-only
git pull --ff-only origin feat/khan-dr-v3-only

bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_dr_v3_only_4096 \
  --num-envs 4096
```

开启参考 FK 编译时增加 `--reference-fk-compile`，并使用不同 run-name。
编译对整体吞吐的收益需要在服务器 GPU 上测量；它不改变采样或归一化语义。
仅检查路径和命令可增加 `--dry-run`，它不会加载 PT 张量或启动 CUDA。

NPZ 混合库沿用原入口：

```bash
bash scripts/training/train_khan_server.sh \
  --data-dir data/npz/khan_mini/mixed \
  --run-name khan_mixed_dr_v3_only_4096 \
  --num-envs 4096
```

两个 wrapper 都保留环境变量清理、路径/资产检查、已有 checkpoint 防覆盖和
`--resume`、`--stop-after-iterations` 功能。只允许从相同本分支配置恢复；
`--stop-after-iterations` 不缩短正式 320k 学习率调度。
旧残差＋足部奖励run的续训还需显式指定`--reference-residual --foot-tracking`；
脚本不会从文件名猜测这些设置，训练入口会检查checkpoint兼容性。

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
