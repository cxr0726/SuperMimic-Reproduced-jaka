# Khan 官方组件迁移分支

分支：`feat/official-window-value-normalization`。基于官方 SuperMimic
`5efbf114e11f725c31d83e217be987c13c0e4ce2`，在现有 Khan 框架内适配。
`main` 上已有训练方案保持在原提交。请为本分支新建训练 run；旧 residual
策略仍可按其 checkpoint 配置 play/export，但不能用旧权重直接续训本次新结构。

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
| critic | bootstrap、GAE target 仅上界限制，Khan 为 0.112/(1−0.99)=11.2 |
| actor 输入 | obs/goal 运行归一化，输入维度仍为 465＋155=620 |
| actor 动作 | **默认姿态相对动作**，PD target=`q_default+0.5*actions`，无参考 residual skip |
| actor 网络 | encoder1024/512/256，decoder2048/1024/512/256，隐藏层均为LayerNorm/GELU，latent32 |
| critic 网络 | 256/128，隐藏层为LayerNorm/GELU，最终输出仍为普通Linear |
| Actor EMA | 0.995，采集/优化使用 live；eval/export 默认自动选择 EMA |

保留当前 clip 内分 bin、尾 bin 合并、物理及想象 H32 合法性、15 body/颈关节奖励、
base_link anchor、91 维 ξ、Khan DR、reset 实际 root 抬高 0.05m、5/5 observation
窗口以及原有优化参数和学习率调度。执行器 delay 和参考 FK 编译默认关闭。
WM、Actor、Critic隐藏层均为Linear→LayerNorm→GELU。encoder最终latent仍经tanh，
decoder动作输出和critic价值输出仍为普通Linear，不添加输出归一化或限幅。
WM 仍使用现有 heading 表示（Khan67维状态特征）、66维速度/pose residual 输出、
8次真实transition监督及原误差尺度；没有同时改成官方33维纯速度预测模型或切换Muon。
运行输入归一化取代WM原固定输入除数，输出scale和loss scale保持现有实现。

`--reference-residual` 是显式对比选项，不是新训练默认。原有 encoder goal noise
保留为 z±0.01m、q±0.05rad；critic/reward 不加该噪声，可用 `--no-goal-noise` 关闭。

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
  --run-name khan_8452_official_clip_ln_gelu_4096 \
  --num-envs 4096
```

开启参考 FK 编译时增加 `--reference-fk-compile`，并使用不同 run-name。
编译对整体吞吐的收益需要在服务器 GPU 上测量；它不改变采样或归一化语义。
仅检查路径和命令可增加 `--dry-run`，它不会加载 PT 张量或启动 CUDA。

NPZ 混合库沿用原入口：

```bash
bash scripts/training/train_khan_server.sh \
  --data-dir data/npz/khan_mini/mixed \
  --run-name khan_mixed_official_clip_ln_gelu_4096 \
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
