# Khan 无延迟与短尾 bin 对照训练（2026-11-08）

本轮新训练默认关闭 actuator delay，开启不足半 bin 尾部合并。两个服务器启动脚本显式传入 `--env.actuator-delay False --env.merge-short-tail-bins True`。根部仅在 reset 时抬高 0.05 m、首步 previous-action=0、action-rate 奖励及其权重、其他 DR、91 维 ξ、605 维 critic、观测和正式网络／优化参数保持原值。没有加入 WM 统计归一化、LayerNorm 或 policy EMA。

关闭延迟时实际执行器不创建 delay buffer，ξ 中最后的 delay_steps 为0。保留 ξ 维度以避免无关的模型结构变化；其归一化值按原中心／尺度为−1。通用机器人模型的历史默认设置保留，以便验证旧检查点；新的 TrainConfig 在解析时默认关闭延迟。直接调用底层 `make_env` 时仍应显式传 `actuator_delay=False`。

## 服务器全量 8452 训练

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_pt_nodelay_4096 \
  --num-envs 4096
```

参考 FK 默认 eager；若要做编译对照，在同一命令后加 `--reference-fk-compile`，并使用不同 run-name。`--dry-run` 只检查路径和打印命令，不导入训练／CUDA；`--stop-after-iterations 50` 只限制执行长度，保留320k学习率调度。环境清理、已有检查点保护、路径检查及 resume 保护继续生效。

## 从全量 PT 精确抽取 mixed480 对应动作

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/data/subset_library.py \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --out data/pt/khan_mixed480_from_8452_compact.pt \
  --subset mixed480 --reuse-existing

bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_mixed480_from_8452_compact.pt \
  --run-name khan_mixed480_pt_nodelay_4096 \
  --num-envs 4096
```

内置名单在 `src/supermimic/data/motion_subsets/khan_mixed480.txt`，选择444条 `mixed__` 加36条 `lafan1__`，不需要服务器存在原 NPZ，排除另外4条 LAFAN。缺失、歧义、非法 metadata 会报错，不会悄悄补足或取前480条。也可用 `--motion-list-dir data/npz/khan_mini/mixed` 替代 `--subset mixed480`，只读取文件名；`--clips` 接受精确 PT 名称。

导出逐片复制已存的67列、原始 dtype 和帧数，不做 FK、重采样、接地、裁剪或新量化。默认拒绝覆盖；`--reuse-existing` 会核对父库SHA、动作布局及全部已存参考帧后才接受已有文件。发布采用同目录临时文件和原子链接，禁止源／目标同路径。

实际子集是480条／692925帧／fp16，约89 MB。它与旧 mixed480 **动作名单相同，但数据并非完全相同**：旧NPZ共692909帧，36条LAFAN中16条多1帧，LAFAN轨迹本身也重新处理；444条mixed的量化位置一致，但速度有小差别。若要严格对照旧数据的训练趋势，另运行：

```bash
bash scripts/training/train_khan_server.sh \
  --data-dir data/npz/khan_mini/mixed \
  --run-name khan_mixed480_old_npz_nodelay_4096 \
  --num-envs 4096
```

该 NPZ 命令也使用本轮无延迟／合并 bin 配置；旧训练与本轮的踝标称常量等历史变化仍需控制，不能只凭曲线把差别归于一种改动。

## 合并规则与检查点

默认 bin 为50参考帧。若某个 clip 尾部余数为1–24帧，且它已有一个完整 bin，则并入其前一个 bin；恰好25帧保留。不跨 clip 合并，不能把本来就短于半 bin 的整条 clip 变长。全量库最短 clip 为101帧，没有这种例外。

全量库由80559变为76464个 bin，合并4095个尾部；独立 bin 最短为25参考帧，最后 terminal 状态仍不用于 reset。两个跪姿的末 bin 从350开始／150开始改为 `[300,358)`／`[100,162)`，实际可重置范围为300–356／100–160。帧数据、clip边界、FK及goal保持不变，`bin_of_frame` 同步采用合并后的映射。

开启合并使用 sampler contract v3，旧 v2 的 G/history 不可按原索引恢复，因此本轮请新建 run。`--resume` 仅用于同一新布局、数据与配置的 run。显式关闭合并时保留原 v2 合同；G1训练默认不启用合并。需要手动恢复旧采样/延迟实验，可在 Python 入口显式传 `--env.merge-short-tail-bins False --env.actuator-delay True`；服务器脚本固定本轮对照配置。

本轮没有改变 H32 replay 资格过滤、softmax温度、采样混合比例、EMA或均值续尾目标。因此合并会缓解独立短尾过采样，但最后31帧仍不能成为固定H32想象起点，也没有保证训练回报一定单调上升。

## TensorBoard 分开观察首步与后续跟踪

统计只累计真实完成回合，日志周期汇总；多卡先归约总量和计数，随后主卡写TB。不会向每步训练额外加入CPU同步，也不改变奖励或采样目标。

| 指标 | 含义 |
| --- | --- |
| `reset/first_step_reward` | 真实 episode 首控制步总奖励，已含 dt |
| `reset/first_step_action_rate_reward` | 首控制步 action-rate 奖励，已含权重和 dt |
| `tracking/post_first_step_norm_return` | 排除首步后的回报，按回合等权，除以单步正奖励上限0.112 |
| `tracking/post_first_step_norm_return_step_weighted` | 排除首步后按实际步数加权的归一化回报 |
| `tracking/post_first_step_action_rate_reward` | 后续控制步的 action-rate 均值 |
| `reset/first_step_target_contribution` | 首步奖励在原采样目标中的总贡献，包含均值续尾对首步的外推 |
| `reset/first_step_action_rate_target_contribution` | 上一项中来自 action-rate 的部分 |
| `tracking/post_first_step_target_contribution` | 原采样目标中剩余控制步的贡献 |
| `sampler/completed_episode_target` | 该日志窗口完成回合的采样目标均值 |

`first_step_target_contribution + post_first_step_target_contribution = completed_episode_target`（浮点容差内）。真实终止不续尾；正常cutoff继续采用原公式。首步不一定全部属于人为reset成本，后续数步也可能仍有reset瞬态；这些名称只定义可复核的边界统计。单步回合只计首步，不伪造后续均值，使用 `reset/completed_episodes` 与 `tracking/post_first_step_episodes` 核对分母。

## 本地验证范围

CPU回归覆盖合并区间/映射、参考数值不变、PT子集完整复制、去重统计、partial reset、采样合同保护及启动脚本参数解析。实际 GPU 流程用480条子集、4环境、50步，正式网络／WM batch2048／策略batch4096／H32／更新比例不变。

12 GB机器关闭激活重计算时，在首个策略更新处显存不足；仅本地流程验证启用已有 `--activation-checkpoint` 选项，完成40次WM、4次策略和80次critic优化步及检查点恢复。服务器脚本不启用激活重计算。这个短测验证实现和更新链路，不证明新的学习曲线或收敛质量，后者需服务器长训与固定协议评估。
