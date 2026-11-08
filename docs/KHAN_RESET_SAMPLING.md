# Khan reset 边界、延迟与尾部采样（2026-11-09）

当前 Khan 新训练默认开启每回合保持的 0–20 ms actuator delay（物理步5 ms，lag为0–4）；reset 后第一个20 ms控制步立即执行本次策略目标，第二控制步恢复正常延迟。所有不足一个50帧 bin 的最终余段并入前一个 bin；物理 reset 起点必须满足 `frame + H <= clip_last`，H默认取策略 horizon=32。G1默认保持原行为。

根部仅在 reset 时抬高0.05 m、首步previous-action=0、action-rate及其权重、其他DR、91维ξ、605维critic、观测和正式网络／优化参数不变。WM和想象仍不建显式延迟队列，ξ仍读取已采样的delay_steps。首步立即执行是物理边界处理，并未额外向WM加入“fresh”状态标志，仍属于隐式近似。没有修改均值续尾采样目标。

实现不是临时把lag置零：先处理动作噪声、scale、clip、encoder bias，再用实际目标回填刚reset环境的整个历史队列。保持lag、队列指针／时钟及未reset环境的历史不变。也没有把raw action改为参考姿态对应的action，因此首步差分仍是 `a0 - 0`。

底层 `SuperMimicEnvCfgArgs` 保留旧默认（首步绕过False、half fraction=.5、min remaining=1），以兼容旧评估与play。正式 `TrainConfig` 将Khan解析为新默认；旧检查点使用自身保存配置验证物理语义，不被强行解释成新版本。

## 服务器全量 8452 训练

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_pt_delay_fresh_h32_4096 \
  --num-envs 4096
```

要隔离延迟的影响，在命令后加 `--no-actuator-delay` 并换新run-name；该对照仍保留full-bin合并和H32限制，避免同时改变多个变量。

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
  --run-name khan_mixed480_pt_delay_fresh_h32_4096 \
  --num-envs 4096
```

内置名单在 `src/supermimic/data/motion_subsets/khan_mixed480.txt`，选择444条 `mixed__` 加36条 `lafan1__`，不需要服务器存在原 NPZ，排除另外4条 LAFAN。缺失、歧义、非法 metadata 会报错，不会悄悄补足或取前480条。也可用 `--motion-list-dir data/npz/khan_mini/mixed` 替代 `--subset mixed480`，只读取文件名；`--clips` 接受精确 PT 名称。

导出逐片复制已存的67列、原始 dtype 和帧数，不做 FK、重采样、接地、裁剪或新量化。默认拒绝覆盖；`--reuse-existing` 会核对父库SHA、动作布局及全部已存参考帧后才接受已有文件。发布采用同目录临时文件和原子链接，禁止源／目标同路径。

实际子集是480条／692925帧／fp16，约89 MB。它与旧 mixed480 **动作名单相同，但数据并非完全相同**：旧NPZ共692909帧，36条LAFAN中16条多1帧，LAFAN轨迹本身也重新处理；444条mixed的量化位置一致，但速度有小差别。若要严格对照旧数据的训练趋势，另运行：

```bash
bash scripts/training/train_khan_server.sh \
  --data-dir data/npz/khan_mini/mixed \
  --run-name khan_mixed480_npz_delay_fresh_h32_4096 \
  --num-envs 4096
```

该 NPZ 命令也使用本轮首步无延迟／full-bin合并／H32配置；旧训练与本轮的踝标称常量等历史变化仍需控制，不能只凭曲线把差别归于一种改动。

## 合并规则与检查点

默认bin为50参考帧。若最终余数为1–49帧，且已有完整bin，则并入前一个bin；整50倍数不合并，不跨clip。最后一个合并bin包含51–99参考帧，但其可reset范围还须截到 `clip_last - 32`（含）。全量库最短clip为101帧，全部8452动作都有合法起点；将来长度≤32的clip保留在库内，但不作为训练物理reset候选，若全库均不合法则清楚报错。

| 8452库布局 | bin总数 | H32合法bin | 合并尾bin数 |
| --- | ---: | ---: | ---: |
| 历史半bin | 76464 | 75141 | 4095 |
| 当前整bin | 72227 | 72227 | 8332 |

库共3818945帧；H1合法起点3810493个，H32合法起点3548481个。帧数据／FK／动作时长没有改变；禁止片尾直接reset不会删除这些帧，它们仍可由更早起点播放到。`return`、`uniform`、`start`三种模式均遵循H32限制。显式forced_frame/reset_to_frame在启用限制时也验证边界；评估使用底层H1，不限制完整动作播放。

例如100k最高概率的 `burning_loop_R_001__A529` 原400–429尾bin全部不足H32；现在合并为参考帧 `[350,431)`，可reset起点350–398。另一完整bin本来最后400前的399帧同样不合法，也被H32过滤。这保证参考足够，不能保证机器人不提前摔倒，也不能保证每个replay状态都够H32：回合向前推进后仍会接近片尾，所以想象起点原有的过滤重采样继续保留。

新采样合同v4记录full-tail/H32；新首步队列语义使用robot contract v7。旧half/H1维持v3，未合并H1维持v2；旧物理配置仍可验证v6。旧策略可继续eval/play/export，但不能把旧采样统计按新bin索引直接resume。本轮应创建新run。需要严格恢复历史无延迟／half/H1训练时，可在Python入口显式给 `--env.actuator-delay False --env.delay-reset-first-step-no-delay False --env.merge-short-tail-bins True --env.tail-bin-merge-fraction 0.5 --env.reset-min-remaining-steps 1`；服务器脚本固定当前新默认，只有delay对照开关。

本轮没有改变softmax温度、混合比例、EMA或均值续尾目标。full-bin合并改变采样粒度，H32改变合法起点分布；不能宣称训练回报必然更高或更单调。2026-11-08无延迟+half-bin实验100k的benchmark为.956515/.948805、0/8452失败，是对照结果，不是当前新方案的长训结果。

## 参考实现与一次性首步成本（只分析）

InstantTrack、Uni-SuperMimic、vibed、supermimic_mjlab主要在想象replay层筛选H32，不要求物理reset剩余H32。MorphMimic默认过滤剩余H+K；关闭avoid_clip_end时还有短horizon有效长度mask/bootstrap。ScaleBFM也可在clip末端截断想象；NJU版本则clamp到末帧并继续H步。本次物理H32是针对当前固定H32训练和高概率短尾的额外约束，不能说是参考库通用默认。

参考源码：[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack/blob/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c/src/instanttrack/train.py#L425-L443)、[MorphMimic默认过滤](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/src/morphmimic/train/trainer.py#L1482-L1488)、[MorphMimic短horizon](https://github.com/lzyang2000/morphmimic/blob/770394b4a0cad9effbf807a6c03435d1699733d0/src/morphmimic/train/imagination.py#L580-L632)、[NJU首步延迟处理](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/blob/f8f60d7bbdba68468c050ef100af02d752ca238e/src/supermimic/envs/actions.py#L73-L75)。

正常cutoff仍用 `G=D+gamma**T*(R/T)/(1-gamma)`。若首步独有负成本为 `-c`，它对G的总影响为 `-c*(1+gamma**T/(T*(1-gamma)))`。gamma=.99时，T=7倍数14.315，T=32倍数3.266，T=50倍数2.210。建议先验证本轮边界修改，再决定是否把仅能明确归因于人工reset的成本从续尾均值里加回：假想公式为 `D+gamma**T*((R+c)/T)/(1-gamma)`，D仍保留实际扣分。真实失败没有续尾；不能抹去全部首步低分、真实失稳或真实终止。本轮没有实现该修正。

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

边界回归包括全／部分reset、lag0–4的首控制步所有物理子步及第二控制步、最终处理目标包含noise/scale/clip/encoder bias、未reset环境队列和时钟保持、首步action-rate保持；采样测试覆盖各尾长、H32等号边界、三种采样模式、强制起点、旧合同兼容和非法采样统计拒绝。

实际GPU训练使用全量8452紧凑PT、4环境、50步，正式网络、WM batch2048、policy batch4096、H32、更新比例和320k学习率调度保留。12GB本地流程验证启用已有activation checkpoint选项，服务器脚本不启用。短测验证实现与优化更新，不证明长训效果；需新run与无delay对照完成固定协议评估。

本轮结果：243项回归通过；全量库GPU的return/uniform/start各抽10万个起点，H32非法数均为0；实际部分reset也满足边界。完成40次WM、4次policy、80次critic优化并恢复检查点。旧100k和新smoke检查点均通过策略加载／有限值推理，旧sampler恢复到新布局被明确拒绝。记录位于 `logs/khan_fresh_delay_h32_smoke_20261109/summary.json` 与 `boundary_validation.json`。
