# Khan 参考残差动作与编码器 goal 噪声

2026-11-09：新的 Khan `isaac_khan` 训练默认启用下一参考帧残差动作，以及只作用于 motion encoder 的 goal 噪声。本次服务器实验的两个 `.sh` 启动器默认关闭执行器延迟，显式传入 `--env.actuator-delay False`；可用 `--actuator-delay` 恢复0–20 ms延迟。Python机器人默认与旧检查点的延迟实现保留，ξ仍为91维，无延迟时delay_steps为0。训练网络规模、优化器、batch、H32、奖励和机器人参数不变。本次不增加 PD 目标裁剪，也不增加二阶动作或加速度惩罚。

## 动作公式与接口

设 decoder 输出为 `delta`，参考窗口包含 `t, t+1, ..., t+4`，则：

```text
a_total = delta + beta * (q_ref(t+1) - q_default) / alpha
q_target = q_default + alpha * a_total
         = q_ref(t+1) + 0.5 * delta       # beta=1, alpha=0.5
```

所有27个 Khan 关节的 `alpha=0.5`；`beta=1`；参考帧偏移为1。
155维 goal 按项排列，下一参考帧关节角位于 `goal[..., 47:74]`。
这与 InstantTrack 的参考残差公式、正式 preset 的 `beta=1` 一致，但其 G1 的
`alpha=0.25*effort_limit/kp` 按关节变化，不能说 Khan 的统一0.5与它逐关节相同。

完整动作合成位于 Policy 内，环境、replay、WM、动作历史及 action-rate 都继续使用
`a_total` 的现有语义。网络并没有被强制限制为小残差；参考基准只是动作参数化。
执行噪声仍按原机制在物理环境中加入，然后经过缩放和encoder bias；只有显式启用延迟时才使用延迟队列。
WM拟合使用队列之前的带执行噪声完整命令；想象仍使用不额外叠加执行噪声的完整策略动作。

导出的 ONNX 包含参考直通分支，接口仍为 packed `620 -> 27`，或 split `465+155 -> 27`。
C++ 仍执行 `q_default + action_scale * actions`，并保存网络完整输出作为 last_action；
不得在 C++ 中再次加参考关节角。ONNX 不包含训练用随机噪声。

## 隔离的参考噪声

| actor 编码器输入 | 维度 | 独立加性均匀噪声 |
|---|---:|---|
| root_pos_diff | 15 | 无 |
| ref_z | 5 | ±0.01 m |
| ref_joint_pos | 135 | ±0.05 rad |

这与实际 Isaac ProjGravObservationsCfg 的幅度一致。物理采集和想象共用生成函数；
每个环境、参考帧、关节独立采样。环境提供的原始 goal 仍保持干净，训练调用显式生成
encoder 的 noisy copy，不修改 MotionLibrary。参考直通分支、critic、奖励、终止及
WM监督状态保持干净，避免将 ±0.05 rad 直接透传到 PD 基准。

新训练的幅度由 `policy.goal_noise_z` 和 `policy.goal_noise_joint_pos` 保存。
实际采集和想象仍分别受其 `obs_noise` 开关控制；独立关闭 goal 噪声可将这两个幅度设为0。
普通 Policy forward、eval、play和export没有隐式随机加噪，即使网络处于 train mode。
真正遥操作输入中的已有噪声仍会影响参考基准，不能把这项训练增强当作参考信号滤波。

## action-rate 保持原定义

```text
r_action_rate = -0.1 * 0.02 * sum_j((a_total[t,j] - a_total[t-1,j])**2)
              = -0.002 * ||a_total[t] - a_total[t-1]||²
```

它包含一次 `dt=0.02` 乘法；没有除以dt，也没有先乘action scale。
物理环境奖励管理器乘dt，想象解析奖励直接乘dt；两侧不重复乘。
惩罚的是完整、未叠加执行噪声的策略动作，不是 decoder residual 或队列实际历史目标。
reset前动作仍初始化为0，因此首步仍有 `-0.002*||a_total[0]||²`，本次未豁免。
零残差不代表完整动作等于零。其他奖励及最大单步正奖励0.112保持不变。

## 服务器命令与对照

紧凑库，不开启参考 FK 编译：

```bash
bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_residual_goal_4096 --num-envs 4096
```

上述命令默认关闭执行器延迟；需要开启时添加 `--actuator-delay`。
需要开启参考FK编译时添加 `--reference-fk-compile`。NPZ训练入口仍为：

```bash
bash scripts/training/train_khan_server.sh \
  --data-dir data/npz/khan_mini/mixed \
  --run-name khan_mixed_residual_goal_4096 --num-envs 4096
```

两种脚本都支持以下独立开关，网络与优化参数不随之改变：

| 对照 | 额外参数 |
|---|---|
| 参考残差＋编码器goal噪声 | 无，新默认 |
| 参考残差＋干净goal | `--no-goal-noise` |
| 原动作空间＋编码器goal噪声 | `--no-reference-residual` |
| 原动作空间＋干净goal | `--no-reference-residual --no-goal-noise` |

可先添加 `--dry-run` 验证路径与命令。脚本继续保留环境变量清理、路径检查、已有
检查点保护及 `--stop-after-iterations`；dry-run不反序列化检查点。

旧检查点缺少新增字段时按原动作空间、无goal噪声加载，eval/play/export保持旧行为。
旧 decoder 已学习完整动作，不能直接打开参考残差后普通resume；这样会重复叠加参考。
从旧 clean-goal checkpoint 恢复原语义时，两个脚本都需添加上述两个关闭开关，并保证
其余物理、数据与采样配置也匹配；恢复启用延迟的旧run还需添加 `--actuator-delay`。
新残差实验应使用新的run。训练恢复会检查动作模式、
参考偏移、beta和goal噪声设置是否一致。

新字段保存在 policy config，参考默认姿态/action scale还保存在网络buffer。
机器人几何、物理合同及ξ定义不因动作重参数化改变。
其中旧 `robot_contract.observation_schema.actor_goal_noise=False` 描述环境返回的
干净goal；编码器新增增强单独记录在 policy config 和 ONNX 的 `training_goal_noise` 中。

## 本次验证

2026-11-09相关测试236项通过；1项原有G1观测测试因短rollout内reset而按既有规则跳过。
包含残差基准/时间帧、噪声幅度与独立性、critic隔离、编译输出/梯度、激活重计算、
旧checkpoint兼容、新checkpoint恢复拒绝规则、服务器脚本和packed/split ONNX。

在RTX4070 SUPER 12GB上分别测试1条LAFAN NPZ和完整8452条紧凑PT库。
两组均使用正式网络宽度、H32、默认compile和优化器，每组45个控制步、2轮更新；
测试时只将环境数设为8、WM/策略batch设为32、replay容量设为100，正式默认值未改。
每组完成20次WM优化、2次actor优化和40次critic优化，参数、优化器状态和训练损失均有限。
恢复NPZ检查点继续1步后，模型和优化器状态与保存值精确一致。

现有120k旧策略64组输入与原decoder公式逐位一致。新smoke检查点导出的packed ONNX
最大动作误差1.49e-8、PD目标误差7.45e-9 rad，随机输入、动态batch和真实参考窗口均通过。
这是功能/数值验证，不是新策略收敛或实机性能结论。
原始结果位于本地 `logs/khan_residual_goal_validation/20261109/validation.json` 及同目录日志。
上述GPU功能验证使用当时的Python延迟默认；其后服务器脚本默认关闭延迟的修改另经启动器测试验证，历史验证结果不代表关闭延迟后的收敛效果。
