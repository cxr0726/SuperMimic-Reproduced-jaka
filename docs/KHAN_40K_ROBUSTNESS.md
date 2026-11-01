# 旧40k Khan策略：录像与抗推评估

本记录评估服务器提供的 `logs/Khan_lafan1_4096/ckpt_0040000.pt`，使用
**本次鲁棒性评估冻结的旧源码及18 cm宽腰盒资产**。不进行训练，也不把旧权重加载到当前15-body/base-anchor配置中。
结果、NPZ及视频是本次工作区的本地产物，不随源码分发。

旧检查点没有模型/profile SHA，不能据此确认服务器训练时的腰盒宽度也是18 cm；
本文结果只对应明确记录的本地冻结模型，不声称完整复原了服务器训练资产。

36段全库40k/100k/200k benchmark使用另一个旧Git版本中的20 cm腰盒，见
[KHAN_LEGACY_BENCHMARK.md](KHAN_LEGACY_BENCHMARK.md)。不要把本文18 cm抗推结果与其混作同一次评测。

## 冻结配置与当前训练的区别

`logs/khan_40k_robustness/baseline_manifest.json` 记录完整资产、源码文件SHA及协议。
旧package的48个Python源文件保存在 `baseline_source/src/supermimic/`，资产及全部mesh副本在
`baseline_assets/`；运行器在导入前选择这份源码，并校验SHA，不依赖当前训练树的语义。

| 项目 | 本文旧40k评估 | 当前新训练配置 |
|---|---|---|
| tracked bodies | 14，无头部 | 15，追加 `Neck_pitch_Link` |
| root / anchor | base / waist | base / base |
| COM body | `waist_yaw_Link` | `waist_yaw_Link`，独立字段 |
| 直接颈部q/qd奖励 | 无 | 0.5 / 0.1 |
| 腰盒全尺寸 m | 0.15×0.18×0.30 | 0.15×0.14×0.30 |
| 最大单步奖励 | 0.1 | 0.112 |
| 额外anchor goal / 完整goal | 9 / 649，waist frame | 9 / 649，base frame |

因此本文结果不能视为新训练配置已经收敛、也不能与新配置回报直接横比。
当前训练/eval对旧无 `robot_contract` 的Khan检查点会报错；本文运行器是明确隔离的旧语义评估入口。
所有测试使用同一份现有NPZ，没有抬升root或修复接地。

## 协议

选取三条40k整段benchmark通过的动作：`dance2_subject1`、`run2_subject4`、`jumps1_subject2`。
每个条件从首帧开始，固定seed=0，无动作/观测/reset噪声、无域随机化，足摩擦固定1.0。
episode上限设为100000秒，由动作结束或原高度/倾斜终止条件决定结果，**不是只检查15秒**。

在动作时间3秒时，按机器人当时root heading的前、后、左、右四个方向施加一次水平速度增量Δv：
直接改变free-root世界系水平qvel，保持角速度和关节速度不变。
单位为 **m/s，不是恒力N**。模型质量40.8325 kg，对应总线动量增量的等效冲量约为
`mass*Δv` N·s，例如1.5 m/s约61.25 N·s，3 m/s约122.50 N·s；没有指定一个持续施力时长。
JSON保存每个条件实际写入的世界系Δv向量，验证其模长与指定值一致。

恢复标准为相对运动参考同时满足：

- root水平位置误差<0.15 m；
- root高度误差<0.10 m；
- 与原终止逻辑同定义的投影重力z分量差<0.15；
- 上述条件连续保持0.5秒。

恢复时间从速度注入开始，计到首次连续0.5秒合格窗口的结束。不会用“root绝对速度接近0”判断
动态跑步/舞蹈/跳跃。另行要求**恢复后继续完成完整clip**，不能把短时间满足恢复条件当作整段成功。

两批测试共90个条件：首批51个为3个baseline和48个非零推扰，追加39个为另3个baseline和36个推扰。
两个批次的baseline分别保留；每个非零强度每条动作只测四个方向、同一seed和同一推扰相位。
这些方向计数**不是多seed、多相位的统计成功率或机器人通用稳定边界**。

## 完整clip结果

无推扰首批baseline均完整通过：

| 动作 | 完整控制步数 / 时长 | 归一化回报 |
|---|---:|---:|
| dance2_subject1 | 11284 / 225.68 s | 0.8994 |
| run2_subject4 | 12240 / 244.80 s | 0.9014 |
| jumps1_subject2 | 12222 / 244.44 s | 0.8578 |

下表每格为“恢复且完整完成数 / 四方向；成功条件平均恢复时间”：

| Δv m/s | dance2_subject1 | run2_subject4 | jumps1_subject2 |
|---:|---:|---:|---:|
| 0.3 | 4/4；0.50 s | 4/4；0.50 s | 4/4；0.50 s |
| 0.6 | 4/4；1.64 s | 4/4；2.37 s | 4/4；4.06 s |
| 1.0 | 4/4；3.36 s | 4/4；3.63 s | 4/4；7.83 s |
| 1.5 | 4/4；4.51 s | 4/4；3.96 s | 4/4；8.47 s |
| 2.0 | 4/4；5.52 s | 4/4；6.44 s | 4/4；7.77 s |
| 3.0 | 1/4；5.18 s | 3/4；7.06 s | 3/4；10.91 s |
| 4.0 | 1/4；21.28 s | 1/4；10.08 s | 0/4；— |

首批48个推扰全部恢复且完成；追加36个推扰中21个恢复且完成、15个失败。
追加失败大多在推后不到1秒触发倾斜终止；舞蹈向后3 m/s的条件未满足恢复标准，持续至30.24秒才触发高度终止。
可见失败样例：

| 动作与推扰 | 失败时刻（动作起点计时） | 首次触发条件 |
|---|---:|---|
| dance2_subject1，向前3 m/s | 3.86 s | root tilt |
| run2_subject4，向右3 m/s | 3.56 s | root tilt |
| jumps1_subject2，向前3 m/s | 3.72 s | root tilt |

总回报仍采用旧协议 `sum(r)/(0.1*实际执行步数)`；早失败的未执行尾段不进入这个分母。
因此选取策略和推扰强度时同时看完整完成标志、恢复时间及失败时刻，不能只比较归一化回报。

## 视频与原始产物

实心机器人是策略仿真，半透明参考用于查看跟踪。对比视频左侧为无推扰，右侧为相应推扰；
叠字注明方向、Δv和时刻。录像默认前15秒，终止状态在自动reset之前捕获，失败后冻结并标注，
失败视频仅保留终止后约1.5秒。视频窗口结束不代表完整评估停止，整段结果来自上述225–245秒运行。

| 内容 | 本地文件 |
|---|---|
| 普通舞蹈 | [baseline_dance2_subject1.mp4](../logs/khan_40k_robustness/videos/baseline_dance2_subject1.mp4) |
| 普通跑步 | [baseline_run2_subject4.mp4](../logs/khan_40k_robustness/videos/baseline_run2_subject4.mp4) |
| 普通跳跃 | [baseline_jumps1_subject2.mp4](../logs/khan_40k_robustness/videos/baseline_jumps1_subject2.mp4) |
| 三动作1.5 m/s成功恢复对比 | [recovered_push_overview.mp4](../logs/khan_40k_robustness/videos/recovered_push_overview.mp4) |
| 三动作3 m/s失败对比 | [failed_push_overview.mp4](../logs/khan_40k_robustness/videos/failed_push_overview.mp4) |

输出目录 `logs/khan_40k_robustness/` 中：

- `results.json` / `results.csv`：51条件及四方向汇总；包含峰值误差、恢复时间、实际Δv、归一化回报、完整完成情况。
- `strong_push/results.json` / `results.csv`：追加39条件，首批测量值不覆盖。
- `trace.csv`、`strong_push/trace.csv`：全片逐步root位置/高度/tilt误差与终止标志；强推trace还含求解器标志。
- `baseline_rollout.npz`、`push_rollout.npz`：普通与1.5 m/s代表动作的前15秒。
- `strong_push/push_rollout.npz`：各动作强推成功/失败代表；其中舞蹈向右4 m/s在推后21.28秒才恢复，前15秒录像不显示其恢复瞬间。
- `validation.json`：逐条件计分、实际Δv及根据原始trace重算恢复时间的校验。
- `baseline_manifest.json`、`baseline_source/`、`baseline_assets/`：冻结版本与SHA。

`peak_*_after_push` 是推扰后直到终止或clip结束的峰值，可能包括后来动作本身的难段，
不能将其全部归因于推扰瞬间；需结合相同动作baseline和原始trace判断。

### 求解器标志

首批5个条件、强推4个条件观测到 `1024=LS_ITERATIONS`，表示原配置的线搜索次数上限；
没有观测到接触/约束容量等位。全部15个强推失败条件的观测标志为0，不能把这些失败归因于容量溢出。

MuJoCo Warp的overflow位会粘滞保留，计数字段表示多少个控制步观察到置位，**不是事件发生次数**；
强推JSON另存首次观察时间。评估未更改原solver的iterations=10、ls_iterations=20。

## 复现命令

本运行器需要已经保存的冻结目录和旧40k检查点；不会用当前配置临时伪装成旧配置：

```bash
env -u PYTHONPATH uv run python scripts/analysis/eval_khan_push_recovery.py

env -u PYTHONPATH uv run python scripts/analysis/eval_khan_push_recovery.py \
  --baseline-dir logs/khan_40k_robustness \
  --out-dir logs/khan_40k_robustness/strong_push --strengths 2 3 4
```

重新渲染使用冻结资产，不重新执行策略或物理仿真：

```bash
MUJOCO_GL=egl env -u PYTHONPATH uv run python scripts/media/render_khan_robustness.py \
  --baseline logs/khan_40k_robustness/baseline_rollout.npz \
  --push logs/khan_40k_robustness/push_rollout.npz \
  --asset-dir logs/khan_40k_robustness/baseline_assets \
  --out-dir logs/khan_40k_robustness/videos --overview-name recovered_push_overview

MUJOCO_GL=egl env -u PYTHONPATH uv run python scripts/media/render_khan_robustness.py \
  --baseline logs/khan_40k_robustness/baseline_rollout.npz \
  --push logs/khan_40k_robustness/strong_push/push_rollout.npz \
  --asset-dir logs/khan_40k_robustness/baseline_assets \
  --out-dir logs/khan_40k_robustness/videos --only-failed --no-render-baseline \
  --overview-name failed_push_overview
```

原始结果来自RTX 4070 SUPER上的MuJoCo Warp。即使无随机噪声，重复GPU仿真的数值轨迹也不保证逐位相同；
这里报告单次固定条件测量，不扩展为多seed或其他推扰时刻的结论。
