# 紧凑参考库 FK 编译

已实现 CUDA compact 参考 body 查询的整体编译，使用 `--env.reference-fk-compile` 显式开启。根据本机短训练尚未确认整体提速的结果，默认保持原 eager 路径。`.pt` 文件格式、存储精度、机器人参数、观测、奖励、随机化和网络优化参数均不改变。非 compact 完整库直接读取预计算 body 数组；CPU compact 查询继续使用 eager。

## 编译范围与精度

编译入口位于 `MotionLibrary._compact_bodies_selected`，覆盖以下完整链路：

```text
frame 索引 → 紧凑参考 gather／fp16 转 fp32
→ root 世界角速度转 body 坐标系
→ 各 link 的位置、四元数、线速度、角速度 FK
→ tracked body 选择
```

`torch.compile(fullgraph=True, dynamic=True)` 捕获完整计算图；动态 batch 维覆盖环境当前帧查询和展平的想象窗口查询。标量、空查询和单帧查询保留 eager，避免为 0／1 特化额外编译。外部多维索引先展平，输出再恢复原 batch 形状。默认策略更新中参考查询数为 `(32 + 1) × 4096 = 135168`，每个环境控制步则按环境数量查询；不会每步遍历整个动作库。

编译入口使用静态 Python body 索引元组，在 FK 输出堆叠前选择所需 body，使编译器可以裁掉未使用的叶节点输出。普通 eager 对照保留原来的全 body FK 后 GPU 张量索引路径；两者保持相同的 body 顺序和计算公式。`G1Kinematics.forward` 新增可选 `body_indices` 参数，默认 `None` 保留全部 body 输出，现有预测机器人调用不变。

仅参考动作侧被编译。预测机器人侧用于可微奖励和 critic 的 FK 保留原有编译／梯度链路。参考 FK 使用 `torch.no_grad()`，不使用 `inference_mode()`，因为其输出随后仍参与 actor 的有梯度运算。明确关闭 `triton.cudagraphs`，避免下一次查询覆盖仍被 rollout／critic 保留的参考输出。

packed 数据仍按原配置以 fp16 存储，读取后以 fp32 进行 FK；本次未启用 AMP 或 fast math。逐元素四元数运算不使用 TF32 矩阵乘法。融合可能改变浮点舍入和结合顺序，因此验证的是明确容差内的数值一致，而非逐 bit 相同；原有 fp16 数据量化误差与编译误差应分别评估。

实现时额外发现并修复了一个动态捕获问题：Python 整数关节类型与 MuJoCo 的 pybind enum 比较，在 Dynamo 动态追踪中可能错误进入 fixed-body 分支。现在计算热路径与模块级普通整数 `_HINGE_TYPE` 比较。原来的 eager／静态捕获比较正确；这是为新增动态编译修复的兼容问题，并不说明此前训练中的 FK 错误。回归测试采用非零关节姿态和速度，避免中立姿态掩盖该错误。

## 使用及关闭方式

本轮验证使用的参考库为：

`/home/cxr0726/workspace/frax/data/library/khan_combined_8452_compact.pt`

该库包含 8452 条动作、3818945 帧、50 Hz、67 维 fp16 参考状态。服务器上的实际路径如下；其余本机验证命令保留原始路径以记录测试来源。服务器完整训练、转换命令见 [KHAN_PT_TO_NPZ.md](KHAN_PT_TO_NPZ.md)。

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -u -m supermimic.scripts.train \
  --robot khan_mini --run-name khan_bones_compact \
  --data.library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --env.num-envs 4096 --env.reference-fk-compile
```

`--data.library` 自动开启 compact。参考 FK 编译单独选择；进行 eager 对照时，移除 `--env.reference-fk-compile` 或使用：

```text
--env.no-reference-fk-compile
```

启动时 `MotionLibrary` 摘要显示 `reference FK=compiled (dynamic batch)` 或 `reference FK=eager`。编译为惰性执行，首次参考查询会包含冷编译时间。这个开关不进入机器人语义 contract，不改变 checkpoint 的观测、动力学或奖励兼容条件。

## 复现验证

数值与接口测试覆盖 G1／Khan、MuJoCo 双精度 oracle、动态捕获的铰链运动、不同 batch 及边界索引、旧输出生命周期、奖励／critic 特征和梯度：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python -m pytest -q tests/test_reference_fk_compile.py
```

同一 `.pt`、同一批 frame 的组件 A/B，分别报告冷编译、预热后的 CUDA event 时间、同步墙钟时间及最大／RMS 数值误差：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmarks/benchmark_reference_fk.py \
  --library /home/cxr0726/workspace/frax/data/library/khan_combined_8452_compact.pt \
  --robot khan_mini --num-envs 4096 --policy-batch 4096 --horizon 32 \
  --samples 30 --warmup 5 \
  --output logs/reference_fk_compile/benchmark.json
```

保持正式网络、batch、优化器和更新比例，仅缩小环境数与执行长度的训练 smoke：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv/bin/python scripts/smoke_train_khan.py \
  --library /home/cxr0726/workspace/frax/data/library/khan_combined_8452_compact.pt \
  --num-envs 32 --steps 150 --activation-checkpoint --reference-fk-compile \
  --run-name khan_compact_fk_smoke
```

eager 对照增加 `--no-reference-fk-compile`，并使用另一个 `--run-name`，避免覆盖已有结果。前 40 步为 warmup；它验证实际更新、有限数值和 checkpoint 恢复，不是长期收敛实验。激活重计算仅用于本机 12 GB 显存测试；正式训练命令仍使用仓库默认的关闭设置。

## 本轮组件实测

2026-11-07，RTX 4070 SUPER，PyTorch 2.14.1。同一完整 8452 条 packed 库、同一批索引，交替测量 eager／compiled 各 30 次；计时包含 GPU 同步。想象窗口仅从具有完整 32 步后续参考的起点产生，使用 time-major 展平布局。

| 查询 | 原 eager 墙钟中位数 | compiled 墙钟中位数 | eager / compiled |
| --- | ---: | ---: | ---: |
| 当前参考，4096 帧 | 6.940 ms | 0.792 ms | 8.76 倍 |
| 想象参考，33×4096 帧 | 11.689 ms | 11.889 ms | 0.983 倍，基本持平、略慢 1.7% |

没有将组件加速倍数当作整体训练倍数。按 UTD=0.4，仅参考查询部分估计每个外循环节省 `6.940−0.792+0.4×(11.689−11.889)=6.068 ms`，不包含 reset 次数变化及其他训练工作。整体还包含物理仿真、世界模型、actor／critic 前后向和优化器更新。

首次真实查询包含约 102 秒的编译／缓存加载成本，首次大 batch 调用约 8 秒；这些时间不计入上表稳态中位数。实际捕获 2 个完整图，无 graph break；预热后反复切换两种形状未增加图数。静态 body 选择前的第一版大查询慢约 10%，保留了诊断数据，最终实现采用上述静态选择。

编译与原 eager 对同一 fp16 解量化输入的最大绝对差：位置 `1.431e-6 m`、四元数分量 `1.014e-6`、线速度 `1.050e-5 m/s`、角速度 `2.242e-5 rad/s`。两种路径均为 fp32 FK；数据存储及旧输出在多次调用后保持不变。

原始数据和全部样本计时在 `logs/reference_fk_compile/benchmark_reference_fk_selected.json`。

## 训练及下游数值验证

最终数值／接口专项测试 16 项全部通过，含 G1 与 Khan 的 CUDA 对照、独立 MuJoCo oracle、输出保留、配置传递，以及奖励、critic 特征、状态和动作梯度。日志为 `logs/reference_fk_compile_pytest_final.log`。切换最终默认值为关闭后，又复核 CPU 配置与回归测试；显式开启的 CUDA 数学路径不变。

使用同一完整库、32 环境、150 步，两种路径各完成 440 次 WM、44 次策略、880 次 critic 更新，所有已记录 TB 标量、参数和优化器状态有限；实际清空／扰动后的 checkpoint 恢复通过。正式网络、batch、H32、优化器、UTD、学习率调度不变；两组均使用本机 12 GB 所需的激活重计算。配置与数据汇总一致，只有 run 名称及 FK 开关不同。

排除首个 50 步窗口后，独立短训练的实测结果如下。它不是多次重复或服务器 4096 环境的结论。

| 指标，51–150 步 | eager | compiled |
| --- | ---: | ---: |
| 实际吞吐 | 1.871 it/s | 1.749 it/s |
| sim 每外循环 | 23.781 ms | 18.169 ms |
| WM 每外循环 | 143.137 ms | 142.978 ms |
| 策略／critic 每外循环 | 367.347 ms | 410.412 ms |

**本轮整体吞吐下降 6.52%，不能据此声称整轮训练已加速。** 两次独立训练输入和权重会逐渐分化，桌面运行负载亦未完全受控。已检查参考输出的小／大 batch 均连续，stride 与原 eager 一致，没有证据支持添加额外 contiguous 拷贝。

另做同进程、同模型权重、同一批参考起点及历史、相同 RNG 的交替 BPTT 对照，包含 H32、4096 batch、critic bootstrap 和激活重计算，不包含后续 critic 拟合及优化器 step：

- 目标函数和最终想象状态相同；reward 最大差 `5.216e-8`，critic 末帧误差特征最大差 `1.034e-6`；全部策略梯度最大差 `2.911e-11`，相对 L2 差 `6.446e-8`。所有模型权重在测试前后 hash 不变。
- 预热后 3 次墙钟样本，eager 为 `1185.86 / 1029.39 / 1031.50 ms`，compiled 为 `1012.89 / 3107.28 / 1021.88 ms`。中位数约快 0.93%，但存在不可忽略的异常停顿，不能将中位数作为稳定提速保证。
- compiled 的异常样本中，参考 FK 仅用 `12.22 ms`，额外约 2.1 秒发生在其后的想象前向；未用 GC 回调或完整 profiler 进一步定因。不能确认是 GC、调度还是其他运行时因素，也不能用未证实的解释抹去慢样本。

报告分别为 `logs/reference_fk_compile/training_ab.json` 与 `logs/reference_fk_compile/benchmark_reference_fk_imagination.json`。复现受控 BPTT：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  .venv/bin/python scripts/benchmarks/benchmark_reference_fk_imagination.py \
  --library /home/cxr0726/workspace/frax/data/library/khan_combined_8452_compact.pt \
  --checkpoint logs/reference_fk_compile/compiled_training/ckpt_latest.pt \
  --output logs/reference_fk_compile/imagination_recheck.json
```

**最终决策：编译实现和显式开关保留，默认关闭。** 在目标服务器先运行组件及真实训练 A/B；确认整体吞吐后再开启。数值测试与短训练支持计算及更新流程正确，不能证明长程收敛、最终回报或跨 GPU 的训练速度完全等价。

## 参考实现与说明

- [Reproduced 的 compact body 查询](/home/cxr0726/workspace/frax/thirdparty/SuperMimic-Reproduced/src/supermimic/envs/mdp/motion_library.py:182)与 [retargeted 的查询](/home/cxr0726/workspace/frax/thirdparty/supermimic-retargeted/src/supermimic/envs/mdp/motion_library.py:184)直接执行参考 FK，没有整体 `torch.compile`。
- [Uni-SuperMimic 编译入口](/home/cxr0726/workspace/frax/thirdparty/Uni-SuperMimic/src/unisupermimic/rl/algos/supermimic/learner.py:220)编译奖励／critic 特征函数；[奖励中的参考 body 查询](/home/cxr0726/workspace/frax/thirdparty/Uni-SuperMimic/src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py:193)位于该外层编译范围内。
- PyTorch 2.14 官方：[torch.compile 参数](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)、[浮点数值一致性的限制](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)、[CUDA Graph 输出生命周期](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)。
