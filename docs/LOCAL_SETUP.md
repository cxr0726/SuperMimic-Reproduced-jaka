# 本机训练环境与最小验证

配置日期：2026-10-31（Asia/Shanghai）。使用现有 `data/npz/lafan1/`，不下载或重新转换数据，不下载 BONES-SEED。

## 环境

依赖按仓库 `uv.lock` 安装到 `.venv/`，不修改系统 Python 或锁文件：

```bash
env -u PYTHONPATH uv sync --locked --group dev
```

系统为 Ubuntu 22.04.5 LTS（x86_64），GCC/G++ 11.4.0，EGL 库已具备。本机为 NVIDIA GeForce RTX 4070 SUPER，显存 12,282 MiB，计算能力 8.9，驱动 595.99.02。驱动报告的 CUDA 13.2 表示驱动支持版本；PyTorch 使用锁定的 CUDA 13.0 运行库。训练使用 MuJoCo Warp 的 CUDA 仿真，无需打开窗口。

| 组件 | 版本 |
|---|---|
| uv | 0.13.6 |
| Python | 3.12.16 |
| PyTorch | 2.14.1+cu130 |
| MuJoCo / MuJoCo Warp | 3.16.0 |
| Warp | 1.18.0 |
| mjlab | 1.6.0，git commit `8bb2bc9e187797689ec82c78c35019afd2fd0376` |
| NumPy / TensorBoard / tyro | 2.5.4 / 2.21.0 / 1.0.16 |

首次安装时官方 PyPI 下载较慢，因此将锁定的大包从原 URL 独立分段预下载，逐包核对文件大小与 SHA-256 后，使用 `uv pip install --require-hashes --no-deps --no-index --find-links ...` 预装。随后上述原始 `uv sync --locked --group dev` 成功完成，`uv pip check` 确认 156 个包全部兼容。`pyproject.toml` 与 `uv.lock` 保持原样。下载校验清单位于 `logs/setup_20261031/wheels/verified_wheels.json`。

后续命令在仓库根目录执行。`uv run` 自动使用 `.venv`；`env -u PYTHONPATH` 避免系统其他 Python 包进入项目环境。

## 最小真实训练

```bash
env -u PYTHONPATH uv run python scripts/smoke_train.py
```

重复执行时指定新目录，脚本会拒绝覆盖已有检查点：

```bash
env -u PYTHONPATH uv run python scripts/smoke_train.py --run-name smoke_lafan1_repeat
```

数据或日志不在默认目录时，可指定 `--data-dir` 和 `--log-root`。

| 配置 | smoke 使用值 |
|---|---|
| 参考动作 | `walk1_subject1.npz` |
| 并行环境 / replay capacity | 32 / 每环境 64 行 |
| 仿真步数 / 预热 | 16 / 12 |
| 更新比例 | 每步 1 次，从第 12 步开始，共 5 次 |
| 世界模型 | 隐层 `(128,128)`，batch 32，每次更新 1 batch，8 步自回归窗口 |
| 策略 | 编码器和解码器均 `(128,64)`，batch 32，想象 horizon 4 |
| Critic | 保留网络默认值；1 epoch、1 minibatch |
| 观测 / 目标 | 保留 10 帧历史、930 维本体观测、680 维目标 |
| PyTorch compile | 关闭，避免最小测试花费大量时间在首次编译 |
| 噪声 / 域随机化 / 重置采样 | 保留训练默认设置 |

MuJoCo Warp 仍会编译并执行真实 CUDA 仿真内核。脚本检查：

- 世界模型、策略和 critic 的参数均有限，且三者参数都实际发生变化。
- 三个 Adam 优化器均执行更新。
- 完成 16 次向量环境步、5 次完整训练更新。
- 检查点保存成功，并通过 `Trainer.load()` 恢复模型、优化器及采样器状态。
- TensorBoard 数值均有限，且包含世界模型损失、策略梯度范数、critic 损失。

输出位于 `logs/smoke_lafan1/`：`config.json`、`summary.json`、`ckpt_latest.pt`、`ckpt_0000016.pt` 和 `tb/`。JSON 汇总包含配置、版本、验证结果、训练指标、耗时和 PyTorch CUDA 峰值显存。显存统计只覆盖 PyTorch allocator，不包含 Warp、驱动或桌面程序的分配。

训练前几次控制台日志可能出现 `train_ret nan (n=0)` 或 `nse8 nan`：前者表示尚无完成的 episode，后者表示 replay 小于在线 8 步误差诊断的 2,048 行门槛。这些是控制台占位值；脚本会另外检查实际训练指标和全部 TensorBoard scalar 的有限性。

该测试验证训练链路可执行，不用于判断策略收敛或复现 README 的回报水平。

## 测试记录

最小训练实际通过，结果来自 `logs/smoke_lafan1/summary.json`：

| 验证项 | 实际结果 |
|---|---|
| 向量环境步 / 完整更新 | 16 / 5，即 512 次环境转移 |
| 世界模型 / 策略 / critic | 分别有 12 / 12 / 6 个参数张量改变，全部参数有限 |
| 三个 Adam 优化器 | 每个参数状态的 step 均为 5 |
| TensorBoard | 全部 scalar 有限，必需的损失与梯度标签齐全 |
| 检查点 | 两份约 6 MiB 的文件均已保存；`Trainer.load()` 恢复验证通过 |
| 脚本计时 | 9.43 秒，从 Trainer 初始化开始，不含 Python 导入和依赖安装 |
| PyTorch 显存峰值 | allocated 50,503,168 bytes（48.16 MiB），reserved 60 MiB；不含 Warp/驱动分配 |

第 16 步日志区间的训练指标（后四次更新的均值）：世界模型 loss `225.7048`、一步 NSE `0.54373`、策略梯度范数 `1.5369e-7`、critic loss `0.00322463`。该短测试的指标只用于确认数值有效和梯度更新，不用于衡量收敛。

完整测试集命令（限制首次 PyTorch 编译的并行度以控制主机内存）：

```bash
env -u PYTHONPATH TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 \
  uv run pytest tests -q --junitxml=logs/setup_20261031/pytest.xml
```

实测结果：**24 passed, 1 warning in 28.76s**，没有跳过或失败。测试包含 CPU/GPU FK 数值一致性、参考动作转换、环境与 replay、解析奖励/观测/终止、compact library 一致性、GAE、完整训练和恢复，以及 300 次世界模型拟合的误差下降检查。唯一提示是第三方 `torch.jit.script` 弃用警告，不影响运行；测试中的默认 PyTorch compile 路径也已执行。

安装、smoke 及测试日志位于 `logs/setup_20261031/`。新增脚本已通过 `ruff check`，代码差异通过 `git diff --check`。临时下载的 wheel 和分段文件在安装验证后清理，哈希清单与日志保留；已安装环境及 uv 缓存保留。

完整数据字段、坐标系、关节顺序及训练使用链路见 [LAFAN1_DATA.md](LAFAN1_DATA.md)。
