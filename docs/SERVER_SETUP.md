# 服务器一键环境配置

适用目录：`/home/jaka/Workspace/SuperMimic-Reproduced-jaka`。脚本默认适用于 Ubuntu / Debian 的 Linux x86_64 服务器，需要联网；缺少系统依赖时需要 sudo 权限。

## 当前环境与驱动兼容性

2026-10-31 在本项目 `.venv` 中实际读取的版本：

| 组件 | 版本 |
|---|---|
| Python | 3.12.16 |
| uv | 0.13.6 |
| PyTorch | 2.14.1+cu130 |
| PyTorch CUDA 运行库（`torch.version.cuda`） | 13.0 |
| MuJoCo / MuJoCo Warp | 3.16.0 / 3.16.0 |
| Warp | 1.18.0 |
| mjlab | 1.6.0，git commit `8bb2bc9e187797689ec82c78c35019afd2fd0376` |

服务器的 NVIDIA 驱动 **580.178.04 可以沿用上述版本**。CUDA 13.0 对应的 Linux 驱动要求为至少 580.65.06；参见 [NVIDIA CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)。项目已有实验记录也使用了驱动 580.178、torch 2.14.1+cu130 和 Warp 1.18.0，见 [EXPERIMENTS.md](EXPERIMENTS.md)。

`nvidia-smi` 的 `CUDA Version` 表示驱动支持的 CUDA 版本，不能用它判断当前 PyTorch 的 CUDA 运行库或是否装了系统 Toolkit。本机驱动为 595.99.02，`nvidia-smi` 显示 CUDA 13.2，而 PyTorch 实际使用 CUDA 13.0。服务器无需升级到本机驱动版本，也无需单独安装系统 CUDA Toolkit / nvcc；项目锁定的 wheel 提供所需运行及即时编译组件。

Warp 1.18.0 wheel 自带的 Toolkit 版本为 **13.4**，与 PyTorch 的 13.0 不同。其已安装包说明要求 R580 或更新驱动，以及 Turing / Compute Capability 7.5 或更新 GPU；当前二进制启用了 CUDA minor compatibility。驱动低于其 Toolkit 时，Warp 会自动选择 CUBIN 编译兼容路径。不要手动强制输出新版 PTX。服务器的具体 GPU 型号尚未读取，因此脚本会实际验证可见 GPU 上的 PyTorch、编译和 Warp 仿真能力。

## 迁移与执行

先将当前项目文件复制到目标目录，包含 `pyproject.toml`、`uv.lock`、`src/`、`scripts/`、`README.md`、`assets/`。Khan_mini 的本地机器人资产已经在 `assets/robots/khan_mini/`，应一并复制。不要复制本机 `.venv/`；它包含当前机器的绝对路径，应该在服务器重建。

训练所需的 `data/`、已有检查点按需要另行复制。安装脚本不下载数据，也不需要数据即可完成 GPU 验证。整个配置过程不安装、运行或同步 `ref_thirdparty` 下各项目的环境，无需迁移该目录。

以服务器普通登录用户执行，不要对整个脚本使用 `sudo`：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
bash scripts/setup/setup_server_env.sh
```

脚本会依次：

1. 检查平台、项目文件、GPU 和驱动版本。
2. 安装缺失的系统包：`ca-certificates curl git build-essential pkg-config libegl1 libgl1 libglfw3 ffmpeg`。缺包时可能要求输入 sudo 密码。
3. 复用当前版本的 uv，或将 uv 0.13.6 安装到 `~/.local/share/supermimic/uv/0.13.6/`；不修改 shell 启动文件。
4. 用 uv 安装 Python 3.12.16，再以 `uv sync --locked --group dev` 对应的锁定安装方式创建项目 `.venv`。不修改 `pyproject.toml` 或 `uv.lock`。
5. 检查依赖一致性、确切版本、mjlab commit、项目训练模块、PyTorch CUDA 矩阵和 `torch.compile` 执行、Warp CUDA kernel、MuJoCo Warp 自由落体仿真、G1 模型资产和 EGL 离屏渲染。
6. 生成 `.venv/activate_supermimic.sh`，方便后续登录使用。

下载涉及数 GB 的 PyTorch / CUDA 组件，需要服务器能够访问 PyPI、PyTorch wheel 源、NVIDIA Python 源、GitHub 和 uv 下载源。首次编译验证可能需要几分钟；日志保存在 `logs/setup_server/`。网络中断后可直接重跑，已下载的 uv 缓存会复用。检测到版本不符、损坏或从其他目录复制的 `.venv` 时，脚本将它移动到 `logs/setup_server/venv_backup_*/` 后重建，不直接删除。

每次重新登录后：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
source .venv/activate_supermimic.sh
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

激活助手会清除容易引入 ROS / 系统包的 `PYTHONPATH` 和 `PYTHONHOME`，并默认设置 `MUJOCO_GL=egl`，适用于无显示器服务器。需要本地原生窗口时，先确保有可用图形桌面，再为对应命令指定 `MUJOCO_GL=glfw`。

已有 LAFAN1 数据时，可进一步验证完整训练链路：

```bash
python scripts/smoke_train.py --run-name server_smoke
```

## 可选参数

```bash
# 系统依赖已经安装，或者由管理员统一管理
bash scripts/setup/setup_server_env.sh --skip-system-packages

# 服务器只用于计算，允许跳过 EGL 渲染验证
bash scripts/setup/setup_server_env.sh --skip-render-check

# 仅重新检查当前项目环境，不安装或同步依赖
bash scripts/setup/setup_server_env.sh --check-only

# 项目实际位于其他目录
bash scripts/setup/setup_server_env.sh --project-dir /another/path/SuperMimic-Reproduced-jaka
```

如果默认 EGL 验证失败，日志会指明阶段；NVIDIA EGL 实现通常由服务器驱动提供。可以先修复驱动的图形组件，或者在不需要离屏渲染时使用 `--skip-render-check`。跳过渲染检查只证明计算和仿真可用。

本脚本已在当前机器运行 `--check-only`，通过全部 GPU、编译、仿真与 EGL 检查；服务器端的下载、系统包安装及硬件验证需要在目标服务器执行。
