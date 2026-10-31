# Khan 服务器训练：原优化参数与anchor观测

本文件对应 `scripts/training/train_khan_server.sh`。脚本采用原仓库的单GPU LAFAN
网络隐藏层与优化配置，选择 Khan 模型/动作；默认4096并行环境，开启额外anchor观测。
不启用激活重计算、AMP、微批或梯度累积，不缩小训练batch和想象H。

此前脚本把原默认参数逐项写在命令里，现在已简化为复用 `TrainConfig` 默认值，只传
机器人、路径、环境数和 `--env.anchor-obs`。CPU测试解析实际命令，对照原默认配置，
确认隐藏层、学习率、batch、H、UTD、奖励/DR等未因简化而改变。
Anchor是按用户要求的明确观测增强：Khan goal从640变为649（窗口末尾追加位置3维+姿态6维），
critic输入从742变751；本体观测870、decoder输入902和所有hidden层不变。
这与参考迁移仓库的anchor语义一致；不保证尚未正式训练的Khan性能一定提高。

## 35/250与参考仓库的来源

能核实的历史是：

1. 本仓库首个提交 `afda1f4a7e29f8d47fdb831b90ff408ea98de685` 已直接使用
   `nconmax=35, njmax=250`，该版本 `tracking_env_cfg.py:237`；未见本仓库计算或调优过程。
2. 锁定mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376` 的通用tracking模板也使用
   同样容量和timestep/solver配置，本仓库沿用了该模板。
3. mjlab提交 `1bf8d1d58bac60d96d140862940de2451afbb652`
   （`Upgrade to mujoco_warp fb9bf88.`）将tracking的 `nconmax=150_000` 改成35，
   同一提交明确了nconmax按world预算共享接触池、njmax为每world约束上限的语义。
   `njmax=250` 更早已存在于mjlab最初公开提交 `a61e0ff`。
4. 公开历史没有给出为什么精确选35、250的峰值统计、容量扫描或公式。它们是上游配置预算，
   不能说由G1关节数推导，也不能称为最小安全值。

例如 `35*4096=143360` 与旧150000的规模接近，但没有文字证据证明35就是这样取整出来的。

用户提供的 `ref_thirdparty/supermimic-retargeted` 在
`src/supermimic/envs/tracking_env_cfg.py:252` 统一设置：

| 机器人 | nconmax | njmax |
|---|---:|---:|
| G1 | 35 | 250 |
| A3 | 256 | 1024 |
| X2 Ultra | 256 | 1024 |

这两个非G1模型使用同一组常量，公开代码/提交说明/文档没有容量测量依据。
该仓库完整训练日志未公开，不能断言作者从未测试；也不能把256/1024当成Khan的最低安全值。
参考A3/X2还关闭了自碰撞，而当前Khan开启，因此容量不能仅按型号直接照搬。

当前Khan保持128/512，明确属于本次迁移的保守预留，未证明35/250不足或128/512最小。
这与网络/优化超参数分开记录；没有再次无依据地改成256/1024。

## 需要同步到服务器的文件

本轮训练适配和anchor启动脚本随本次提交推送到GitHub。服务器已有仓库可用
`git pull --ff-only origin main` 获取；首次部署可克隆 `git@github.com:cxr0726/SuperMimic-Reproduced-jaka.git`。
也可同步完整工作区或使用部署包。
必须包含 `src/`、`scripts/`、`assets/robots/khan_mini/`、`pyproject.toml`、`uv.lock`、
`README.md` 和 `.python-version`。

本次会话的本地交付还提供 `dist/khan_server_code_20261031.tar.gz`（不随Git提交），
包含当前源码、脚本、Khan资产、文档和测试，
内部 `DEPLOYMENT_MANIFEST.json` 列出文件SHA256。它不包含数据、`.venv`、Git目录、运行日志或ref。
可将该压缩包复制到服务器，在项目父目录解压：

```bash
mkdir -p /home/jaka/Workspace
tar -xzf khan_server_code_20261031.tar.gz -C /home/jaka/Workspace
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
```

数据需另外同步整个 `data/npz/khan_mini/lafan1/`（36个NPZ，约969 MiB），保持这个相对路径。
不要把本机 `.venv` 复制过去；服务器重建环境。独立Khan资产已完整打包，无需ref目录、原始PKL或URDF源目录。

## 配置环境并检查命令

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka
bash scripts/setup/setup_server_env.sh --project-dir "$PWD"

# 只检查路径、打印完整命令，不启动CUDA/训练，不创建训练日志
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_anchor_1024 --dry-run
```

安装细节见 [SERVER_SETUP.md](SERVER_SETUP.md)。脚本从自身路径定位项目，换项目目录无需改源码。
使用项目 `.venv/bin/python`，清除PYTHONPATH/PYTHONHOME及遗留torchrun配置；只启动单GPU。
`CUDA_VISIBLE_DEVICES` 可用于选卡，例如命令前加 `CUDA_VISIBLE_DEVICES=0`。

## 先验证50步，再继续正式训练

服务器实际显存尚未读取，不保证24 GB一定容纳Khan。原G1实验记录约24 GB，原计划预估25–27 GB，
都不是Khan实测保证。先执行50步以覆盖首次完整WM、actor反向、critic及Adam状态分配：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_anchor_1024 --stop-after-iterations 50
```

成功后日志应显示 `it 50 ... updates 4`，并保存
`logs/khan_lafan1_anchor_1024/ckpt_latest.pt`。然后从该检查点继续到正式320000步：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 1024 --run-name khan_lafan1_anchor_1024 \
  --resume logs/khan_lafan1_anchor_1024/ckpt_latest.pt
```

恢复时重新预热40步填充新进程replay，沿用已有模型/Adam状态/全局迭代和采样器统计。
恢复时使用同一套36段数据。要改成原4096并行数，将上述1024全部改成4096并使用相应run-name。
也可直接从头完整运行：

```bash
bash scripts/training/train_khan_server.sh --num-envs 4096 --run-name khan_lafan1_anchor_full
```

这是前台命令，不会替用户自动启动后台任务。控制台可直接查看，TensorBoard位于
`logs/<run-name>/tb`，解析后的完整配置位于 `logs/<run-name>/config.json`。
已有检查点时脚本要求明确 `--resume` 或新run-name，避免静默覆盖。
旧640维Khan检查点不能直接恢复到649维训练；恢复前会检查anchor及goal/history布局。
要复现旧配置可显式加 `--no-anchor-obs`，不应对旧模型静默补维或部分恢复。

## 从原配置复用的主要参数

| 项目 | 值 |
|---|---|
| 世界模型 / termination hidden | 2048×3 / 64×2 |
| 策略encoder、decoder hidden | 各2048/1024/512/256 |
| latent / critic hidden | 32 / 256、128 |
| WM batch / window / 每轮批数 | 2048 / 8 / 10 |
| policy batch / H | 4096 / 32 |
| critic | 5 epochs × 4 minibatches |
| UTD / actor:WM外轮 | 0.4 / 1:1 |
| warmup / replay | 40步 / 每环境1000行 |
| WM、actor、critic LR | 单GPU均2e-4 |
| gamma / lambda | 0.99 / 0.95 |
| 训练长度 / LR衰减 | 320000步 / 250000步余弦衰减到0.03× |
| obs/reset/域随机化/执行噪声 | 按原配置开启 |
| anchor observation / 激活重计算 | 开启（本次明确变更） / 关闭 |
| torch.compile | default |

`--stop-after-iterations` 只限制当前进程执行步数，不缩短学习率衰减时间基准。
脚本复用原配置，不提供batch/H等参数覆写入口。完整观察、奖励和DR列表见
[KHAN_TRAINING.md](KHAN_TRAINING.md)。12 GB失败记录见 [KHAN_MEMORY.md](KHAN_MEMORY.md)。

## 本地验证边界

启动脚本的CPU测试会解析生成命令并与真实 `TrainConfig` 对照，确认语法和全部正式参数。
还测试了路径含空格、dry-run不执行解释器、恢复与短测、缺失环境/资产/数据、检查点覆盖保护。
本轮没有在服务器执行训练，也没有再次在本地12 GB设备启动必然失败的完整原配置。

当前anchor版本完整测试集 **132 passed，5 warnings，33.08 s**，无失败或跳过。
包含原G1训练/恢复回归、Khan实际CUDA649维观测（噪声开/关）与FK解析对齐、
正式hidden层649维策略/751维critic反传、旧640维及其他布局不匹配检查点的提前拒绝。
12 GB本地验证不能替代服务器上的正式batch/H完整训练，模型效果尚未验证。
