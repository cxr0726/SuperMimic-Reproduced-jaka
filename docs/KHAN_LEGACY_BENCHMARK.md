# Khan 旧策略全库 benchmark：40k、100k 与 200k

2026-11-01，新增评估 `ckpt_0100000.pt`。文件内部 `iteration=100000`，即100k/10万，
不是10k；此前 `ckpt_0200000.pt` 是200k/20万。

## 隔离旧配置

`scripts/analysis/eval_legacy_khan_benchmark.py` 从已提交版本
`fcad4a5f69b3b3c896b559314d1cff430b81990f` 导出源码、MJCF、profile及mesh，
再从独立目录导入原benchmark。不会切换分支或覆盖工作区文件。

| 项目 | 本次旧策略 benchmark | 当前服务器新训练 |
|---|---|---|
| tracked bodies | 14，不含头部 | 15，包含 `Neck_pitch_Link` |
| root / anchor | `base_link` / `waist_yaw_Link` | `base_link` / `base_link` |
| COM随机化对象 | `waist_yaw_Link` | `waist_yaw_Link` |
| 直接颈部位置/速度奖励 | 无 | 权重0.5/0.1 |
| 腰盒全尺寸 m | 0.15×0.20×0.30 | 0.15×0.14×0.30 |
| 最大单步奖励 | 0.1 | 0.112 |
| anchor goal / 总goal | 9 / 649 | 9 / 649 |

旧checkpoint未保存模型SHA，因此这里明确复现的是指定Git版本，不能从checkpoint证明服务器
当时的所有资产字节完全一致。40k抗推测试另使用18 cm宽腰盒，见
[KHAN_40K_ROBUSTNESS.md](KHAN_40K_ROBUSTNESS.md)，不是本文全库benchmark的配置。

本次使用现有36段Khan LAFAN NPZ，共396990帧，50 Hz；36并行环境，seed=0。
每段从首帧开始，确定性策略；关闭obs/action/reset噪声和域随机化，足摩擦固定1.0。
保留原高度/倾斜终止条件；由整段结束决定完成，不截成10秒训练回合，不计自动重置后的轨迹。
未抬升动作，也未改变NPZ。

## 结果

40k、200k为此前保存的相同旧配置全库结果，100k为本次新测；没有重新运行前两项。
训练回报来自同一TensorBoard在对应checkpoint精确step处的值。

| checkpoint | 训练回报 | benchmark均值 / 步加权 | 摔倒动作数 | 执行步数 / 完整步数 | 覆盖率 |
|---|---:|---:|---:|---:|---:|
| 40k（历史） | 0.8196 | 0.9146 / 0.9187 | 2/36 | 394703 / 396990 | 99.42% |
| 100k（本次） | 0.8345 | **0.9278 / 0.9304** | **0/36** | **396990 / 396990** | **100%** |
| 200k（历史） | 0.8429 | 0.8842 / 0.9253 | 34/36 | 88891 / 396990 | 22.39% |

本次100k没有出现200k的频繁摔倒，36段均完成。它还说明这批旧模型的长片稳定性并非从40k到200k
单调变化；不能单凭短训练回合的回报增长选择最终策略。

原benchmark归一化分母使用实际执行步数，早失败后尚未执行的尾段不计入。
因此200k的步加权0.9253不能解释为完成了大部分动作。把未执行部分计零的整库步加权回报分别为
40k 0.9134、100k 0.9304、200k 0.2072。摔倒数及覆盖率需与回报一起查看。
GPU物理重复执行存在数值差异，这些是各次固定条件结果，不是多seed统计。

100k checkpoint的policy、WM和critic参数均有限；独立重算逐动作回报分解、汇总及步数通过，
冻结源码与资产SHA在运行前后相同。当前训练绑定另行确认仍为15 body/base anchor/waist COM。

## 复现

先确保本地Git历史包含上述旧提交；浅克隆可能需要先补取相应历史。数据和checkpoint独立复制，
不随代码仓库提交。

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python -u \
  scripts/analysis/eval_legacy_khan_benchmark.py \
  --ckpt logs/Khan_lafan1_4096/ckpt_0100000.pt \
  --npz-dir data/npz/khan_mini/lafan1 --num-envs 36 --device cuda:0 \
  --out logs/Khan_lafan1_4096/analysis/benchmark_ckpt_0100000.json
```

输出JSON旁保存 `.manifest.json`、`.validation.json` 和 `_legacy_snapshot/`。
manifest记录checkpoint、各NPZ、源码及资产SHA。快照已存在时拒绝覆盖；复测选用新的 `--out` 文件名。
本机控制台另重定向到同目录 `.log`；这些运行产物均不提交Git。

批评估旧run必须显式选择旧版本，新配置checkpoint默认使用当前benchmark：

```bash
SKIP_EXISTING=1 bash scripts/analysis/eval_khan_checkpoints.sh \
  logs/Khan_lafan1_4096 data/npz/khan_mini/lafan1 36 \
  --legacy-revision fcad4a5f69b3b3c896b559314d1cff430b81990f
```

## 服务器新训练

当前训练代码、模型与checkpoint校验保持新版语义，旧100k不用于续训：

```bash
bash scripts/training/train_khan_server.sh \
  --num-envs 4096 --run-name khan_base_head_neck_v2
```

正式网络、优化器、batch、H、更新比例沿用原仓库，激活重计算默认关闭。
环境安装、数据同步和50步预检见 [KHAN_SERVER_TRAINING.md](KHAN_SERVER_TRAINING.md)。
