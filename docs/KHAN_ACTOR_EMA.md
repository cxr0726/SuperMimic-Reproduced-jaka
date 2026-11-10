# Khan：紧凑 encoder 与 Actor EMA

2026-11-10。新训练默认配置：

| 项目 | 当前值 |
|---|---|
| encoder | `155 → 1024 → 512 → 256 → 32` |
| decoder | `497 → 2048 → 1024 → 512 → 256 → 27` |
| actor可训练参数 | **4,605,243**，原为6,863,163，减少约32.9% |
| EMA decay | `0.995`，每次actor optimizer更新后执行 |
| 训练采集／想象／actor optimizer | live actor |
| eval／play／ONNX默认 | checkpoint中有EMA则用EMA，否则用live |
| 输入／输出 | packed620（goal155＋prop465）→完整action27 |

WM、critic结构、奖励、obs、DR、H32、batch、UTD及学习率调度保留当前设置。参考残差仍为 `q_target=q_ref(t+1)+0.5*decoder_residual`，encoder-only goal噪声不进入干净reference skip。服务器脚本执行器delay仍默认关闭，参考FK仍默认eager。

EMA只平均可训练参数；q0、action_scale和其他buffer逐次精确复制。shadow不参加优化或梯度同步。与InstantTrack一致，首次actor更新后直接复制live，之后执行 `ema=.995*ema+.005*live`；warmup、WM及critic更新都不单独推进EMA计数。它不是实时action滤波，不增加部署推理网络。

## 服务器训练

在服务器拉取当前提交后，新建run：

```bash
cd /home/jaka/Workspace/SuperMimic-Reproduced-jaka

bash scripts/training/train_khan_compact_server.sh \
  --library /home/jaka/Workspace/SuperMimic-Reproduced-jaka/data/pt/khan_combined_8452_compact.pt \
  --run-name khan_8452_residual_ema_enc1024_4096 \
  --num-envs 4096
```

可先追加 `--dry-run` 检查路径和实际命令；正式启动移除。FK编译对照可追加 `--reference-fk-compile` 并使用独立run名。脚本保留环境变量清理、路径检查和已有检查点保护，不下载数据。

NPZ启动器 `scripts/training/train_khan_server.sh` 读取同一套网络/EMA默认值，不额外复制超参数清单。

## 保存与恢复

完整checkpoint：

- `policy`：live actor，始终用于恢复训练及其Adam状态。
- `policy_ema`：EMA完整state_dict，含精确的物理buffer。
- `policy_ema_updates`：actor更新次数，独立于WM更新和环境iteration。
- `policy_ema_decay`：衰减系数，恢复时检查一致。
- 原WM／critic／optimizer／sampler／robot contract字段全部保留。

新run可按原方式 `--resume logs/同run/ckpt_latest.pt`。旧160k的encoder是2048/1024/512/256，不能直接恢复到新encoder；尺寸不符会在加载模型与optimizer之前明确拒绝。旧策略继续按checkpoint内的原尺寸正常eval/play/export；未保存hidden尺寸的历史文件使用原四层fallback。

同架构旧checkpoint没有EMA时，从已经恢复的live权重初始化平均模型，计数从0开始；不能推算其过去的EMA。Python入口的 `--policy-trainer.ema-decay 0` 可关闭EMA；禁用时不保存EMA字段。启用EMA的恢复要求衰减相同。

## 指定用于推理的权重

公共CLI统一支持：

```text
--policy-weights auto   # 默认：EMA存在则用EMA，否则live
--policy-weights ema    # 明确要求EMA；缺失或损坏报错
--policy-weights live   # 明确用live，便于成对评估
```

覆盖eval_benchmark、play/play_pt、eval_push、diag_drift、record_rollout及export_onnx。加载器不把损坏的EMA静默当成legacy文件。基准manifest和ONNX/JSON记录选择来源、平均次数和decay。

示例：

```bash
.venv/bin/python scripts/export_onnx.py \
  --ckpt logs/khan_8452_residual_ema_enc1024_4096/ckpt_0100000.pt \
  --out exports/khan_8452_residual_ema_enc1024_4096/ckpt_0100000_ema.onnx \
  --policy-weights ema
```

ONNX已包含reference skip，C++继续使用 `q_target=q0+0.5*actions`，不重复叠加参考。新架构只影响ONNX内部权重，620→27协议和机器人参数不变。

## 验证

- 326项相关测试通过：EMA更新顺序、冻结shadow、buffer复制、恢复后下一步训练一致、旧文件兼容、尺寸拒绝、EMA/live选择、非有限EMA拒绝，以及ONNX packed/split动态batch的ORT一致性。
- 本地RTX4070 SUPER、8环境、完整8452紧凑库、50步GPU smoke通过；正式网络、WM batch2048×10、actor batch4096、H32及原优化设置保留。测试使用激活重算以适配12GB，服务器默认不启用。
- 40次WM optimizer、4次actor optimizer、80次critic optimizer；EMA恰好更新4次。各模型参数、optimizer和TB已记录标量有限，物理buffer精确相等；扰动后WM/live/critic、optimizer、sampler、EMA及计数均恢复。
- smoke采用Python既有delay开启配置，覆盖其队列路径；服务器脚本仍显式关闭delay。它是运行正确性验证，不是收敛或sim2real性能证明。
- 8环境×50步replay不足在线NSE8诊断的2048条门槛，终端的`nse8 nan`为未执行该额外诊断的占位；真实WM拟合记录`wm/nse_kstep`有限。
- 实际smoke完整网络的EMA ONNX导出并通过ORT验证，最大action误差1.49e-8；另用原160k真实checkpoint确认自动live fallback及历史四层结构仍正常。

本地验证产物：`logs/smoke_actor_ema_compact_encoder_20261110/summary.json`。logs、数据、导出模型和ref_thirdparty不属于本次提交。
